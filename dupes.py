"""BestShots: дубликаты — схема БД, выбор оригинала, запросы для интерфейса и карантин (перемещение в _Дубли).

Только стандартная библиотека. Оригиналы НИКОГДА не удаляются: дубли перемещаются в <главная папка>\\_Дубли
с сохранением структуры подпапок и могут быть возвращены по журналу. Окончательное удаление — только вручную.
"""
import csv
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime

import config

RAW_EXT = {".nef", ".cr2", ".arw", ".dng"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".avi"}
QUARANTINE = "_Дубли"
JOURNAL_NAME = "журнал.csv"

PH_MAX = config.DUP_PH_MAX
DH_MAX = config.DUP_DH_MAX
CLIP_MIN = config.DUP_CLIP_MIN
DATE_TOLERANCE_S = config.DUP_DATE_TOLERANCE_S
PIXEL_CORR_MIN = config.DUP_PIXEL_CORR_MIN
PIXEL_MAD_MAX = config.DUP_PIXEL_MAD_MAX

SCHEMA = """
CREATE TABLE IF NOT EXISTS dup_files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    dir TEXT, name TEXT, stem TEXT,        -- в нижнем регистре: для имён и пар RAW+JPEG
    size INTEGER NOT NULL, mtime REAL NOT NULL,
    kind TEXT NOT NULL,                    -- photo | raw | video
    sha256 TEXT,
    width INTEGER, height INTEGER, exif_date TEXT, has_exif INTEGER, orientation INTEGER,
    ph0 INTEGER, ph1 INTEGER, ph2 INTEGER, ph3 INTEGER,   -- pHash исходный и повёрнутый на 90/180/270
    dh0 INTEGER, dh1 INTEGER, dh2 INTEGER, dh3 INTEGER,
    tiny BLOB,                             -- серая копия 16×16 (256 байт) для пиксельной проверки
    emb BLOB,                              -- CLIP, если не было в analysis
    read_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_dup_files_size ON dup_files(size);
CREATE INDEX IF NOT EXISTS idx_dup_files_name ON dup_files(name);
CREATE TABLE IF NOT EXISTS dup_groups (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                    -- exact | visual
    orig_path TEXT NOT NULL,
    uncertain INTEGER NOT NULL DEFAULT 0,
    confirmed INTEGER NOT NULL DEFAULT 0,
    orig_manual INTEGER NOT NULL DEFAULT 0,  -- оригинал выбран пользователем вручную
    state TEXT NOT NULL DEFAULT 'open',    -- open | applied | ignored
    signature TEXT,
    n INTEGER, wasted INTEGER,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS dup_members (
    group_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (group_id, path)
);
CREATE INDEX IF NOT EXISTS idx_dup_members_path ON dup_members(path);
CREATE TABLE IF NOT EXISTS dup_checked (path TEXT PRIMARY KEY, at TEXT);
CREATE TABLE IF NOT EXISTS dup_prio (path TEXT PRIMARY KEY, prio INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS dup_moves (
    id INTEGER PRIMARY KEY,
    group_id INTEGER, orig_path TEXT, from_path TEXT, to_path TEXT, kind TEXT, size INTEGER,
    root TEXT, moved_at TEXT, restored_at TEXT
);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


def ensure_schema(con):
    con.executescript(SCHEMA)
    if "orig_manual" not in {r[1] for r in con.execute("PRAGMA table_info(dup_groups)")}:
        con.execute("ALTER TABLE dup_groups ADD COLUMN orig_manual INTEGER NOT NULL DEFAULT 0")  # миграция
        con.commit()
    con.executescript("""CREATE TABLE IF NOT EXISTS excluded (path TEXT PRIMARY KEY, at TEXT);
                         CREATE TABLE IF NOT EXISTS roots (path TEXT PRIMARY KEY, added_at TEXT, last_scan TEXT);""")


def under(path, base):
    return path == base or path.startswith(base.rstrip("\\/") + os.sep)


def prefix(folder):
    return folder.rstrip("\\/") + os.sep


def kind_of(name):
    ext = os.path.splitext(name)[1].lower()
    if ext in RAW_EXT:
        return "raw"
    if ext in PHOTO_EXT:
        return "photo"
    if ext in VIDEO_EXT:
        return "video"
    return None


def excluded_paths(con):
    try:
        return [r[0] for r in con.execute("SELECT path FROM excluded")]
    except sqlite3.OperationalError:
        return []


def skip_dir(path, name, excluded=()):
    """Скрытые, системные, исключённые папки и карантин _Дубли в поиске не участвуют."""
    if name == QUARANTINE or name.startswith((".", "$")):
        return True
    try:
        if getattr(os.stat(path), "st_file_attributes", 0) & 0x6:
            return True
    except OSError:
        return False
    return any(under(path, e) for e in excluded)


def is_raw_jpeg_pair(a, b):
    """Одинаковое имя без расширения в одной папке, один файл RAW, другой — обычное фото."""
    if os.path.dirname(a).lower() != os.path.dirname(b).lower():
        return False
    sa, ea = os.path.splitext(os.path.basename(a))
    sb, eb = os.path.splitext(os.path.basename(b))
    if sa.lower() != sb.lower():
        return False
    ea, eb = ea.lower(), eb.lower()
    return (ea in RAW_EXT and eb in PHOTO_EXT) or (eb in RAW_EXT and ea in PHOTO_EXT)


# ---------------------------------------------------------------- приоритет папок и выбор оригинала

def prio_map(con):
    return {p: v for p, v in con.execute("SELECT path, prio FROM dup_prio")}


def folder_prio(path, pm):
    """Приоритет ближайшей помеченной папки-предка: +1 «приоритетная», −1 «скорее копии», 0 — обычная."""
    d = os.path.dirname(path)
    while True:
        if d in pm:
            return pm[d]
        parent = os.path.dirname(d)
        if parent == d:
            return 0
        d = parent


def _date_key(m):
    if m.get("exif_date"):
        return m["exif_date"]
    return datetime.fromtimestamp(m["mtime"]).isoformat() if m.get("mtime") else "9999"


def original_key(m, pm):
    """Меньше — лучше. Порядок: разрешение → EXIF с датой → больший файл (исходный формат) →
    более ранняя дата → приоритет папки → более короткий путь."""
    return (-((m.get("width") or 0) * (m.get("height") or 0)), 0 if m.get("has_exif") else 1, -(m.get("size") or 0),
            _date_key(m), -folder_prio(m["path"], pm), len(m["path"]), m["path"])


def pick_original(members, pm):
    return min(members, key=lambda m: original_key(m, pm))


# ---------------------------------------------------------------- построение групп

def build_groups(con, folder, rows, exact_groups, confirmed, errors_total=0):
    """rows — все файлы области поиска; exact_groups — {sha: [строки]}; confirmed — [(a, b, rot, dph, ddh, cos)]."""
    by_path = {r["path"]: r for r in rows}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for members in exact_groups.values():
        first = members[0]["path"]
        for m in members[1:]:
            union(m["path"], first)
    links = []
    for a, b, rot, dph, ddh, cos, pcorr in confirmed:
        union(a["path"], b["path"])
        links.append((a["path"], dph, ddh, cos, pcorr))
    comps = {}
    for p in list(parent):
        comps.setdefault(find(p), []).append(p)
    link_by_root = {}
    for p, dph, ddh, cos, pcorr in links:
        link_by_root.setdefault(find(p), []).append((dph, ddh, cos, pcorr))

    pm = prio_map(con)
    old = {}
    for r in con.execute("SELECT * FROM dup_groups WHERE state != 'applied'"):
        old[r["signature"]] = r
    pref = prefix(folder)
    with con:
        ids = [r[0] for r in con.execute(
            "SELECT DISTINCT g.id FROM dup_groups g JOIN dup_members m ON m.group_id = g.id "
            "WHERE g.state != 'applied' AND substr(m.path, 1, ?) = ?", (len(pref), pref))]
        con.executemany("DELETE FROM dup_members WHERE group_id = ?", [(i,) for i in ids])
        con.executemany("DELETE FROM dup_groups WHERE id = ?", [(i,) for i in ids])
        for root, paths in comps.items():
            if len(paths) < 2:
                continue
            members = [by_path[p] for p in paths if p in by_path]
            if len(members) < 2:
                continue
            shas = {m.get("sha256") for m in members}
            kind = "exact" if len(shas) == 1 and None not in shas else "visual"
            uncertain = 0
            if kind == "visual":
                for dph, ddh, cos, pcorr in link_by_root.get(root, []):
                    if (dph >= config.DUP_PH_MAX - config.DUP_UNCERTAIN_MARGIN + 1
                            or ddh >= config.DUP_DH_MAX - config.DUP_UNCERTAIN_MARGIN + 1
                            or cos < config.DUP_CLIP_MIN + 0.03 or pcorr < config.DUP_PIXEL_CORR_MIN + 0.0007):
                        uncertain = 1
            orig = pick_original(members, pm)["path"]
            sig = hashlib.sha1("\n".join(sorted(paths)).encode("utf-8")).hexdigest()
            prev = old.get(sig)
            confirmed_flag, state, manual = 0, "open", 0
            if prev:  # выбор пользователя сохраняется, если состав группы не изменился
                confirmed_flag, state = prev["confirmed"], prev["state"]
                if prev["orig_manual"] and prev["orig_path"] in paths:
                    orig, manual = prev["orig_path"], 1
            wasted = sum(m["size"] for m in members if m["path"] != orig)
            cur = con.execute("INSERT INTO dup_groups (kind, orig_path, uncertain, confirmed, orig_manual, state, signature, n, "
                              "wasted, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (kind, orig, uncertain, confirmed_flag, manual, state, sig, len(members), wasted, now()))
            gid = cur.lastrowid
            con.executemany("INSERT INTO dup_members (group_id, path) VALUES (?,?)", [(gid, m["path"]) for m in members])


def mark_checked(con, folder):
    con.execute("INSERT OR REPLACE INTO dup_checked (path, at) VALUES (?, ?)", (folder, now()))
    con.commit()


def checked_at(con_rows, path):
    """Дата проверки: отметка самой папки или ближайшего предка (con_rows — словарь путь → дата)."""
    d = path
    while True:
        if d in con_rows:
            return con_rows[d]
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


# ---------------------------------------------------------------- запросы для интерфейса

def folder_where(folder, col="m.path"):
    if not folder:
        return "", ()
    p = prefix(folder)
    return f" AND substr({col}, 1, ?) = ?", (len(p), p)


def summary(con, folder=""):
    where, params = folder_where(folder)
    base = (f"FROM dup_groups g WHERE g.state = 'open' AND EXISTS (SELECT 1 FROM dup_members m WHERE m.group_id = g.id{where})")
    out = {}
    for kind in ("exact", "visual"):
        n, files, wasted = con.execute(
            f"SELECT COUNT(*), COALESCE(SUM(g.n - 1), 0), COALESCE(SUM(g.wasted), 0) {base} AND g.kind = ?",
            (*params, kind)).fetchone()
        out[kind] = {"groups": n, "files": files, "bytes": wasted}
    out["uncertain"] = con.execute(f"SELECT COUNT(*) {base} AND g.uncertain = 1", params).fetchone()[0]
    out["confirmed_visual"] = con.execute(f"SELECT COUNT(*) {base} AND g.kind = 'visual' AND g.confirmed = 1",
                                          params).fetchone()[0]
    fp, fparams = folder_where(folder, "path")
    out["samename"] = len(samename_groups(con, folder))
    out["rawjpeg"] = len(rawjpeg_pairs(con, folder))
    out["errors"] = con.execute(f"SELECT COUNT(*) FROM dup_files WHERE read_error IS NOT NULL{fp}", fparams).fetchone()[0]
    out["files_total"] = con.execute(f"SELECT COUNT(*) FROM dup_files WHERE 1=1{fp}", fparams).fetchone()[0]
    out["quarantined"] = con.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM dup_moves WHERE restored_at IS NULL").fetchone()
    out["quarantined"] = {"files": out["quarantined"][0], "bytes": out["quarantined"][1]}
    return out


def _member_rows(con, gid, orig):
    pm = prio_map(con)
    rows = con.execute("SELECT f.* FROM dup_members m JOIN dup_files f ON f.path = m.path WHERE m.group_id = ?", (gid,)).fetchall()
    out = []
    for f in rows:
        date = f["exif_date"] or (datetime.fromtimestamp(f["mtime"]).isoformat(timespec="seconds") if f["mtime"] else "")
        out.append({"id": f["id"], "path": f["path"], "name": os.path.basename(f["path"]), "size": f["size"],
                    "width": f["width"], "height": f["height"], "date": date, "has_exif": bool(f["has_exif"]),
                    "kind": f["kind"], "is_orig": f["path"] == orig, "prio": folder_prio(f["path"], pm)})
    out.sort(key=lambda m: (not m["is_orig"], m["path"]))
    return out


def list_groups(con, kind="", folder="", uncertain=False, offset=0, limit=20):
    where, params = folder_where(folder)
    cond = "g.state = 'open'"
    if kind in ("exact", "visual"):
        cond += f" AND g.kind = '{kind}'"
    if uncertain:
        cond += " AND g.uncertain = 1"
    q = (f"FROM dup_groups g WHERE {cond} AND EXISTS (SELECT 1 FROM dup_members m WHERE m.group_id = g.id{where})")
    total = con.execute(f"SELECT COUNT(*) {q}", params).fetchone()[0]
    rows = con.execute(f"SELECT g.* {q} ORDER BY g.kind, g.wasted DESC, g.id LIMIT ? OFFSET ?", (*params, limit, offset)).fetchall()
    groups = []
    for g in rows:
        groups.append({"id": g["id"], "kind": g["kind"], "uncertain": bool(g["uncertain"]), "confirmed": bool(g["confirmed"]),
                       "wasted": g["wasted"], "members": _member_rows(con, g["id"], g["orig_path"])})
    return {"total": total, "groups": groups}


def samename_groups(con, folder=""):
    """Одинаковые имена, но разное содержимое (только подсказка): камеры повторяют DSC_0001 и т. п."""
    where, params = folder_where(folder, "path")
    names = [r[0] for r in con.execute(
        f"SELECT name FROM dup_files WHERE kind IN ('photo','raw'){where} GROUP BY name HAVING COUNT(*) > 1", params)]
    group_of = {}
    for gid, path in con.execute("SELECT m.group_id, m.path FROM dup_members m JOIN dup_groups g ON g.id = m.group_id "
                                 "WHERE g.state != 'applied'"):
        group_of[path] = gid
    out = []
    for name in names:
        files = con.execute("SELECT path, size, width, height, sha256 FROM dup_files WHERE name = ? AND kind IN ('photo','raw')"
                            " AND read_error IS NULL", (name,)).fetchall()
        idents = {("g", group_of[f["path"]]) if f["path"] in group_of else ("s", f["sha256"] or f["path"]) for f in files}
        if len(idents) >= 2:
            out.append({"name": files[0]["path"].rsplit(os.sep, 1)[-1], "files": [
                {"path": f["path"], "size": f["size"], "width": f["width"], "height": f["height"]} for f in files]})
    out.sort(key=lambda g: g["name"].lower())
    return out


def rawjpeg_pairs(con, folder=""):
    where, params = folder_where(folder, "path")
    rows = con.execute(f"SELECT dir, stem, COUNT(*) c, SUM(kind = 'raw') r FROM dup_files WHERE kind IN ('photo','raw'){where} "
                       f"GROUP BY dir, stem HAVING r >= 1 AND c > r", params).fetchall()
    out = []
    for r in rows:
        files = con.execute("SELECT path, size, kind FROM dup_files WHERE dir = ? AND stem = ? AND kind IN ('photo','raw')",
                            (r["dir"], r["stem"])).fetchall()
        out.append({"files": [{"path": f["path"], "size": f["size"], "kind": f["kind"]} for f in files]})
    return out


def read_errors(con, folder="", limit=500):
    where, params = folder_where(folder, "path")
    return [{"path": r["path"], "error": r["read_error"]} for r in con.execute(
        f"SELECT path, read_error FROM dup_files WHERE read_error IS NOT NULL{where} LIMIT ?", (*params, limit))]


def journal(con, limit=300):
    rows = con.execute("SELECT * FROM dup_moves ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- карантин

def root_for(con, path):
    best = None
    try:
        for (r,) in con.execute("SELECT path FROM roots"):
            if under(path, r) and (best is None or len(r) > len(best)):
                best = r
    except sqlite3.OperationalError:
        pass
    return best or os.path.dirname(path)


def quarantine_path(root, path):
    rel = os.path.relpath(path, root)
    return os.path.join(root, QUARANTINE, rel)


def _unique(dst):
    if not os.path.exists(dst):
        return dst
    stem, ext = os.path.splitext(dst)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def _move_file(src, dst):
    """Перемещение: на том же диске — переименование (без копирования); иначе копирование с проверкой и удаление исходного."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        same = os.stat(src).st_dev == os.stat(os.path.dirname(dst)).st_dev
    except OSError:
        same = False
    if same:
        os.replace(src, dst)
        return
    shutil.copy2(src, dst)
    if os.path.getsize(dst) != os.path.getsize(src):
        os.remove(dst)
        raise OSError("размер копии не совпал, файл не перемещён")
    os.remove(src)


def _csv_log(root, rows):
    folder = os.path.join(root, QUARANTINE)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, JOURNAL_NAME)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if new:
            w.writerow(["Дата", "Действие", "Откуда", "Куда", "Оригинал", "Тип дубля", "Группа"])
        w.writerows(rows)


def plan_group_moves(con, group_ids):
    """Что будет перемещено: [(группа, kind, orig, путь, размер, куда)]. Оригинал остаётся на месте."""
    plan = []
    for gid in group_ids:
        g = con.execute("SELECT * FROM dup_groups WHERE id = ?", (gid,)).fetchone()
        if not g or g["state"] != "open":
            continue
        for (p,) in con.execute("SELECT path FROM dup_members WHERE group_id = ?", (gid,)):
            if p == g["orig_path"] or not os.path.exists(p):
                continue
            root = root_for(con, p)
            size = con.execute("SELECT size FROM dup_files WHERE path = ?", (p,)).fetchone()
            plan.append({"group": gid, "kind": g["kind"], "orig": g["orig_path"], "path": p,
                         "size": size[0] if size else os.path.getsize(p), "root": root, "dest": quarantine_path(root, p)})
    return plan


def apply_moves(con, group_ids):
    """Перемещает дубли выбранных групп в карантин. Возвращает отчёт; ошибки не прерывают работу."""
    plan = plan_group_moves(con, group_ids)
    report = {"moved": 0, "bytes": 0, "errors": [], "roots": sorted({p["root"] for p in plan})}
    csv_rows, touched = {}, set()
    for item in plan:
        dst = _unique(item["dest"])
        try:
            _move_file(item["path"], dst)
        except OSError as e:
            report["errors"].append(f"{item['path']}: {e}")
            continue
        stamp = now()
        con.execute("INSERT INTO dup_moves (group_id, orig_path, from_path, to_path, kind, size, root, moved_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (item["group"], item["orig"], item["path"], dst, item["kind"], item["size"], item["root"], stamp))
        # из рабочих таблиц убираем; решения и метки сохраняем и переносим на оригинал, если у него их нет
        p, o = item["path"], item["orig"]
        for table, cols in (("decisions", "path, keep, decided_at"), ("labels", "path, keep, at")):
            try:
                row = con.execute(f"SELECT keep FROM {table} WHERE path = ?", (p,)).fetchone()
                if row and not con.execute(f"SELECT 1 FROM {table} WHERE path = ?", (o,)).fetchone():
                    con.execute(f"INSERT INTO {table} ({cols}) VALUES (?,?,?)", (o, row[0], stamp))
            except sqlite3.OperationalError:
                pass
        for table in ("scores", "analysis", "files", "dup_files"):
            try:
                con.execute(f"DELETE FROM {table} WHERE path = ?", (p,))
            except sqlite3.OperationalError:
                pass
        con.execute("DELETE FROM dup_members WHERE group_id = ? AND path = ?", (item["group"], p))
        csv_rows.setdefault(item["root"], []).append(
            [stamp, "перемещён", p, dst, o, "точная копия" if item["kind"] == "exact" else "та же фотография", item["group"]])
        touched.add(item["group"])
        report["moved"] += 1
        report["bytes"] += item["size"]
    for gid in touched:
        con.execute("UPDATE dup_groups SET state = 'applied' WHERE id = ?", (gid,))
    for root in report["roots"]:  # карантин не сканируется и не анализируется во всех вкладках
        con.execute("INSERT OR IGNORE INTO excluded (path, at) VALUES (?, ?)", (os.path.join(root, QUARANTINE), now()))
    con.commit()
    for root, rows in csv_rows.items():
        try:
            _csv_log(root, rows)
        except OSError as e:
            report["errors"].append(f"журнал.csv: {e}")
    return report


def restore_moves(con, move_ids):
    """Возвращает файлы из карантина на прежние места. Существующие файлы не перезаписываются."""
    report = {"restored": 0, "conflicts": [], "missing": [], "errors": []}
    csv_rows = {}
    for mid in move_ids:
        m = con.execute("SELECT * FROM dup_moves WHERE id = ? AND restored_at IS NULL", (mid,)).fetchone()
        if not m:
            continue
        if not os.path.exists(m["to_path"]):
            report["missing"].append(m["to_path"])
            continue
        if os.path.exists(m["from_path"]):
            report["conflicts"].append(m["from_path"])  # на исходном месте уже что-то есть — не перезаписываем
            continue
        try:
            _move_file(m["to_path"], m["from_path"])
        except OSError as e:
            report["errors"].append(f"{m['to_path']}: {e}")
            continue
        stamp = now()
        con.execute("UPDATE dup_moves SET restored_at = ? WHERE id = ?", (stamp, mid))
        csv_rows.setdefault(m["root"], []).append([stamp, "возвращён", m["from_path"], m["to_path"], m["orig_path"],
                                                   "точная копия" if m["kind"] == "exact" else "та же фотография",
                                                   m["group_id"]])
        report["restored"] += 1
    con.commit()
    for root, rows in csv_rows.items():
        try:
            _csv_log(root, rows)
        except OSError as e:
            report["errors"].append(f"журнал.csv: {e}")
    return report


# ---------------------------------------------------------------- превью и просмотр (для интерфейса)

SOI = bytes([0xFF, 0xD8, 0xFF])  # начало JPEG


def embedded_jpeg(path):
    """Самое большое встроенное JPEG-превью RAW-файла (NEF/CR2/ARW/DNG содержат такие превью)."""
    import io
    from PIL import Image
    with open(path, "rb") as f:
        data = f.read()
    best, best_area, pos, tries = None, 0, data.find(SOI), 0
    while pos != -1 and tries < 12:
        tries += 1
        try:
            with Image.open(io.BytesIO(data[pos:])) as im:
                area = im.size[0] * im.size[1]
            if area > best_area:
                best, best_area = pos, area
        except Exception:
            pass
        pos = data.find(SOI, pos + 3)
    if best is None:
        raise ValueError("нет встроенного JPEG-превью")
    return data[best:]


def open_image(path):
    import io
    from PIL import Image
    if os.path.splitext(path)[1].lower() in RAW_EXT:
        return Image.open(io.BytesIO(embedded_jpeg(path)))
    return Image.open(path)


def render_jpeg(path, size, quality=85):
    """JPEG не крупнее size px с учётом EXIF Orientation (оригинал не меняется)."""
    import io
    from PIL import Image, ImageOps
    with open_image(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (size, size))
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()
