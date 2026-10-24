"""BestShots: папки — главная папка (корни), подсчёт фото на диске, статусы и дерево подпапок.

Только чтение имён папок и подсчёт файлов; никаких операций с самими файлами. Используется из review.py.
Статус папки считается по базе (files/scores, события «проверено» и экспортированные, ручные отметки),
число фото на диске — фоновым обходом с кэшем в таблице folder_cache.
"""
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path

import config

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic"}
QUARANTINE = "_Дубли"  # карантин дубликатов: не сканируется и не анализируется

SCHEMA = """
CREATE TABLE IF NOT EXISTS roots (
    path TEXT PRIMARY KEY,
    added_at TEXT,
    last_scan TEXT
);
CREATE TABLE IF NOT EXISTS folder_status (
    path TEXT PRIMARY KEY,
    done INTEGER NOT NULL,         -- 1 — вручную отмечена отработанной
    at TEXT
);
CREATE TABLE IF NOT EXISTS folder_cache (
    path TEXT PRIMARY KEY,
    direct_total INTEGER NOT NULL, -- фото прямо в папке (на диске)
    direct_new INTEGER NOT NULL,   -- из них новых/изменённых относительно базы
    accessible INTEGER NOT NULL,
    has_dirs INTEGER NOT NULL,
    scanned_at TEXT
);
CREATE TABLE IF NOT EXISTS excluded (
    path TEXT PRIMARY KEY,
    at TEXT
);
CREATE TABLE IF NOT EXISTS event_exports (
    event_id INTEGER PRIMARY KEY,
    at TEXT
);
"""

LOCK = threading.Lock()          # общий замок записи в БД (review.py использует его же)
VERSION = {"n": 0}               # растёт при любом изменении, влияющем на статусы папок
_STATS = {"ver": -1, "val": None}
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="folders")


def touch():
    VERSION["n"] += 1


def now():
    return datetime.now().isoformat(timespec="seconds")


def norm(p):
    return str(Path(p).resolve())


def under(path, base):
    """path совпадает с base или лежит внутри (по границе имени папки)."""
    return path == base or path.startswith(base.rstrip("\\/") + os.sep)


def prefix_of(folder):
    return folder.rstrip("\\/") + os.sep


def is_hidden(entry):
    """Скрытые и системные папки не показываем (имя на «.» или «$», атрибуты Windows HIDDEN/SYSTEM)."""
    if entry.name.startswith((".", "$")):
        return True
    try:
        attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & 0x6)


def natural_key(name):
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", name)]


# ---------------------------------------------------------------- корни и исключения

def roots(con):
    return [r[0] for r in con.execute("SELECT path FROM roots ORDER BY added_at, path")]


def root_of(con, path):
    best = None
    for r in roots(con):
        if under(path, r) and (best is None or len(r) > len(best)):
            best = r
    return best


def add_root(con, path):
    """Добавляет корень. Вложенную в существующий корень папку не дублирует; новый корень поглощает вложенные."""
    path = norm(path)
    for r in roots(con):
        if under(path, r):
            return {"path": r, "note": "inside"}
    for r in roots(con):
        if under(r, path):
            con.execute("DELETE FROM roots WHERE path = ?", (r,))
    con.execute("INSERT OR IGNORE INTO roots (path, added_at) VALUES (?, ?)", (path, now()))
    con.commit()
    touch()
    return {"path": path, "note": "added"}


def set_root(con, path):
    """Главная папка: единственный корень (интерфейс рассчитан на одну)."""
    con.execute("DELETE FROM roots")
    con.commit()
    return add_root(con, path)


def infer_root(con, hint=""):
    """Для базы, созданной до появления панели «Папки»: корень по подсказке или по общему пути файлов."""
    if roots(con):
        return None
    try:
        paths = [r[0] for r in con.execute("SELECT path FROM files")]
    except sqlite3.OperationalError:  # базы ещё нет
        return None
    if not paths:
        return None
    if hint and os.path.isdir(hint) and any(under(p, norm(hint)) for p in paths):
        return add_root(con, hint)["path"]
    try:
        common = os.path.commonpath([os.path.dirname(p) for p in paths])
    except ValueError:
        return None
    return add_root(con, common)["path"] if os.path.isdir(common) else None


def excluded(con):
    return [r[0] for r in con.execute("SELECT path FROM excluded")]


def is_excluded(path, ex):
    return any(under(path, e) for e in ex)


def touch_last_scan(con, folder):
    for r in roots(con):
        if under(folder, r) or under(r, folder):
            con.execute("UPDATE roots SET last_scan = ? WHERE path = ?", (now(), r))
    con.commit()


# ---------------------------------------------------------------- фоновый подсчёт фото на диске

_WALK_LOCK = threading.Lock()
WALK = {"running": False, "dirs": 0, "photos": 0, "started": 0.0, "finished": 0.0, "timed_out": False}


def start_walk(db_path):
    """Запускает фоновый обход. Если он уже идёт — просит повторить его после завершения (корни могли смениться)."""
    with _WALK_LOCK:
        if WALK["running"]:
            WALK["again"] = True
            return False
        WALK["running"] = True
    threading.Thread(target=_walk, args=(db_path,), daemon=True, name="folders-walk").start()
    return True


def _flush(con, batch):
    with LOCK:
        con.executemany("INSERT OR REPLACE INTO folder_cache (path, direct_total, direct_new, accessible, has_dirs, "
                        "scanned_at) VALUES (?,?,?,?,?,?)", batch)
        con.commit()


def _walk(db_path):
    while True:
        _walk_once(db_path)
        with _WALK_LOCK:
            if not WALK.pop("again", False):
                WALK.update(running=False, finished=time.time())
                touch()
                return


def _walk_once(db_path):
    WALK.update(running=True, dirs=0, photos=0, started=time.time(), finished=0.0, timed_out=False)
    con = None
    try:
        con = sqlite3.connect(db_path, timeout=15)
        known = {}
        try:
            known = {p: (s, m) for p, s, m in con.execute("SELECT path, size, mtime FROM files WHERE kind='photo'")}
        except sqlite3.OperationalError:
            pass
        ex = excluded(con)
        deadline = time.time() + config.FOLDER_WALK_TIMEOUT_S
        for root in roots(con):
            seen, batch, stack = set(), [], [root]
            while stack:
                if time.time() > deadline:
                    WALK["timed_out"] = True
                    break
                d = stack.pop()
                total = new = has_dirs = 0
                ok = 1
                try:
                    with os.scandir(d) as it:
                        for e in it:
                            try:
                                if e.is_dir(follow_symlinks=False):
                                    if is_hidden(e) or is_excluded(e.path, ex) or e.name == QUARANTINE:
                                        continue
                                    has_dirs = 1
                                    stack.append(e.path)
                                elif os.path.splitext(e.name)[1].lower() in PHOTO_EXT:
                                    st = e.stat()
                                    total += 1
                                    k = known.get(e.path)
                                    if not k or k[0] != st.st_size or abs((k[1] or 0) - st.st_mtime) > 1.0:
                                        new += 1
                            except OSError:
                                continue
                except OSError:
                    ok = 0
                seen.add(d)
                batch.append((d, total, new, ok, has_dirs, now()))
                WALK["dirs"] += 1
                WALK["photos"] += total
                if len(batch) >= 100:
                    _flush(con, batch)
                    batch = []
            if batch:
                _flush(con, batch)
            if not WALK["timed_out"]:  # устаревшие записи (папки удалены/исключены) убираем только после полного обхода
                stale = [r[0] for r in con.execute("SELECT path FROM folder_cache") if under(r[0], root) and r[0] not in seen]
                if stale:
                    with LOCK:
                        con.executemany("DELETE FROM folder_cache WHERE path = ?", [(p,) for p in stale])
                        con.commit()
    finally:
        if con:
            con.close()
        touch()


# ---------------------------------------------------------------- статусы

def compute_stats(con):
    """Сводка по всем известным папкам внутри корней. Кэшируется до следующего touch()."""
    ver = VERSION["n"]
    if _STATS["ver"] == ver and _STATS["val"] is not None:
        return _STATS["val"]
    rs = roots(con)
    def q(sql, default):  # таблиц может ещё не быть (пустая база) — каждый запрос отдельно
        try:
            return con.execute(sql).fetchall()
        except sqlite3.OperationalError:
            return default

    cache = {r[0]: r for r in q("SELECT path, direct_total, direct_new, accessible, has_dirs FROM folder_cache", [])}
    done_events = {r[0] for r in q("SELECT event_id FROM event_reviews WHERE reviewed = 1 "
                                   "UNION SELECT event_id FROM event_exports", [])}
    manual = {r[0] for r in q("SELECT path FROM folder_status WHERE done = 1", [])}
    dup_checked = {r[0]: r[1] for r in q("SELECT path, at FROM dup_checked", [])}
    dup_prio = {r[0]: r[1] for r in q("SELECT path, prio FROM dup_prio", [])}
    analyzed = q("SELECT f.path, s.event_id FROM files f JOIN scores s ON s.path = f.path WHERE f.kind = 'photo'", [])

    memo = {}

    def covered(d):  # папка или любой её родитель отмечены вручную
        if d in memo:
            return memo[d]
        x, res = d, False
        while True:
            if x in manual:
                res = True
                break
            parent = os.path.dirname(x)
            if parent == x:
                break
            x = parent
        memo[d] = res
        return res

    direct = defaultdict(lambda: [0, 0])  # папка -> [проанализировано, отработано]
    for path, event_id in analyzed:
        d = os.path.dirname(path)
        direct[d][0] += 1
        if event_id in done_events or covered(d):
            direct[d][1] += 1

    dirs = set()
    for d in list(cache) + list(direct) + list(manual):
        r = next((r for r in rs if under(d, r)), None)
        if not r:
            continue
        x = d
        while under(x, r):
            dirs.add(x)
            if x == r:
                break
            x = os.path.dirname(x)
    kids = defaultdict(list)
    for d in dirs:
        if d not in rs:
            kids[os.path.dirname(d)].append(d)

    nodes = {}
    for d in sorted(dirs, key=lambda x: -x.count(os.sep)):  # сначала самые глубокие
        c = cache.get(d)
        da, dd = direct.get(d, (0, 0))
        dtotal = max(c[1] if c else 0, da)
        ch = [nodes[k] for k in kids.get(d, [])]
        sub = [n for n in ch if n["total"] > 0]
        node = {"path": d, "name": os.path.basename(d) or d, "kids": [n["path"] for n in ch],
                "direct_total": dtotal, "direct_analyzed": da, "direct_done": dd,
                "total": dtotal + sum(n["total"] for n in ch),
                "analyzed": da + sum(n["analyzed"] for n in ch),
                "done": dd + sum(n["done"] for n in ch),
                "pending": (c[2] if c else 0) + sum(n["pending"] for n in ch),
                "accessible": bool(c[3]) if c else True, "has_dirs": bool(c[4]) if c else True,
                "manual": d in manual, "dup_checked": _nearest(d, dup_checked), "dup_prio": dup_prio.get(d, 0), "sub_total": len(sub), "sub_done": sum(1 for n in sub if n["green"]) if sub else 0}
        # непроанализированные новые файлы не мешают (о них говорит оранжевая точка), но в самой папке
        # должно быть проанализировано хотя бы что-то, если фото в ней есть
        direct_ok = da == dd and (da > 0 or dtotal == 0)
        node["green"] = node["manual"] or (node["analyzed"] > 0 and direct_ok and all(n["green"] for n in sub))
        nodes[d] = node
    val = {"nodes": nodes, "roots": rs,
           "all": sum(nodes[r]["total"] for r in rs if r in nodes),
           "done": sum(nodes[r]["done"] for r in rs if r in nodes)}
    _STATS.update(ver=ver, val=val)
    return val


def _nearest(path, mapping):
    """Значение для папки или ближайшего предка (например, дата проверки дублей)."""
    d = path
    while True:
        if d in mapping:
            return mapping[d]
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _job_overlaps(path, job):
    if not job or job.get("state") != "running" or job.get("kind", "analyze") != "analyze":
        return False
    return any(under(path, f) or under(f, path) for f in job.get("folders", []))


def public(node, job=None, excl=False):
    """JSON-представление узла дерева: статус (gray / blue / yellow / green), метки, признаки."""
    total, analyzed = node["total"], node["analyzed"]
    if excl:
        status = "excluded"
    elif _job_overlaps(node["path"], job) and total > 0 and (total > analyzed or node["pending"] > 0 or analyzed == 0):
        status = "blue"
    elif node["green"]:
        status = "green"
    elif analyzed > 0:
        status = "yellow"
    else:
        status = "gray"
    return {"name": node["name"], "path": node["path"], "total": total, "analyzed": analyzed, "done": node["done"],
            "pending": node["pending"], "status": status, "manual": node["manual"],
            "sub_total": node["sub_total"], "sub_done": node["sub_done"], "accessible": node["accessible"],
            "has_dirs": node["has_dirs"], "dup_checked": node.get("dup_checked"), "dup_prio": node.get("dup_prio", 0),
            "dot": analyzed > 0 and node["pending"] > 0 and not excl,
            "excluded": excl}


def _empty_node(path):
    return {"path": path, "name": os.path.basename(path) or path, "kids": [], "direct_total": 0, "direct_analyzed": 0,
            "direct_done": 0, "total": 0, "analyzed": 0, "done": 0, "pending": 0, "accessible": True,
            "has_dirs": True, "manual": False, "sub_total": 0, "sub_done": 0, "green": False, "dup_checked": None,
            "dup_prio": 0}


def list_children(con, path, job=None):
    """Подпапки path для дерева (подгрузка по требованию). Чтение имён с таймаутом для медленных дисков."""
    stats = compute_stats(con)
    ex = excluded(con)

    def scan():
        out = []
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False) and not is_hidden(e):
                        out.append(e.path)
                except OSError:
                    continue
        return out

    res = {"path": path, "children": [], "slow": False, "error": ""}
    try:
        entries = _POOL.submit(scan).result(timeout=config.FOLDER_LIST_TIMEOUT_S)
    except FutureTimeout:
        res["slow"] = True  # диск отвечает слишком медленно — показываем то, что есть в кэше
        entries = (stats["nodes"].get(path) or _empty_node(path))["kids"]
    except OSError as e:
        res["error"] = "Нет доступа к папке" if isinstance(e, PermissionError) else f"Папка недоступна: {e}"
        entries = []
    for p in sorted(entries, key=lambda x: natural_key(os.path.basename(x))):
        node = stats["nodes"].get(p) or _empty_node(p)
        res["children"].append(public(node, job, is_excluded(p, ex) or os.path.basename(p) == QUARANTINE))
    return res


def root_nodes(con, job=None):
    stats = compute_stats(con)
    ex = excluded(con)
    return [public(stats["nodes"].get(r) or _empty_node(r), job, is_excluded(r, ex)) for r in stats["roots"]]


def progress_totals(con):
    s = compute_stats(con)
    return {"all_photos": s["all"], "done_photos": s["done"]}


# ---------------------------------------------------------------- действия

def mark_done(con, path, done):
    if done:
        con.execute("INSERT INTO folder_status (path, done, at) VALUES (?,1,?) "
                    "ON CONFLICT(path) DO UPDATE SET done=1, at=excluded.at", (path, now()))
    else:
        con.execute("DELETE FROM folder_status WHERE path = ?", (path,))
    con.commit()
    touch()


def exclude_folder(con, path):
    """Исключает папку из анализа: файлы папки убираются из files/analysis/scores и не сканируются дальше.
    decisions, labels, event_reviews и сами оригиналы не затрагиваются."""
    pref = prefix_of(path)
    n = len(pref)
    con.execute("INSERT OR REPLACE INTO excluded (path, at) VALUES (?, ?)", (path, now()))
    removed = 0
    for table in ("scores", "analysis", "files"):
        cur = con.execute(f"DELETE FROM {table} WHERE substr(path, 1, ?) = ?", (n, pref))
        if table == "files":
            removed = cur.rowcount
    con.execute("DELETE FROM folder_cache WHERE path = ? OR substr(path, 1, ?) = ?", (path, n, pref))
    con.commit()
    touch()
    return removed


def include_folder(con, path):
    con.execute("DELETE FROM excluded WHERE path = ?", (path,))
    con.commit()
    touch()


def open_in_explorer(path):
    """Открывает папку в Проводнике (только просмотр)."""
    if sys.platform == "win32":
        subprocess.Popen(["explorer", os.path.normpath(path)])
    else:
        subprocess.Popen(["xdg-open", path])
