"""BestShots, этап 1: сканирование папки с фото/видео и запись метаданных в SQLite.

Исходные файлы только читаются — никаких изменений, перемещений и копирования.
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import power
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_EXT = {".mp4", ".mov"}

EVENT_GAP = timedelta(hours=3)
SERIES_GAP = timedelta(seconds=10)

# EXIF-теги
TAG_MAKE, TAG_MODEL, TAG_ORIENTATION = 0x010F, 0x0110, 0x0112
TAG_EXIF_IFD = 0x8769
TAG_EXPOSURE, TAG_FNUMBER, TAG_ISO = 0x829A, 0x829D, 0x8827
TAG_DT_ORIGINAL, TAG_FOCAL, TAG_FOCAL_35 = 0x9003, 0x920A, 0xA405

NAME_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])"
    r"(?:[-_ T.]?([01]\d|2[0-3])[-_.:]?([0-5]\d)[-_.:]?([0-5]\d)|(?!\d))"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,            -- photo | video
    size INTEGER NOT NULL,
    mtime REAL,
    sha256 TEXT,
    date_taken TEXT NOT NULL,      -- ISO 8601, локальное время без часового пояса
    date_source TEXT NOT NULL,     -- exif | filename | mtime
    width INTEGER,
    height INTEGER,
    orientation INTEGER,           -- EXIF Orientation (1..8) или NULL
    camera_model TEXT,
    exposure_time REAL,            -- секунды
    f_number REAL,
    iso INTEGER,
    focal_length REAL,             -- мм
    focal_length_35mm INTEGER,
    event_id INTEGER,
    series_id INTEGER,             -- NULL, если кадр не входит в серию
    read_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_date ON files(date_taken);
"""

COLUMNS = ["path", "kind", "size", "mtime", "sha256", "date_taken", "date_source", "width", "height",
           "orientation", "camera_model", "exposure_time", "f_number", "iso", "focal_length",
           "focal_length_35mm", "read_error"]


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def parse_exif_date(s):
    if not s:
        return None
    s = str(s).strip().replace("\x00", "")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
        return dt if dt.year > 1900 else None
    return None


def date_from_name(name):
    m = NAME_DATE_RE.search(name)
    if not m:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    try:
        return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0))
    except ValueError:
        return None


def read_image(path):
    """Возвращает словарь с метаданными изображения (ошибки не пробрасываются)."""
    info = {}
    try:
        with Image.open(path) as img:
            info["width"], info["height"] = img.size
            exif = img.getexif()
            sub = exif.get_ifd(TAG_EXIF_IFD) if exif else {}
    except Exception as e:  # битый файл / нет поддержки формата
        info["read_error"] = f"{type(e).__name__}: {e}"
        return info

    make = (exif.get(TAG_MAKE) or "").strip().strip("\x00")
    model = (exif.get(TAG_MODEL) or "").strip().strip("\x00")
    if model:
        brand = make.split()[0].lower() if make else ""
        info["camera_model"] = model if not brand or model.lower().startswith(brand)             else f"{make} {model}"
    info["orientation"] = int(exif.get(TAG_ORIENTATION) or 0) or None
    info["exif_date"] = parse_exif_date(sub.get(TAG_DT_ORIGINAL))
    info["exposure_time"] = to_float(sub.get(TAG_EXPOSURE))
    info["f_number"] = to_float(sub.get(TAG_FNUMBER))
    iso = sub.get(TAG_ISO)
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None
    info["iso"] = int(iso) if iso else None
    info["focal_length"] = to_float(sub.get(TAG_FOCAL))
    f35 = sub.get(TAG_FOCAL_35)
    info["focal_length_35mm"] = int(f35) if f35 else None
    return info


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_file(path, kind, known=None):
    """known — прежняя запись из БД (size, mtime, sha256): хэш пересчитывается, только если файл изменился."""
    st = path.stat()
    if known and known[0] == st.st_size and known[1] == st.st_mtime and known[2]:
        digest = known[2]
    else:
        digest = sha256_of(path)
    info = read_image(path) if kind == "photo" else {}
    if info.get("exif_date"):
        date, source = info["exif_date"], "exif"
    elif (d := date_from_name(path.name)):
        date, source = d, "filename"
    else:
        date, source = datetime.fromtimestamp(st.st_mtime).replace(microsecond=0), "mtime"
    return {
        "path": str(path), "kind": kind, "size": st.st_size, "mtime": st.st_mtime, "sha256": digest,
        "date_taken": date.isoformat(timespec="seconds"), "date_source": source,
        "width": info.get("width"), "height": info.get("height"),
        "orientation": info.get("orientation"), "camera_model": info.get("camera_model"),
        "exposure_time": info.get("exposure_time"), "f_number": info.get("f_number"),
        "iso": info.get("iso"), "focal_length": info.get("focal_length"),
        "focal_length_35mm": info.get("focal_length_35mm"),
        "read_error": info.get("read_error"),
    }


def is_hidden_dir(path, name):
    """Скрытые и системные папки (имя на «.» или «$», атрибуты Windows) пропускаем, как и в дереве папок."""
    if name == "_Дубли" or name.startswith((".", "$")) or name.lower() == "__macosx":  # _Дубли — карантин дубликатов
        return True
    try:
        return bool(getattr(os.stat(path, follow_symlinks=False), "st_file_attributes", 0) & 0x6)
    except OSError:
        return False


def under(path, base):
    return path == base or path.startswith(base.rstrip("\\/") + os.sep)


def iter_media(root, excluded=()):
    """Фото и видео внутри root; скрытые/системные и исключённые папки пропускаются."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not is_hidden_dir(os.path.join(dirpath, d), d)
                             and not any(under(os.path.join(dirpath, d), e) for e in excluded))
        for name in sorted(filenames):
            if name.startswith("._"):  # служебные файлы macOS («._IMG.JPG») — не изображения
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in PHOTO_EXT:
                yield Path(dirpath, name), "photo"
            elif ext in VIDEO_EXT:
                yield Path(dirpath, name), "video"


def assign_groups(rows):
    """rows — список dict, отсортированный по date_taken. Проставляет event_id и series_id."""
    event_id = series_id = 0
    prev = None
    for r in rows:
        dt = datetime.fromisoformat(r["date_taken"])
        if prev is None or dt - prev > EVENT_GAP:
            event_id += 1
        r["event_id"] = event_id
        prev = dt
        r["series_id"] = None
    # серии — только среди фото одного события
    last = None
    for r in rows:
        if r["kind"] != "photo":
            continue
        dt = datetime.fromisoformat(r["date_taken"])
        if last and last["event_id"] == r["event_id"] and \
                dt - datetime.fromisoformat(last["date_taken"]) <= SERIES_GAP:
            if last["series_id"] is None:
                series_id += 1
                last["series_id"] = series_id
            r["series_id"] = last["series_id"]
        last = r


def report(rows, root, events_limit):
    out = print
    photos = [r for r in rows if r["kind"] == "photo"]
    videos = [r for r in rows if r["kind"] == "video"]
    out(f"\n=== Отчёт: {root} ===")
    out(f"Найдено файлов: {len(rows)} (фото: {len(photos)}, видео: {len(videos)})")
    out(f"Общий размер: {sum(r['size'] for r in rows) / 1024 ** 2:.1f} МБ")

    out("\nПо годам:")
    for y, n in sorted(Counter(r["date_taken"][:4] for r in rows).items()):
        out(f"  {y}: {n}")

    src = Counter(r["date_source"] for r in rows)
    out("\nИсточник даты:")
    for s in ("exif", "filename", "mtime"):
        out(f"  {s}: {src.get(s, 0)}")
    no_exif = [r for r in rows if r["date_source"] != "exif"]
    out(f"Без даты в EXIF: {len(no_exif)} (из них видео: {sum(r['kind'] == 'video' for r in no_exif)})")

    cams = Counter(r["camera_model"] or "—" for r in photos)
    out("\nКамеры: " + ", ".join(f"{k} ({v})" for k, v in cams.most_common()))

    errors = [r for r in rows if r["read_error"]]
    if errors:
        out(f"\nНе удалось прочитать метаданные: {len(errors)}")
        for r in errors[:10]:
            out(f"  {r['path']}: {r['read_error']}")

    events = {}
    for r in rows:
        events.setdefault(r["event_id"], []).append(r)
    series_ids = {r["series_id"] for r in rows if r["series_id"]}
    in_series = sum(1 for r in rows if r["series_id"])
    out(f"\nСерии (интервал ≤ {SERIES_GAP.seconds} с): {len(series_ids)}, кадров в сериях: {in_series}")
    out(f"События (разрыв > {EVENT_GAP.seconds // 3600} ч): {len(events)}")
    for i, (eid, items) in enumerate(events.items()):
        if i >= events_limit:
            out(f"  ... ещё {len(events) - events_limit} (флаг --events-limit)")
            break
        a, b = items[0]["date_taken"], items[-1]["date_taken"]
        ns = len({r["series_id"] for r in items if r["series_id"]})
        out(f"  #{eid}: {a.replace('T', ' ')} — {b.replace('T', ' ')}  "
            f"файлов: {len(items)}, серий: {ns}")


def main():
    power.prefer_performance()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="BestShots: сканирование фото/видео в SQLite")
    ap.add_argument("source", type=Path, help="папка-источник")
    ap.add_argument("--db", type=Path, default=Path("bestshots.db"), help="файл базы (по умолч. bestshots.db)")
    ap.add_argument("--events-limit", type=int, default=30, help="сколько событий показать в отчёте")
    args = ap.parse_args()

    if not args.source.is_dir():
        sys.exit(f"Папка не найдена: {args.source}")
    args.source = args.source.resolve()
    if not HEIF_OK:
        print("Внимание: pillow-heif не установлен — HEIC будут без метаданных EXIF.", file=sys.stderr)

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(files)")}
    for col, typ in (("mtime", "REAL"), ("sha256", "TEXT"), ("orientation", "INTEGER")):
        if col not in have:  # миграция базы этапа 1
            con.execute(f"ALTER TABLE files ADD COLUMN {col} {typ}")
    known = {r["path"]: (r["size"], r["mtime"], r["sha256"]) for r in con.execute("SELECT * FROM files")}

    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    excluded = [r[0] for r in con.execute("SELECT path FROM excluded")] if "excluded" in tables else []
    media = list(iter_media(args.source, excluded))
    scanned = []
    for i, (p, k) in enumerate(media, 1):
        scanned.append(scan_file(p, k, known.get(str(p))))
        if i % 10 == 0 or i == len(media):  # для индикатора в интерфейсе (review.py)
            print("PROGRESS " + json.dumps({"stage": "scan", "done": i, "total": len(media)}), flush=True)
    seen = {r["path"] for r in scanned}
    root = str(args.source)  # удаляем из базы только отсутствующие файлы ВНУТРИ сканируемой папки
    stale = [p for p in known if under(p, root) and p not in seen]
    upd = ",".join(f"{c}=excluded.{c}" for c in COLUMNS if c != "path")
    with con:
        con.executemany(
            f"INSERT INTO files ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))}) "
            f"ON CONFLICT(path) DO UPDATE SET {upd}", [[r[c] for c in COLUMNS] for r in scanned])
        con.executemany("DELETE FROM files WHERE path=?", [(p,) for p in stale])
        # события и серии считаются по всей базе, чтобы нумерация не пересекалась между источниками
        rows = [dict(r) for r in con.execute("SELECT * FROM files ORDER BY date_taken, path")]
        assign_groups(rows)
        con.executemany("UPDATE files SET event_id=?, series_id=? WHERE path=?",
                        [(r["event_id"], r["series_id"], r["path"]) for r in rows])
        if "roots" in tables:  # время последнего сканирования корня
            stamp = datetime.now().isoformat(timespec="seconds")
            for (r,) in con.execute("SELECT path FROM roots").fetchall():
                if under(root, r) or under(r, root):
                    con.execute("UPDATE roots SET last_scan = ? WHERE path = ?", (stamp, r))
    con.close()
    rows = [r for r in rows if r["path"] in seen]

    report(rows, args.source, args.events_limit)
    print(f"\nБаза: {args.db.resolve()}")


if __name__ == "__main__":
    main()
