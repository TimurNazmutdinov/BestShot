"""BestShots, этап 3: локальный интерфейс финального отбора.

Сервер слушает только 127.0.0.1, всё работает офлайн. Исходные фото только читаются;
экспорт копирует файлы (shutil.copy2), ничего не перемещая.
"""
import argparse
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import about
import config
import dupes
import folders
import power
from folders import under
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
THUMBS_DIR = ROOT / "thumbs"
VIEW_SIZE = 2048
RAW_LIMIT = 8 * 1024 * 1024  # JPEG/PNG меньше этого размера и без поворота отдаём как есть

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    path TEXT PRIMARY KEY,         -- связь с files.path
    keep INTEGER,
    print INTEGER,                 -- устарело, не используется
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS labels (
    path TEXT PRIMARY KEY,
    keep INTEGER NOT NULL,
    at TEXT
);
CREATE TABLE IF NOT EXISTS event_reviews (
    event_id INTEGER PRIMARY KEY,
    reviewed INTEGER NOT NULL,
    at TEXT
);
"""

# итоговое решение = decisions, если есть, иначе рекомендация системы (scores.selected)
KEEP_SQL = "COALESCE(d.keep, s.selected)"
FROM_SQL = """FROM files f JOIN scores s ON s.path = f.path JOIN analysis a ON a.path = f.path
              LEFT JOIN decisions d ON d.path = f.path WHERE f.kind = 'photo'"""

SERVER = {"obj": None}
DB_PATH = "bestshots.db"
OUT_DIR = ROOT / "Export"
PORT = 8765
WRITE_LOCK = folders.LOCK


def db():
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    return con


def now():
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- данные

def data_ready(con):
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return {"scores", "analysis", "files"} <= names and con.execute("SELECT COUNT(*) FROM scores").fetchone()[0] > 0


def folder_clause(folder):
    """Условие «файл лежит в папке folder (с подпапками)»; пустая папка — без фильтра."""
    if not folder:
        return "", ()
    pref = folders.prefix_of(folder)
    return " AND substr(f.path, 1, ?) = ?", (len(pref), pref)


def summary(con, folder=""):
    empty = {"events": 0, "count": 0, "kept": 0, "reviewed": 0, "all_photos": 0, "done_photos": 0}
    if not data_ready(con):
        return {"events": [], "totals": empty, "ready": False}
    clause, params = folder_clause(folder)
    rows = con.execute(f"""
        SELECT s.event_id AS id, MIN(f.date_taken) AS start, MAX(f.date_taken) AS end, COUNT(*) AS count,
               SUM({KEEP_SQL}) AS kept, SUM(s.reject) AS rejects,
               COALESCE(r.reviewed, 0) AS reviewed, (x.event_id IS NOT NULL) AS exported
        FROM files f JOIN scores s ON s.path = f.path JOIN analysis a ON a.path = f.path
             LEFT JOIN decisions d ON d.path = f.path
             LEFT JOIN event_reviews r ON r.event_id = s.event_id
             LEFT JOIN event_exports x ON x.event_id = s.event_id
        WHERE f.kind = 'photo' {clause} GROUP BY s.event_id ORDER BY MIN(f.date_taken)""", params).fetchall()
    events = [dict(r) for r in rows]
    kept, count = con.execute(f"SELECT COALESCE(SUM({KEEP_SQL}), 0), COUNT(*) {FROM_SQL}").fetchone()
    totals = {"events": len(events), "count": count, "kept": kept,
              "reviewed": sum(1 for e in events if e["reviewed"])}
    totals.update(folders.progress_totals(con))
    if not totals["all_photos"]:  # папки ещё не посчитаны — по базе
        totals["all_photos"] = count
    return {"events": events, "totals": totals, "ready": True}


def chunk_groups(groups, size):
    """Делит группы на части примерно по size кадров; группу похожих не разрезаем. size = 0 — одна часть."""
    if not size:
        return [groups]
    parts, cur, n = [], [], 0
    for g in groups:
        if cur and n + len(g["photos"]) > size:
            parts.append(cur)
            cur, n = [], 0
        cur.append(g)
        n += len(g["photos"])
    if cur:
        parts.append(cur)
    return parts or [[]]


def event_data(con, event_id, folder="", part=1, size=0, flt="all", hide_reject=False):
    """Кадры события одной «частью» (для больших событий): фильтр применяется до разбивки."""
    clause, params = folder_clause(folder)
    rows = con.execute(f"""
        SELECT f.id, f.path, f.date_taken, f.camera_model, s.group_id, s.group_size, s.score, s.emotion,
               s.reject, s.reject_reason, s.selected AS best, {KEEP_SQL} AS keep,
               a.aesthetic, a.n_faces, a.smile, a.eyes_open, a.blink_frac, a.sharp_center,
               (a.thumb IS NOT NULL) AS has_thumb
        {FROM_SQL} AND s.event_id = ? {clause} ORDER BY f.date_taken, f.path""", (event_id, *params)).fetchall()
    groups, order = {}, []
    for r in rows:
        p = dict(r)
        p["name"] = Path(p.pop("path")).name
        if p["group_id"] not in groups:
            groups[p["group_id"]] = []
            order.append(p["group_id"])
        groups[p["group_id"]].append(p)
    out = []
    for gid in order:  # группы хронологически, лучший кадр группы первым
        ph = sorted(groups[gid], key=lambda p: (-p["best"], -p["score"]))
        if hide_reject:
            ph = [p for p in ph if not p["reject"]]
        if flt == "reject":
            ph = [p for p in ph if p["reject"]]
        elif flt == "keep":
            ph = [p for p in ph if p["keep"]]
        if ph:
            out.append({"id": gid, "photos": ph, "size": len(groups[gid])})
    parts = chunk_groups(out, size)
    part = max(1, min(part, len(parts)))
    return {"id": event_id, "groups": parts[part - 1], "part": part, "parts": len(parts), "size": size,
            "total": sum(len(g["photos"]) for g in out), "shown": sum(len(g["photos"]) for g in parts[part - 1])}


def file_row(con, file_id):
    return con.execute("SELECT f.*, a.thumb FROM files f LEFT JOIN analysis a ON a.path = f.path WHERE f.id = ?",
                       (file_id,)).fetchone()


def set_decision(con, file_id, keep):
    row = file_row(con, file_id)
    if not row:
        raise KeyError("нет такого файла")
    con.execute("INSERT INTO decisions (path, keep, decided_at) VALUES (?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET keep=excluded.keep, decided_at=excluded.decided_at",
                (row["path"], 1 if keep else 0, now()))
    con.commit()


def reset_event(con, event_id, folder=""):
    """Сбрасывает решения кадров события (при фильтре по папке — только её кадров) к рекомендации системы."""
    if folder:
        pref = folders.prefix_of(folder)
        con.execute("DELETE FROM decisions WHERE path IN (SELECT path FROM scores WHERE event_id = ? "
                    "AND substr(path, 1, ?) = ?)", (event_id, len(pref), pref))
    else:
        con.execute("DELETE FROM decisions WHERE path IN (SELECT path FROM scores WHERE event_id = ?)", (event_id,))
    con.commit()


def set_reviewed(con, event_id, value):
    con.execute("INSERT INTO event_reviews (event_id, reviewed, at) VALUES (?,?,?) "
                "ON CONFLICT(event_id) DO UPDATE SET reviewed=excluded.reviewed, at=excluded.at",
                (event_id, 1 if value else 0, now()))
    con.commit()
    folders.touch()  # статусы папок обновляются сразу


# ---------------------------------------------------------------- картинки

@lru_cache(maxsize=12)
def render_view(path, mtime):
    """JPEG до 2048 px с учётом EXIF Orientation (оригинал не меняется)."""
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (VIEW_SIZE, VIEW_SIZE))
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((VIEW_SIZE, VIEW_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def original_bytes(row):
    """(данные, content-type) для просмотра."""
    p = Path(row["path"])
    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg") and row["size"] <= RAW_LIMIT and row["orientation"] in (None, 1):
        return p.read_bytes(), "image/jpeg"
    return render_view(str(p), p.stat().st_mtime), "image/jpeg"


# ---------------------------------------------------------------- экспорт

# событие проверено, если пользователь отметил его «Проверено»: только такие события идут в экспорт и в обучение
REVIEWED_SQL = "EXISTS (SELECT 1 FROM event_reviews er WHERE er.event_id = s.event_id AND er.reviewed = 1)"


def export_rows(con, include_unreviewed=False):
    """Отобранные кадры для экспорта. По умолчанию — только из проверенных событий: рекомендации системы
    по событиям, которые пользователь не просматривал, не копируются."""
    cond = "" if include_unreviewed else f" AND {REVIEWED_SQL}"
    return con.execute(f"""SELECT f.path, f.size, f.date_taken, s.event_id, {REVIEWED_SQL} AS reviewed FROM files f
                           JOIN scores s ON s.path = f.path LEFT JOIN decisions d ON d.path = f.path
                           WHERE f.kind = 'photo' AND {KEEP_SQL} = 1{cond} ORDER BY f.date_taken, f.path""").fetchall()


def dest_dir(out, row):
    """Экспорт раскладывается только по годам съёмки: OUT\\ГГГГ\\имя_файла."""
    return out / row["date_taken"][:4]


def export_preview(con):
    rows = export_rows(con)
    everything = export_rows(con, True)
    events_total = con.execute("SELECT COUNT(DISTINCT event_id) FROM scores").fetchone()[0]
    events_reviewed = con.execute("SELECT COUNT(DISTINCT s.event_id) FROM scores s WHERE " + REVIEWED_SQL).fetchone()[0]
    return {"out": str(OUT_DIR), "count": len(rows), "bytes": sum(r["size"] for r in rows),
            "events_total": events_total, "events_reviewed": events_reviewed,
            "unreviewed_count": len(everything) - len(rows),
            "unreviewed_bytes": sum(r["size"] for r in everything) - sum(r["size"] for r in rows)}


def year_index(folder):
    """Файлы, уже лежащие в папке года (в том числе в подпапках прежней раскладки): имя -> [(размер, mtime)]."""
    index = {}
    if folder.is_dir():
        for dirpath, _dirs, names in os.walk(folder):
            for n in names:
                try:
                    st = os.stat(os.path.join(dirpath, n))
                except OSError:
                    continue
                index.setdefault(n, []).append((st.st_size, st.st_mtime))
    return index


def copy_one(src, folder, index):
    """Копирует src в folder. Возвращает 'copied' или 'skipped'. Чужие файлы не перезаписывает."""
    st = src.stat()
    if any(size == st.st_size and abs(mt - st.st_mtime) < 2 for size, mt in index.get(src.name, [])):
        return "skipped", 0  # такой файл уже экспортирован (в т. ч. по прежней раскладке «год\\событие»)
    folder.mkdir(parents=True, exist_ok=True)
    stem, suffix = src.stem, src.suffix
    n = 1
    while True:
        dst = folder / (src.name if n == 1 else f"{stem}_{n}{suffix}")
        if not dst.exists():
            shutil.copy2(src, dst)
            index.setdefault(src.name, []).append((st.st_size, st.st_mtime))
            return "copied", st.st_size
        ds = dst.stat()
        if ds.st_size == st.st_size and abs(ds.st_mtime - st.st_mtime) < 2:
            return "skipped", 0
        n += 1


def run_export(con, include_unreviewed=False):
    report = {"copied": 0, "skipped": 0, "bytes": 0, "errors": [], "out": str(OUT_DIR)}
    indexes, events = {}, set()
    for r in export_rows(con, include_unreviewed):
        year = dest_dir(OUT_DIR, r)
        if year not in indexes:
            indexes[year] = year_index(year)
        if r["reviewed"]:  # «экспортировано» (и значит «отработано») — только проверенные события
            events.add(r["event_id"])
        try:
            res, nbytes = copy_one(Path(r["path"]), year, indexes[year])
        except OSError as e:
            report["errors"].append(f"{r['path']}: {e}")
            continue
        report[res] += 1
        report["bytes"] += nbytes
    con.executemany("INSERT OR REPLACE INTO event_exports (event_id, at) VALUES (?, ?)", [(e, now()) for e in events])
    con.commit()
    folders.touch()  # события экспортированы — статусы папок обновляются
    return report


# ---------------------------------------------------------------- обучение на выборе пользователя

TRAIN = {"running": False, "text": "", "ok": None}


def save_labels(con, event_ids):
    """Финальное состояние (отобрано / нет) всех кадров событий — метки для обучения."""
    event_ids = list(event_ids)
    if not event_ids:
        return 0
    rows = con.execute(f"""SELECT s.path, {KEEP_SQL} AS keep FROM scores s LEFT JOIN decisions d ON d.path = s.path
                           WHERE s.event_id IN ({','.join('?' * len(event_ids))})""", event_ids).fetchall()
    con.executemany("INSERT INTO labels (path, keep, at) VALUES (?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET keep=excluded.keep, at=excluded.at",
                    [(r["path"], r["keep"], now()) for r in rows])
    con.commit()
    return len(rows)


def touched_events(con):
    """События, отмеченные пользователем «Проверено»: только по ним есть подтверждённый выбор, на нём и учимся.
    Непросмотренные события (где отобрана лишь рекомендация системы) метками не становятся."""
    return [r[0] for r in con.execute("SELECT DISTINCT s.event_id FROM scores s WHERE " + REVIEWED_SQL)]


def _train_worker():
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        p = subprocess.run([sys.executable, str(ROOT / "train.py"), "--db", str(DB_PATH), "--json"],
                           cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=flags, timeout=1800)
        lines = [l for l in p.stdout.splitlines() if not l.startswith("JSON:")]
        TRAIN["text"] = "\n".join(lines).strip() or (p.stderr.strip()[-500:] or "нет вывода")
        TRAIN["ok"] = p.returncode == 0
    except Exception as e:
        TRAIN["text"], TRAIN["ok"] = f"{type(e).__name__}: {e}", False
    finally:
        TRAIN["running"] = False


def start_training():
    if TRAIN["running"]:
        return False
    TRAIN.update(running=True, text="", ok=None)
    threading.Thread(target=_train_worker, daemon=True).start()
    return True


# ---------------------------------------------------------------- журнал, настройки, фоновый анализ

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "bestshots.log"
SETTINGS_FILE = Path(os.environ.get("BESTSHOTS_SETTINGS") or ROOT / "settings.json")  # переменная — для тестов
MODELS_DIR = ROOT / "models"
EXPECTED_MODELS_MB = 2900  # ориентир для индикатора первой загрузки моделей

log = logging.getLogger("bestshots")


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(**kv):
    data = load_settings()
    data.update(kv)
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


JOB = {"state": "idle", "kind": "analyze", "stage": "", "done": 0, "total": 0, "msg": "", "error": "", "folder": "", "folders": [],
       "eta": None}
JOB_LOCK = threading.Lock()
JOB_RUN = {"proc": None, "stop": False}


def child_python():
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":  # дочерним процессам нужен stdout
        exe = exe.with_name("python.exe")
    return str(exe)


def friendly_error(tail):
    text = "\n".join(tail)
    low = text.lower()
    if any(k in low for k in ("urlerror", "getaddrinfo", "max retries", "connectionerror", "timed out",
                              "name resolution", "httperror", "could not resolve", "offline mode")):
        return ("Не удалось скачать модели: нет подключения к интернету. Подключитесь к сети и нажмите "
                "«Анализировать» снова — загрузка нужна только один раз.")
    if "memoryerror" in low or "not enough memory" in low or "out of memory" in low:
        return "Не хватило оперативной памяти. Закройте другие программы и повторите."
    if "no space left" in low or "errno 28" in low:
        return "Не хватает места на диске."
    return "Не удалось завершить обработку."


def _stage_progress(msg):
    d = json.loads(msg)
    with JOB_LOCK:
        if d.get("stage") != JOB["stage"]:
            JOB["stage_started"] = time.time()
        JOB.update(stage=d.get("stage", ""), done=d.get("done", 0), total=d.get("total", 0), msg=d.get("msg", ""))
        el = time.time() - JOB.get("stage_started", time.time())
        JOB["eta"] = round(el / d["done"] * (d["total"] - d["done"])) if d.get("done") and d.get("total") else None


def run_step(args):
    """Запускает scan.py / analyze.py без окна консоли. Возвращает (код возврата, хвост вывода)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    log.info("запуск: %s", " ".join(map(str, args)))
    proc = subprocess.Popen([child_python(), "-u", *map(str, args)], cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            creationflags=flags)
    JOB_RUN["proc"] = proc
    tail = deque(maxlen=40)
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("PROGRESS "):
            try:
                _stage_progress(line[9:])
            except ValueError:
                pass
        elif line:
            log.info(line)
            tail.append(line)
    rc = proc.wait()
    JOB_RUN["proc"] = None
    return rc, list(tail)


def remaining_photos():
    con = db()
    try:
        return con.execute("SELECT COUNT(*) FROM files f WHERE f.kind = 'photo' AND NOT EXISTS "
                           "(SELECT 1 FROM analysis a WHERE a.path = f.path)").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        con.close()


def _job_worker(steps, auto=False):
    try:
        queue = list(steps)
        left = None
        while queue:
            args = queue.pop(0)
            if args[0] == "analyze.py" and "--score-only" not in args:
                with JOB_LOCK:
                    JOB["part_no"] = JOB.get("part_no", 0) + 1
            rc, tail = run_step(args)
            if JOB_RUN["stop"]:
                if args[0] == "analyze.py" and "--score-only" not in args:
                    # анализ прервали: баллы считаются в конце, поэтому досчитываем их по уже обработанным кадрам
                    with JOB_LOCK:
                        JOB.update(stage="score", done=0, total=0, msg="Остановлено. Подготовка результатов по обработанным кадрам…")
                    JOB_RUN["stop"] = False
                    run_step(["analyze.py", "--db", DB_PATH, "--score-only"])
                with JOB_LOCK:
                    JOB.update(state="stopped", error="")
                log.info("остановлено пользователем")
                return
            if rc != 0:
                with JOB_LOCK:
                    JOB.update(state="error", error=friendly_error(tail))
                log.error("шаг %s завершился с кодом %s", args[0], rc)
                return
            if auto and args[0] == "analyze.py" and not queue:
                # порция готова: если ещё остались непроанализированные фото и работа продвигается — берём следующую
                now_left = remaining_photos()
                if now_left > 0 and (left is None or now_left < left):
                    left = now_left
                    queue.append(args)
                    with JOB_LOCK:
                        JOB.update(stage="analyze", done=0, total=0, msg="")
        with JOB_LOCK:
            JOB.update(state="done", error="")
        log.info("готово")
    except Exception as e:
        log.exception("сбой фоновой обработки")
        with JOB_LOCK:
            JOB.update(state="error", error=f"Не удалось запустить обработку: {type(e).__name__}: {e}")
    finally:
        JOB_RUN["proc"] = None
        folders.touch()
        folders.start_walk(DB_PATH)  # пересчитать фото на диске и признак «есть новые файлы»


def validate_folders(folder_list):
    """Папки должны лежать внутри главной папки. Возвращает (список, ошибка)."""
    con = db()
    try:
        rs = folders.roots(con)
    finally:
        con.close()
    if not rs:
        return None, "Сначала выберите главную папку с фото."
    chosen = []
    for f in folder_list or rs:
        if not f or not Path(f).is_dir():
            return None, "Папка не найдена. Проверьте путь."
        n = folders.norm(f)
        if not any(under(n, r) for r in rs):
            return None, "Папка вне главной папки."
        chosen.append(n)
    return [f for f in chosen if not any(f != g and under(f, g) for g in chosen)], None  # вложенные не дублируем


def _begin_job(kind, chosen, steps, auto=False):
    with JOB_LOCK:
        if JOB["state"] == "running":
            return {"error": "Обработка уже идёт."}
        JOB.update(state="running", kind=kind, stage="scan", done=0, total=0, msg="", error="", folder=chosen[0],
                   folders=chosen, eta=None, stage_started=time.time(), part_no=0)
    JOB_RUN["stop"] = False
    power.keep_awake_while(lambda: JOB["state"] == "running")  # длинное задание не должно прерываться сном
    threading.Thread(target=_job_worker, args=(steps, auto), daemon=True).start()
    return {"ok": True}


def start_job(folder_list, limit=0, auto=False):
    """Анализ указанных папок (пусто — все корни). limit > 0 — порция: не больше limit новых фото за запуск,
    после каждой порции баллы и события пересчитываются, результат сразу виден во вкладке «Отбор»."""
    chosen, err = validate_folders(folder_list)
    if err:
        return {"error": err}
    steps = [["scan.py", f, "--db", DB_PATH, "--events-limit", "0"] for f in chosen]
    analyze = ["analyze.py", "--db", DB_PATH] + (["--limit", str(int(limit))] if limit and int(limit) > 0 else [])
    return _begin_job("analyze", chosen, steps + [analyze], auto=bool(auto and limit and int(limit) > 0))


def start_dupes(folder):
    """Поиск дубликатов в папке (по умолчанию — вся главная папка); повторный запуск продолжает с кэша."""
    chosen, err = validate_folders([folder] if folder else [])
    if err:
        return {"error": err}
    return _begin_job("dupes", chosen, [["dupes_scan.py", f, "--db", DB_PATH] for f in chosen])


def stop_job():
    JOB_RUN["stop"] = True
    proc = JOB_RUN.get("proc")
    if proc and proc.poll() is None:
        proc.kill()  # прогресс уже сохранён пакетами в БД, при следующем запуске продолжится с места остановки
    return {"ok": True}


def job_snapshot():
    with JOB_LOCK:
        return dict(JOB)


def job_status():
    with JOB_LOCK:
        d = dict(JOB)
    if d["state"] == "running" and d["stage"] == "download":
        size = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file()) if MODELS_DIR.exists() else 0
        d["models_mb"] = round(size / 1048576)
        d["models_total_mb"] = EXPECTED_MODELS_MB
    d["log"] = str(LOG_FILE)
    return d


def choose_folder():
    """Системный диалог выбора папки (tkinter в отдельном процессе — работает и в окне, и в браузере)."""
    code = ("import tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk()\nr.withdraw()\n"
            "r.attributes('-topmost', True)\nprint(filedialog.askdirectory(title='Папка с фото'))\n")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    p = subprocess.run([child_python(), "-c", code], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, creationflags=flags, timeout=600)
    return {"path": p.stdout.strip().replace("/", os.sep)}


# ---------------------------------------------------------------- API папок

def ui_state():
    return load_settings().get("ui", {})


def roots_info(con):
    rs = folders.roots(con)
    return {"roots": rs, "main": rs[0] if rs else "", "ui": ui_state(), "walk": dict(folders.WALK),
            "choices": {"analyze": config.ANALYZE_PART_CHOICES, "review": config.REVIEW_PART_CHOICES,
                        "review_default": config.REVIEW_PART_SIZE},
            "ready": data_ready(con)}


def folders_query(data):
    """Дерево: подпапки запрошенных узлов + сводка корней. Только чтение имён папок и подсчёт."""
    con = db()
    try:
        rs = folders.roots(con)
        job = job_snapshot()
        nodes = {}
        for p in data.get("paths", []):
            try:
                resolved = folders.norm(p)
            except (OSError, ValueError):
                continue
            if any(under(resolved, r) for r in rs):  # читаем только внутри главной папки
                nodes[p] = folders.list_children(con, resolved, job)
        out = {"nodes": nodes, "roots": folders.root_nodes(con, job), "walk": dict(folders.WALK), "job": job.get("state")}
        try:
            out["to_analyze"] = con.execute("SELECT COUNT(*) FROM files f WHERE f.kind = 'photo' AND NOT EXISTS "
                                            "(SELECT 1 FROM analysis a WHERE a.path = f.path)").fetchone()[0]
        except sqlite3.OperationalError:
            out["to_analyze"] = 0
        out.update(folders.progress_totals(con))
        return out
    finally:
        con.close()


@lru_cache(maxsize=300)
def dup_render(path, mtime, size_px):
    return dupes.render_jpeg(path, size_px, 82 if size_px <= 400 else 90)


def dup_scope_ids(con, scope, data):
    """Группы для массовых действий: все точные копии / все подтверждённые визуальные / одна группа."""
    if scope == "group":
        return [int(g) for g in (data.get("group_ids") or [data.get("group_id")]) if g is not None]
    where, params = dupes.folder_where(data.get("view_folder") or "")
    kind = {"exact_all": "g.kind = 'exact'",
            "visual_confident": "g.kind = 'visual' AND g.uncertain = 0",  # все уверенные визуальные группы
            }.get(scope, "g.kind = 'visual' AND g.confirmed = 1")
    return [r[0] for r in con.execute(
        f"SELECT g.id FROM dup_groups g WHERE g.state = 'open' AND {kind} AND EXISTS "
        f"(SELECT 1 FROM dup_members m WHERE m.group_id = g.id{where})", params)]


def check_in_roots(con, path):
    resolved = folders.norm(path)
    if not any(under(resolved, r) for r in folders.roots(con)):
        raise PermissionError("Папка вне главной папки")
    return resolved


# ---------------------------------------------------------------- HTTP

class Server(ThreadingHTTPServer):
    allow_reuse_address = False  # на Windows иначе два сервера могут занять один порт
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "BestShotsReview"

    def log_message(self, *args):
        pass

    # --- вспомогательное
    def send_bytes(self, code, data, ctype, cache=False):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionError, OSError):
            pass

    def send_json(self, obj, code=200):
        self.send_bytes(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def host_ok(self):
        allowed = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
        if self.headers.get("Host") not in allowed:  # защита от DNS rebinding
            self.send_json({"error": "bad host"}, 403)
            return False
        return True

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    # --- GET
    def do_GET(self):
        if not self.host_ok():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/":
                return self.send_bytes(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            con = db()
            try:
                query = parse_qs(urlsplit(self.path).query)
                view = (query.get("folder") or [""])[0]
                if path == "/api/events":
                    return self.send_json(summary(con, view))
                m = re.fullmatch(r"/api/event/(\d+)", path)
                if m:
                    return self.send_json(event_data(
                        con, int(m[1]), view, int((query.get("part") or ["1"])[0]), int((query.get("size") or ["0"])[0]),
                        (query.get("filter") or ["all"])[0], (query.get("hide") or ["0"])[0] == "1"))
                if path == "/api/roots":
                    return self.send_json(roots_info(con))
                if path == "/api/about":
                    rs = folders.roots(con)
                    return self.send_json(about.info(root=ROOT, db_path=DB_PATH, main_root=rs[0] if rs else "", port=PORT,
                                                     out_dir=OUT_DIR, log_file=LOG_FILE))
                if path == "/api/about/sizes":
                    return self.send_json(about.sizes_state())
                if path == "/api/dupes/summary":
                    return self.send_json(dupes.summary(con, view))
                if path == "/api/dupes/groups":
                    return self.send_json(dupes.list_groups(
                        con, (query.get("kind") or [""])[0], view, (query.get("uncertain") or ["0"])[0] == "1",
                        int((query.get("offset") or ["0"])[0]), int((query.get("limit") or ["20"])[0])))
                if path == "/api/dupes/extra":
                    kind = (query.get("kind") or [""])[0]
                    if kind == "samename":
                        return self.send_json({"items": dupes.samename_groups(con, view)})
                    if kind == "rawjpeg":
                        return self.send_json({"items": dupes.rawjpeg_pairs(con, view)})
                    return self.send_json({"items": dupes.read_errors(con, view)})
                if path == "/api/dupes/journal":
                    return self.send_json({"items": dupes.journal(con)})
                m = re.fullmatch(r"/(dthumb|dphoto)/(\d+)", path)
                if m:
                    row = con.execute("SELECT path, mtime FROM dup_files WHERE id = ?", (int(m[2]),)).fetchone()
                    if not row:
                        return self.send_json({"error": "not found"}, 404)
                    return self.send_bytes(200, dup_render(row["path"], row["mtime"], 360 if m[1] == "dthumb" else 2048),
                                           "image/jpeg", cache=True)
                if path == "/api/job":
                    return self.send_json(job_status())
                if path == "/api/settings":
                    return self.send_json({"ready": data_ready(con), "ui": ui_state()})
                if path == "/api/train/status":
                    return self.send_json(TRAIN)
                if path == "/api/export/preview":
                    return self.send_json(export_preview(con))
                m = re.fullmatch(r"/(thumb|photo)/(\d+)", path)
                if m:
                    return self.serve_image(con, m[1], int(m[2]))
            finally:
                con.close()
            self.send_json({"error": "not found"}, 404)
        except Exception as e:  # ответ об ошибке вместо обрыва соединения
            log.exception("ошибка GET %s", self.path)
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def serve_image(self, con, kind, file_id):
        row = file_row(con, file_id)  # отдаём только файлы из таблицы files
        if not row:
            return self.send_json({"error": "not found"}, 404)
        if kind == "thumb" and row["thumb"]:
            t = (ROOT / row["thumb"]).resolve()
            if THUMBS_DIR.resolve() in t.parents and t.is_file():
                return self.send_bytes(200, t.read_bytes(), "image/jpeg", cache=True)
        data, ctype = original_bytes(row)
        self.send_bytes(200, data, ctype, cache=True)

    # --- POST
    def do_POST(self):
        if not self.host_ok():
            return
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
            return self.send_json({"error": "bad origin"}, 403)
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.send_json({"error": "json required"}, 415)
        path = urlsplit(self.path).path
        try:
            data = self.body()
            view = data.get("view_folder") or ""
            # операции без записи в БД (и долгие, как системный диалог) — вне замка записи
            if path == "/api/folders/query":
                return self.send_json(folders_query(data))
            if path == "/api/choose-folder":
                return self.send_json(choose_folder())
            if path == "/api/shutdown":
                self.send_json({"ok": True})
                threading.Thread(target=SERVER["obj"].shutdown, daemon=True).start()
                return
            if path == "/api/job/stop":
                return self.send_json(stop_job())
            if path == "/api/job/start":
                one = (data.get("folder") or "").strip().strip('"')
                res = start_job([one] if one else [], int(data.get("limit") or 0), bool(data.get("auto")))
                return self.send_json(res, 400 if "error" in res else 200)
            if path == "/api/dupes/find":
                res = start_dupes((data.get("folder") or "").strip().strip('"'))
                return self.send_json(res, 400 if "error" in res else 200)
            if path == "/api/about/open":  # только папка программы и папка журналов
                folders.open_in_explorer(str(ROOT if data.get("what") == "app" else LOG_DIR))
                return self.send_json({"ok": True})
            if path == "/api/folders/refresh":
                return self.send_json({"started": folders.start_walk(DB_PATH)})
            if path == "/api/ui-state":
                ui = dict(ui_state())
                ui.update(data.get("ui") or {})
                save_settings(ui=ui)
                return self.send_json({"ok": True})
            if path == "/api/folders/open":
                con = db()
                try:
                    folders.open_in_explorer(check_in_roots(con, data.get("path") or ""))
                finally:
                    con.close()
                return self.send_json({"ok": True})
            with WRITE_LOCK:
                con = db()
                try:
                    if path == "/api/decision":
                        set_decision(con, int(data["id"]), data.get("keep"))
                    elif (m := re.fullmatch(r"/api/event/(\d+)/reset", path)):
                        reset_event(con, int(m[1]), view)
                    elif (m := re.fullmatch(r"/api/event/(\d+)/reviewed", path)):
                        set_reviewed(con, int(m[1]), data.get("reviewed"))
                    elif path == "/api/export":
                        report = run_export(con, bool(data.get("include_unreviewed")))
                        # метки для обучения — только из проверенных событий (рекомендации системы в метки не попадают)
                        report["labels"] = save_labels(con, touched_events(con))
                        report["training"] = start_training() if report["labels"] else False
                        return self.send_json(report)
                    elif path == "/api/train":
                        report = {"labels": save_labels(con, touched_events(con))}
                        if not report["labels"]:
                            report.update(started=False, note="Нет проверенных событий: отметьте события «Проверено», тогда по ним можно обучаться.")
                            return self.send_json(report)
                        report["started"] = start_training()
                        return self.send_json(report)
                    elif path == "/api/roots/set" or path == "/api/roots/add":
                        target = (data.get("path") or "").strip().strip('"')
                        if not target or not os.path.isdir(target):
                            return self.send_json({"error": "Папка не найдена. Проверьте путь."}, 400)
                        res = folders.set_root(con, target) if path.endswith("/set") else folders.add_root(con, target)
                        save_settings(root=res["path"])
                        folders.start_walk(DB_PATH)
                        out = roots_info(con)
                        out["note"] = res["note"]
                        return self.send_json(out)
                    elif path == "/api/dupes/set-original":
                        g = con.execute("SELECT * FROM dup_groups WHERE id = ?", (int(data["group_id"]),)).fetchone()
                        member = con.execute("SELECT f.size FROM dup_members m JOIN dup_files f ON f.path = m.path "
                                             "WHERE m.group_id = ? AND m.path = ?", (g["id"], data["path"])).fetchone() if g else None
                        if not member:
                            return self.send_json({"error": "Файл не входит в группу"}, 404)
                        total = con.execute("SELECT COALESCE(SUM(f.size), 0) FROM dup_members m JOIN dup_files f ON f.path = m.path "
                                            "WHERE m.group_id = ?", (g["id"],)).fetchone()[0]
                        con.execute("UPDATE dup_groups SET orig_path = ?, orig_manual = 1, wasted = ? WHERE id = ?", (data["path"], total - member[0], g["id"]))
                        con.commit()
                        return self.send_json({"ok": True})
                    elif path == "/api/dupes/confirm":
                        con.execute("UPDATE dup_groups SET confirmed = ? WHERE id = ?", (1 if data.get("confirmed") else 0, int(data["group_id"])))
                        con.commit()
                        return self.send_json({"ok": True})
                    elif path == "/api/dupes/ignore":
                        con.execute("UPDATE dup_groups SET state = 'ignored' WHERE id = ?", (int(data["group_id"]),))
                        con.commit()
                        return self.send_json({"ok": True})
                    elif path == "/api/dupes/apply":
                        ids = dup_scope_ids(con, data.get("scope") or "group", data)
                        plan = dupes.plan_group_moves(con, ids)
                        if data.get("dry_run"):
                            return self.send_json({"groups": len({p["group"] for p in plan}), "files": len(plan),
                                                   "bytes": sum(p["size"] for p in plan),
                                                   "dest": sorted({os.path.join(p["root"], dupes.QUARANTINE) for p in plan})})
                        report = dupes.apply_moves(con, ids)
                        folders.touch()
                        folders.start_walk(DB_PATH)
                        return self.send_json(report)
                    elif path == "/api/dupes/restore":
                        scope = data.get("scope") or "ids"
                        if scope == "all":
                            ids = [r[0] for r in con.execute("SELECT id FROM dup_moves WHERE restored_at IS NULL")]
                        elif scope == "group":
                            ids = [r[0] for r in con.execute("SELECT id FROM dup_moves WHERE restored_at IS NULL AND group_id = ?",
                                                              (int(data["group_id"]),))]
                        else:
                            ids = [int(i) for i in data.get("ids") or []]
                        report = dupes.restore_moves(con, ids)
                        folders.touch()
                        folders.start_walk(DB_PATH)
                        return self.send_json(report)
                    elif path == "/api/folders/dupe-prio":
                        target = check_in_roots(con, data.get("path") or "")
                        prio = int(data.get("prio") or 0)
                        if prio:
                            con.execute("INSERT OR REPLACE INTO dup_prio (path, prio) VALUES (?, ?)", (target, prio))
                        else:
                            con.execute("DELETE FROM dup_prio WHERE path = ?", (target,))
                        con.commit()
                        folders.touch()
                        return self.send_json({"ok": True})
                    elif path == "/api/folders/mark":
                        folders.mark_done(con, check_in_roots(con, data.get("path") or ""), bool(data.get("done")))
                        return self.send_json({"ok": True})
                    elif path == "/api/folders/exclude":
                        target = check_in_roots(con, data.get("path") or "")
                        if data.get("excluded", True):
                            removed = folders.exclude_folder(con, target)
                            return self.send_json({"ok": True, "removed": removed})
                        folders.include_folder(con, target)
                        return self.send_json({"ok": True})
                    else:
                        return self.send_json({"error": "not found"}, 404)
                    return self.send_json(summary(con, view))
                finally:
                    con.close()
        except PermissionError as e:
            self.send_json({"error": str(e)}, 403)
        except KeyError as e:
            self.send_json({"error": str(e)}, 404)
        except Exception as e:
            log.exception("ошибка POST %s", self.path)
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


# ---------------------------------------------------------------- страница

PAGE = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BestShots — отбор</title>
<style>
:root{--bg:#f4f4f2;--panel:#fff;--text:#1d1d1b;--muted:#6b6b66;--line:#d9d9d4;--accent:#2563eb;--keep:#1f9d55;--bad:#c62828;--chip:#ececE8;--yellow:#e0a800;--orange:#f57c00;--gray:#9a9a94}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--panel:#222;--text:#ececec;--muted:#9a9a94;--line:#3a3a3a;--accent:#6ea0ff;--keep:#3ec878;--bad:#ef6b6b;--chip:#333;--yellow:#f0c040;--orange:#ff9a3c;--gray:#7a7a74}}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--text);font:14px/1.4 system-ui,"Segoe UI",sans-serif;display:flex;flex-direction:column}
button{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}
button:disabled{opacity:.45;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
header{display:flex;gap:16px;align-items:center;padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:16px;margin:0}
#sizes{display:flex}
#sizes button{border-radius:0;padding:5px 12px}
#sizes button:first-child{border-radius:6px 0 0 6px}
#sizes button:last-child{border-radius:0 6px 6px 0}
.totals{display:flex;gap:14px;color:var(--muted);flex:1;flex-wrap:wrap}
.totals b{color:var(--text)}
main{flex:1;display:flex;min-height:0}
/* левая колонка: «Папки» и «События» */
#side{width:var(--side-w,320px);min-width:200px;max-width:70vw;flex:none;display:flex;flex-direction:column;background:var(--panel);min-height:0}
#vsplit{width:6px;flex:none;cursor:col-resize;background:var(--line);opacity:.5}
#vsplit:hover,#hsplit:hover{background:var(--accent);opacity:1}
#hsplit{height:6px;flex:none;cursor:row-resize;background:var(--line);opacity:.5}
.panel{display:flex;flex-direction:column;min-height:0}
#pfolders{flex:0 0 var(--fp-h,330px)}
#pevents{flex:1 1 0}
.panel.collapsed{flex:0 0 auto!important}
.panel.collapsed .pbody{display:none}
.phead{display:flex;align-items:center;gap:6px;padding:5px 8px;border-bottom:1px solid var(--line);font-weight:600;background:var(--chip)}
.phead .sp{flex:1}
.phead button{padding:1px 8px;font-weight:400}
.pbody{flex:1;overflow:auto;min-height:0}
.hl{outline:2px solid var(--accent);outline-offset:-2px;animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{50%{outline-color:transparent}}
/* дерево папок */
.frow{display:flex;align-items:center;gap:5px;padding:3px 6px;cursor:pointer;white-space:nowrap;font-size:13px}
.frow:hover{background:var(--chip)}
.frow.sel{background:var(--chip);box-shadow:inset 3px 0 var(--accent)}
.frow.dim{opacity:.5}
.arrow{width:14px;flex:none;text-align:center;color:var(--muted);user-select:none}
.mk{width:12px;flex:none;text-align:center;line-height:1}
.sdot{display:inline-block;width:10px;height:10px;border-radius:50%}
.st-gray{background:var(--gray)}.st-blue{background:var(--accent)}.st-yellow{background:var(--yellow)}.st-excluded{background:transparent;border:1px dashed var(--gray)}
.tick{color:var(--keep);font-weight:700;font-size:13px}
.fmain{flex:1;min-width:0}
.fl1{display:flex;align-items:center;gap:6px}
.fname{overflow:hidden;text-overflow:ellipsis;flex:1 1 auto;min-width:0}
.fcnt{color:var(--muted);font-size:12px;flex:none}
.flab{color:var(--muted);font-size:11px;overflow:hidden;text-overflow:ellipsis;line-height:1.2}
.odot{width:8px;height:8px;border-radius:50%;background:var(--orange);flex:none}
#fsetup{margin:8px;padding:10px;border-radius:8px;border:1px solid var(--line)}
#fsetup .row,.row{display:flex;gap:6px;margin:8px 0;flex-wrap:wrap}
#fsetup input,.row input[type=text]{flex:1;min-width:140px;font:inherit;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--text)}
#jobBox{padding:0 8px}
#jobBox:not(:empty){padding:8px}
#fwalk{padding:2px 8px}
.menu{position:fixed;z-index:70;background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,.25);padding:4px;min-width:210px;display:none}
.menu.open{display:block}
.menu button{display:block;width:100%;text-align:left;border:0;background:none;padding:6px 10px;border-radius:5px}
.menu button:hover{background:var(--chip)}
.menu hr{border:0;border-top:1px solid var(--line);margin:4px 0}
/* события */
.ev{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.ev:hover{background:var(--chip)}
.ev.cur{background:var(--chip);box-shadow:inset 3px 0 var(--accent)}
.ev .t{font-weight:600;display:flex;justify-content:space-between;gap:6px}
.ev .s{color:var(--muted);font-size:12px}
.done{color:var(--keep)}
#content{flex:1;overflow-y:auto;padding:12px 16px;position:relative;min-width:0}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px;position:sticky;top:-12px;background:var(--bg);padding:8px 0;z-index:5}
.toolbar .sp{flex:1}
.toolbar label{color:var(--muted);display:flex;gap:4px;align-items:center;cursor:pointer}
.ghead{color:var(--muted);font-size:12px;margin:14px 0 4px}
.grouprow{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}
:root{--gw:230px;--gm:150px}
body[data-size=M]{--gw:368px;--gm:240px}
body[data-size=L]{--gw:575px;--gm:375px}
.grouprow .card{flex:none;width:var(--gw)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--gm),1fr));gap:8px}
.card{position:relative;background:var(--panel);border:3px solid transparent;border-radius:8px;overflow:hidden;cursor:pointer;user-select:none}
.card.keep{border-color:var(--keep)}
.card:not(.keep) img{opacity:.35;filter:grayscale(.5)}
.card.focus{outline:3px solid var(--accent);outline-offset:1px}
.card img{display:block;width:100%;aspect-ratio:3/2;object-fit:contain;background:#000}
.badges{position:absolute;top:4px;left:4px;right:4px;display:flex;gap:4px;flex-wrap:wrap;pointer-events:none}
.badge{background:rgba(0,0,0,.72);color:#fff;border-radius:4px;padding:1px 6px;font-size:12px}
.badge.bad{background:var(--bad)}
.bar{display:flex;justify-content:space-between;align-items:center;padding:3px 6px;font-size:12px;color:var(--muted)}
.empty{color:var(--muted);padding:40px;text-align:center}
#viewer{position:fixed;inset:0;background:rgba(0,0,0,.97);z-index:50;display:none;flex-direction:column;color:#eee}
#viewer.open{display:flex}
#vimg{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;padding:8px}
#vimg img{max-width:100%;max-height:100%;object-fit:contain}
#vinfo{padding:8px 16px;font-size:13px;display:flex;gap:6px 18px;flex-wrap:wrap;align-items:center;background:#111;border-top:1px solid #333}
#vinfo .st{font-weight:700}
#vinfo .bad{color:#ff8a80}
.vnav{position:absolute;top:50%;transform:translateY(-50%);background:rgba(255,255,255,.12);border:0;color:#fff;font-size:28px;padding:10px 14px}
#vprev{left:8px}#vnext{right:8px}
#vclose{position:absolute;top:8px;right:12px;background:rgba(255,255,255,.12);border:0;color:#fff;font-size:20px}
.bar-outer{height:10px;background:var(--chip);border-radius:6px;overflow:hidden;margin:6px 0 4px}
.bar-inner{height:100%;background:var(--accent);width:0;transition:width .4s}
.err{color:var(--bad);white-space:pre-wrap}
dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;max-width:560px;width:90%}
dialog::backdrop{background:rgba(0,0,0,.5)}
dialog pre{white-space:pre-wrap;font:13px/1.5 inherit;margin:0 0 12px}
.hint{color:var(--muted);font-size:12px}
.legend{margin:8px 8px 10px;padding-top:6px;border-top:1px solid var(--line);color:var(--muted);font-size:10.5px;line-height:1.35;display:flex;flex-direction:column;gap:2px}
.legend i{font-style:normal;display:inline-block;min-width:14px;text-align:center;vertical-align:middle}
.legend .sdot,.legend .odot{margin-right:2px}

/* вкладки */
#tabs{display:flex;gap:2px;padding:0 14px;background:var(--panel);border-bottom:1px solid var(--line)}
#tabs button{border:0;border-bottom:3px solid transparent;border-radius:0;background:none;padding:8px 18px;font-weight:600;color:var(--muted)}
#tabs button.on{color:var(--text);border-bottom-color:var(--accent)}
/* дубли */
#dupes{flex:1;overflow-y:auto;padding:12px 16px;min-width:0}
.drow{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
.drow .sp{flex:1}
.dsum{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
.dcard{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:170px}
.dcard b{font-size:18px;display:block}
.dgroup{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:10px 0;padding:8px}
.dghead{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.dghead > :not(.dact){flex:0 1 auto;min-width:0}
.dact{margin-left:auto;display:flex;gap:6px;flex:none;flex-wrap:nowrap;align-items:center}
.dact button{white-space:nowrap}
.dfilt{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.warn{color:var(--orange)}
.dmembers{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
.dm{flex:none;width:210px;border:2px solid var(--line);border-radius:8px;padding:4px;background:var(--bg)}
.dm.orig{border-color:var(--keep)}
.dimg{position:relative;height:140px;background:#000;border-radius:5px;overflow:hidden;display:flex;align-items:center;justify-content:center;cursor:zoom-in}
.dimg img{max-width:100%;max-height:100%;object-fit:contain}
.dimg .badge{position:absolute;top:4px;left:4px}
.vph{font-size:42px;color:#999}
.dinfo{font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:4px}
.dmeta{font-size:11px;color:var(--muted)}
.dbtns{display:flex;gap:6px;align-items:center;margin-top:4px;font-size:12px}
.dbtns button{padding:2px 6px;font-size:12px}
#dview{position:fixed;inset:0;background:rgba(0,0,0,.95);z-index:55;display:none;flex-direction:column;color:#eee}
#dview.open{display:flex}
#dview .cmp{flex:1;min-height:0;display:flex;gap:10px;padding:10px}
#dview figure{flex:1;min-width:0;margin:0;display:flex;flex-direction:column}
#dview figure img{flex:1;min-height:0;object-fit:contain;max-width:100%}
#dview figcaption{font-size:12px;padding:4px 2px;word-break:break-all}
#dview .x{position:absolute;top:8px;right:12px;background:rgba(255,255,255,.12);border:0;color:#fff;font-size:20px}
details{margin:10px 0}
summary{cursor:pointer;font-weight:600}
.dlist{font-size:12px;margin:6px 0;max-height:320px;overflow:auto}
.dlist div{padding:2px 0;border-bottom:1px solid var(--line);word-break:break-all}
.dchk{color:var(--keep);font-size:12px;flex:none}
.partnav{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.partnav button{padding:3px 10px}
.partnav.bottom{justify-content:center;margin:16px 0 6px}
#settings{max-width:880px;width:94%;max-height:90vh;padding:0;overflow:hidden}
#settings[open]{display:flex;flex-direction:column}
.sethead{display:flex;gap:8px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--chip)}
.sethead .sp{flex:1}
#setBody{padding:6px 16px 14px;overflow:auto;font-size:12.5px}
#setBody h3{margin:14px 0 4px;font-size:13px;border-bottom:1px solid var(--line);padding-bottom:2px}
#setBody table{width:100%;border-collapse:collapse}
#setBody td,#setBody th{padding:2px 6px;vertical-align:top;text-align:left;border-bottom:1px dotted var(--line);word-break:break-word}
#setBody td:first-child{color:var(--muted);width:34%;white-space:nowrap}
#setBody th{color:var(--muted);font-weight:400}
.setbtns{margin-top:6px;display:flex;gap:8px}
.ok{color:var(--keep)}.no{color:var(--bad)}
</style>
</head>
<body>
<header>
  <h1>BestShots · финальный отбор</h1>
  <div class="totals" id="totals"></div>
  <div id="sizes" title="Размер превью: S — мелкие, M — средние, L — крупные"><button data-size="S" title="Мелкие превью">S</button><button data-size="M" title="Средние превью (≈1,6×)">M</button><button data-size="L" title="Крупные превью (≈2,5×)">L</button></div>
  <span id="trainStatus" class="hint"></span>
  <button id="trainBtn" title="Обучить личную модель на выборе в событиях, отмеченных «Проверено». Ваши отметки не сбрасываются">Обучить на моём выборе</button>
  <button id="exportBtn" class="primary" title="Скопировать отобранные фото из проверенных событий в папку экспорта (по годам). Оригиналы не изменяются">Экспорт…</button>
  <button id="settingsBtn" title="Настройки и сведения о программе: версия, стек технологий, модели на диске">⚙</button>
  <button id="quitBtn" title="Остановить BestShots и закрыть программу">Выход</button>
</header>
<nav id="tabs"><button data-tab="dupes" title="Первичная очистка архива от дубликатов: поиск точных копий и одинаковых фотографий, перемещение в карантин _Дубли">Дубли</button><button data-tab="select" class="on" title="Основной отбор лучших фото по событиям">Отбор</button></nav>
<main>
  <div id="side">
    <div class="panel" id="pfolders">
      <div class="phead">
        <button id="fcollapse" title="Свернуть или развернуть панель «Папки»">▾</button><span>Папки</span><span class="sp"></span>
        <button id="analyzeNewBtn" class="primary" title="Проанализировать только новые и изменённые файлы во всей главной папке">Анализировать новое</button>
        <button id="fmenuBtn" title="Меню панели: сменить главную папку, обновить подсчёт">⋯</button>
      </div>
      <div class="pbody" id="fbody">
        <div id="fsetup" class="hl" style="display:none">
          <div><b>Выберите главную папку с фото</b></div>
          <div class="hint">Все фото лежат в одной папке (подпапки любой вложенности). Выберите её один раз — дальше работайте по подпапкам и следите за статусом. Файлы только читаются.</div>
          <div class="row"><input id="rootInput" type="text" placeholder="Например: D:\Фото" spellcheck="false" title="Путь к главной папке с фото"><button id="browseBtn" title="Выбрать папку в системном диалоге Windows">Обзор…</button></div>
          <button id="rootOk" class="primary" title="Запомнить эту папку как главную">Выбрать</button>
        </div>
        <div id="fpart" class="hint" style="padding:2px 8px"><label title="Большой архив можно анализировать порциями: после каждой порции результаты сразу доступны во вкладке «Отбор», следующая порция запускается той же кнопкой «Анализировать новое»">Анализ: <select id="analyzePart" title="Сколько новых фото анализировать за один запуск"></select></label> <label title="После каждой порции сразу брать следующую, пока не останется непроанализированных фото. Можно остановить в любой момент"><input type="checkbox" id="analyzeAuto"> подряд до конца</label> <span id="toAnalyze" title="Фото из главной папки, которые ещё не проанализированы"></span></div>
        <div id="jobBox"></div>
        <div id="fwalk" class="hint"></div>
        <div id="ftree"></div>
        <div id="flegend" class="legend" title="Расшифровка цветовой разметки папок">
          <span><i class="sdot st-gray"></i> не проанализирована</span>
          <span><i class="sdot st-blue"></i> идёт анализ</span>
          <span><i class="sdot st-yellow"></i> проанализирована, не проверена («12 из 48» — проверено частично)</span>
          <span><i class="tick">✓</i> отработана: все события проверены или экспортированы (у папки с подпапками — все подпапки)</span>
          <span><i class="odot"></i> появились новые или изменённые файлы</span>
          <span><i class="dchk">⧉✓</i> дубли проверены</span>
          <span><i class="dchk">⬆</i>/<i class="warn">⬇</i> приоритетная папка / «скорее копии» (для дублей)</span>
          <span><i class="sdot st-excluded"></i> исключена из анализа (серым — недоступна)</span>
        </div>
      </div>
    </div>
    <div id="hsplit" title="Потяните, чтобы изменить высоту панели «Папки»"></div>
    <div class="panel" id="pevents">
      <div class="phead"><button id="ecollapse" title="Свернуть или развернуть список событий">▾</button><span>События</span><span class="hint" style="font-weight:400" title="В списке только проанализированные фото. Событие — серия снимков одной съёмки: новое событие начинается, если между кадрами больше 3 часов. Фото, которые ещё не проанализированы, здесь не видны — запустите «Анализировать новое»">· обработанные фото, по датам съёмки</span></div>
      <div class="pbody" id="events"></div>
    </div>
  </div>
  <div id="vsplit" title="Потяните, чтобы изменить ширину левой панели"></div>
  <div id="content"><div class="empty">Загрузка…</div></div>
  <div id="dupes" style="display:none"></div>
</main>
<div id="viewer">
  <button id="vclose" title="Закрыть просмотр (Esc)">✕</button>
  <button class="vnav" id="vprev" title="Предыдущий кадр (←)">‹</button>
  <button class="vnav" id="vnext" title="Следующий кадр (→)">›</button>
  <div id="vimg"><img id="vpic" alt=""></div>
  <div id="vinfo"></div>
</div>
<div id="dview"><button class="x" id="dviewClose" title="Закрыть просмотр (Esc)">✕</button><div class="cmp" id="dviewBody"></div></div>
<div id="ctx" class="menu"></div>
<dialog id="settings"><div class="sethead"><b>Настройки · О программе и стек технологий</b><span class="sp"></span>
  <button id="setCopy" title="Скопировать все сведения текстом — для отчёта об ошибке">Скопировать сведения</button>
  <button id="setClose" title="Закрыть окно настроек">Закрыть</button></div><div id="setBody"></div></dialog>
<dialog id="dlg"><pre id="dlgText"></pre><div style="text-align:right;display:flex;gap:8px;justify-content:flex-end" id="dlgBtns"></div></dialog>
<script>
"use strict";
const S = {events: [], totals: null, cur: null, data: null, photos: new Map(), filter: "all", hideRej: false, focus: null, viewer: false,
  roots: [], root: "", folder: "", expanded: new Set(), nodes: {}, rootNodes: [], walk: {}, ui: {}, job: "idle"};
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const nf = (n) => Number(n || 0).toLocaleString("ru-RU");

async function api(method, url, body) {
  if (body && typeof body === "object" && body.view_folder === undefined) body = {...body, view_folder: S.folder};
  const r = await fetch(url, {method, headers: body ? {"Content-Type": "application/json"} : {}, body: body ? JSON.stringify(body) : undefined});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}
const fmtDate = (s) => s.slice(8, 10) + "." + s.slice(5, 7) + "." + s.slice(0, 4);
const fmtTime = (s) => s.slice(11, 16);
const fmtMB = (b) => (b / 1048576 >= 1024 ? (b / 1073741824).toFixed(2) + " ГБ" : (b / 1048576).toFixed(1) + " МБ");
const num = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));
const eventsUrl = () => "/api/events?folder=" + encodeURIComponent(S.folder || "");

let uiTimer = null, uiPending = {};
function saveUi(patch) {  // ширина и состояние панелей, раскрытые узлы — в settings.json
  Object.assign(S.ui, patch);
  Object.assign(uiPending, patch);  // изменения, сделанные подряд, не затирают друг друга
  clearTimeout(uiTimer);
  uiTimer = setTimeout(() => {
    const ui = uiPending; uiPending = {};
    api("POST", "/api/ui-state", {ui, view_folder: ""}).catch(() => {});
  }, 400);
}

// ---------------------------------------------------------------- события
function applySummary(j) {
  S.events = j.events; S.totals = j.totals;
  renderTotals();
  renderEvents();
}
function renderTotals() {
  const t = S.totals; if (!t) return;
  const pct = t.all_photos ? Math.round(100 * t.done_photos / t.all_photos) : 0;
  $("#totals").innerHTML = `<span title="Сколько кадров отобрано (ваши решения и рекомендации системы)">Отобрано: <b>${nf(t.kept)}</b></span>` +
    `<span title="Фото в событиях, отмеченных «проверено» или экспортированных, и в папках, отмеченных вручную">Отработано: <b>${nf(t.done_photos)}</b> из ${nf(t.all_photos)} фото (${pct}%)</span>`;
}
function renderEvents() {
  const box = $("#events");
  box.innerHTML = S.events.map((e) => {
    const same = e.start.slice(0, 10) === e.end.slice(0, 10);
    const when = fmtDate(e.start) + " " + fmtTime(e.start) + (same ? "–" + fmtTime(e.end) : "");
    const mark = e.reviewed ? '<span class="done" title="Событие отмечено проверенным">✓</span>' : (e.exported ? '<span class="done" title="Отобранное из события экспортировано">⇪</span>' : "");
    return `<div class="ev${e.id === S.cur ? " cur" : ""}" data-id="${e.id}" title="Открыть событие ${e.id}"><div class="t"><span>${when}</span>${mark}</div>` +
      `<div class="s">Событие ${e.id} · ${e.count} кадров · отобрано ${e.kept}</div></div>`;
  }).join("") || `<div class="empty">${S.roots.length ? "В выбранной папке нет проанализированных фото. Запустите анализ." : "Нет данных. Выберите главную папку с фото."}</div>`;
}
$("#events").addEventListener("click", (ev) => {
  const el = ev.target.closest(".ev"); if (el) openEvent(+el.dataset.id);
});

async function reloadEvents(keep = true) {
  applySummary(await api("GET", eventsUrl()));
  if (keep && S.cur != null && S.events.some((e) => e.id === S.cur)) { await openEvent(S.cur); return; }
  if (S.events.length) await openEvent((S.events.find((e) => !e.reviewed) || S.events[0]).id);
  else { S.cur = null; S.data = null; S.photos = new Map(); $("#content").innerHTML = `<div class="empty">${S.roots.length ? "В выбранной папке нет проанализированных фото." : "Выберите главную папку с фото (панель «Папки» слева)."}</div>`; }
}

async function openEvent(id, part = 1) {
  S.cur = id; S.focus = null; S.part = part;
  renderEvents();
  await loadEventPart();
  $("#content").scrollTop = 0;
}
// большое событие показывается частями (по умолчанию ~150 кадров): иначе окно зависает на тысячах карточек
async function loadEventPart() {
  const q = `?folder=${encodeURIComponent(S.folder || "")}&part=${S.part}&size=${S.partSize}&filter=${S.filter}&hide=${S.hideRej ? 1 : 0}`;
  S.data = await api("GET", `/api/event/${S.cur}${q}`);
  S.part = S.data.part; S.parts = S.data.parts;
  S.photos = new Map();
  for (const g of S.data.groups) for (const p of g.photos) { p.group_n = g.size || g.photos.length; S.photos.set(p.id, p); }
  renderEvent();
}
async function goPart(p) {
  if (p < 1 || p > S.parts) return;
  S.part = p; S.focus = null;
  await loadEventPart();
  $("#content").scrollTop = 0;
}
function partNavHTML() {
  const opts = (S.choices.review || [150]).map((n) => `<option value="${n}" ${n === S.partSize ? "selected" : ""}>${n ? "по " + n : "все сразу"}</option>`).join("");
  const nav = S.parts > 1 ? `<button data-part="prev" ${S.part <= 1 ? "disabled" : ""} title="Предыдущая часть события">◀</button><span title="Событие разбито на части, чтобы открываться быстро">Часть <b>${S.part}</b> из ${S.parts}</span><button data-part="next" ${S.part >= S.parts ? "disabled" : ""} title="Следующая часть события">▶</button>` : "";
  return `<span class="partnav">${nav}<label title="Сколько кадров показывать на странице. Чем меньше, тем быстрее открывается большое событие">кадров на странице: <select id="partSize" title="Размер части">${opts}</select></label><span class="hint">всего ${nf(S.data.total)}</span></span>`;
}

const passes = () => true;  // фильтры применяет сервер до разбивки на части
function visibleList() {
  const out = [];
  for (const b of S.blocks || []) for (const p of b.photos) out.push(p);
  return out;
}

function cardHTML(p) {
  const badges = (p.group_n > 1 && p.best ? '<span class="badge" title="Лучший кадр в группе похожих">★</span>' : "") +
    (p.reject ? `<span class="badge bad" title="Брак: система не рекомендует этот кадр, но отобрать его можно вручную">брак: ${esc(p.reject_reason)}</span>` : "");
  return `<div class="card${p.keep ? " keep" : ""}" data-id="${p.id}" title="${esc(p.name)} — клик: отобрать/снять, двойной клик: просмотр">` +
    `<img loading="lazy" src="/thumb/${p.id}" alt=""><div class="badges">${badges}</div>` +
    `<div class="bar"><span title="Итоговый балл кадра">${num(p.score)}</span><span>${p.keep ? "✔ отобрано" : ""}</span></div></div>`;
}

function renderEvent() {
  const c = $("#content");
  if (!S.data) return;
  const ev = S.events.find((e) => e.id === S.cur);
  // блоки: группы с повторами — отдельный ряд, подряд идущие одиночные кадры — одна сетка
  S.blocks = [];
  for (const g of S.data.groups) {
    const ph = g.photos.filter(passes);
    if (!ph.length) continue;
    if ((g.size || g.photos.length) > 1) S.blocks.push({type: "group", id: g.id, total: g.size || g.photos.length, photos: ph});
    else {
      const last = S.blocks[S.blocks.length - 1];
      if (last && last.type === "grid") last.photos.push(...ph); else S.blocks.push({type: "grid", photos: [...ph]});
    }
  }
  const f = (k, label, tip) => `<button data-f="${k}" class="${S.filter === k ? "on" : ""}" title="${tip}">${label}</button>`;
  let html = `<div class="toolbar"><b>Событие ${S.cur}</b>` + partNavHTML() +
    f("all", "Все", "Показать все кадры события") + f("reject", "Только брак", "Показать только брак (смаз, тёмные, пересвет, низкое качество)") +
    f("keep", "Только отобранные", "Показать только отобранные кадры") +
    `<label title="Не показывать кадры, помеченные как брак"><input type="checkbox" id="hideRej" ${S.hideRej ? "checked" : ""}> скрыть брак</label><span class="sp"></span>` +
    `<button id="bestBtn" title="Сбросить ваши решения в этом событии к рекомендации системы">Взять рекомендованные</button>` +
    `<button id="revBtn" class="${ev && ev.reviewed ? "on" : ""}" title="Отметить событие проверенным (или снять отметку): папки с такими событиями станут «отработанными»">${ev && ev.reviewed ? "✓ Проверено" : "Отметить проверенным"}</button>` +
    `<button id="revNextBtn" class="primary" title="Отметить это событие проверенным и сразу открыть следующее (Shift+Enter)">✓ Проверено и дальше →</button>` +
    `<button id="nextBtn" title="Перейти к следующему событию, не отмечая это проверенным">Следующее событие →</button></div>`;
  if (!S.blocks.length) html += '<div class="empty">Нет кадров под выбранный фильтр.</div>';
  for (const b of S.blocks) {
    if (b.type === "group") html += `<div class="ghead">Группа ${b.id} · ${b.total} кадров</div><div class="grouprow">${b.photos.map(cardHTML).join("")}</div>`;
    else html += `<div class="ghead">Одиночные кадры</div><div class="grid">${b.photos.map(cardHTML).join("")}</div>`;
  }
  if (S.parts > 1) html += `<div class="partnav bottom"><button data-part="prev" ${S.part <= 1 ? "disabled" : ""} title="Предыдущая часть события">← Предыдущая часть</button><span>Часть <b>${S.part}</b> из ${S.parts}</span><button data-part="next" ${S.part >= S.parts ? "disabled" : ""} title="Следующая часть события">Следующая часть →</button></div>`;
  if (S.part >= S.parts) html += '<div class="partnav bottom"><button id="revNextBtn2" class="primary" title="Отметить это событие проверенным и сразу открыть следующее">✓ Отметить проверенным и перейти к следующему событию →</button></div>';
  c.innerHTML = html + '<div class="hint" style="margin-top:14px">Клик или Пробел — отобрано/нет · двойной клик или Enter — просмотр · стрелки — навигация</div>';
  setFocus(S.focus && S.photos.has(S.focus) && passes(S.photos.get(S.focus)) ? S.focus : null, false);
}

function updateCard(p) {
  const old = document.querySelector(`.card[data-id="${p.id}"]`);
  if (old) {
    const t = document.createElement("div"); t.innerHTML = cardHTML(p);
    const n = t.firstElementChild; if (S.focus === p.id) n.classList.add("focus");
    old.replaceWith(n);
  }
  if (S.viewer && S.vid === p.id) renderViewer();
}

function setFocus(id, scroll = true) {
  S.focus = id;
  document.querySelectorAll(".card.focus").forEach((e) => e.classList.remove("focus"));
  if (id == null) return;
  const el = document.querySelector(`.card[data-id="${id}"]`);
  if (el) { el.classList.add("focus"); if (scroll) el.scrollIntoView({block: "nearest", inline: "nearest"}); }
}

async function save(p, keep) {
  p.keep = keep ? 1 : 0;  // сразу показываем, потом подтверждаем сервером
  updateCard(p);
  try { applySummary(await api("POST", "/api/decision", {id: p.id, keep: p.keep})); }
  catch (e) { alert("Не удалось сохранить решение: " + e.message); openEvent(S.cur); }
}
const toggleKeep = (p) => save(p, !p.keep);

$("#content").addEventListener("click", (ev) => {
  const t = ev.target;
  if (t.dataset.f) { S.filter = t.dataset.f; return openEvent(S.cur, 1); }
  if (t.id === "hideRej") { S.hideRej = t.checked; return openEvent(S.cur, 1); }
  const pb = t.closest("[data-part]"); if (pb) return goPart(S.part + (pb.dataset.part === "next" ? 1 : -1));
  if (t.id === "bestBtn") return api("POST", `/api/event/${S.cur}/reset`, {}).then((j) => { applySummary(j); openEvent(S.cur); });
  if (t.id === "revBtn") {
    const e = S.events.find((x) => x.id === S.cur);
    return api("POST", `/api/event/${S.cur}/reviewed`, {reviewed: !(e && e.reviewed)}).then((j) => { applySummary(j); renderEvent(); loadTree(); });
  }
  if (t.id === "nextBtn") return nextEvent();
  if (t.id === "revNextBtn" || t.id === "revNextBtn2") return markAndNext();
  const card = t.closest(".card"); if (!card) return;
  const p = S.photos.get(+card.dataset.id);
  setFocus(p.id, false);
  toggleKeep(p);
});
$("#content").addEventListener("change", (ev) => {
  if (ev.target.id !== "partSize") return;
  S.partSize = +ev.target.value;
  try { localStorage.setItem("bestshots.partsize", String(S.partSize)); } catch (e) {}
  openEvent(S.cur, 1);
});
$("#content").addEventListener("dblclick", (ev) => {
  const card = ev.target.closest(".card");
  if (card && !ev.target.closest("button")) openViewer(+card.dataset.id);
});
async function markAndNext() {
  const go = async () => {
    try {
      const idx = S.events.findIndex((e) => e.id === S.cur);
      const next = idx >= 0 && idx + 1 < S.events.length ? S.events[idx + 1].id : null;  // порядок списка не меняется от отметки
      applySummary(await api("POST", `/api/event/${S.cur}/reviewed`, {reviewed: true}));
      loadTree();
      if (next != null) await openEvent(next);
      else { renderEvent(); dialog("Событие отмечено проверенным. Это последнее событие в списке.", [["Закрыть", null, true]]); }
    } catch (e) { alert(e.message); }
  };
  if (S.parts > 1 && S.part < S.parts) {  // большое событие: смотрели не все части
    dialog(`Событие разбито на ${S.parts} частей, открыта часть ${S.part}. Отметить всё событие проверенным и перейти к следующему?`, [["Отмена"], ["Да, отметить", go, true]]);
  } else go();
}
function nextEvent() {
  const i = S.events.findIndex((e) => e.id === S.cur);
  if (i >= 0 && i + 1 < S.events.length) openEvent(S.events[i + 1].id);
}

// ---- размер превью (S / M / L), запоминается в localStorage
function setSize(sz) {
  document.body.dataset.size = sz;
  document.querySelectorAll("#sizes button").forEach((b) => b.classList.toggle("on", b.dataset.size === sz));
  try { localStorage.setItem("bestshots.size", sz); } catch (e) {}
  const el = S.focus != null ? document.querySelector(`.card[data-id="${S.focus}"]`) : null;
  if (el) el.scrollIntoView({block: "nearest", inline: "nearest"});
}
$("#sizes").addEventListener("click", (ev) => { if (ev.target.dataset.size) setSize(ev.target.dataset.size); });
let savedSize = "S";
try { savedSize = localStorage.getItem("bestshots.size") || "S"; } catch (e) {}
setSize(["S", "M", "L"].includes(savedSize) ? savedSize : "S");

// ---- просмотр
function openViewer(id) { S.viewer = true; S.vid = id; setFocus(id); $("#viewer").classList.add("open"); renderViewer(); }
function closeViewer() { S.viewer = false; $("#viewer").classList.remove("open"); $("#vpic").removeAttribute("src"); setFocus(S.vid); }
function renderViewer() {
  const p = S.photos.get(S.vid); if (!p) return;
  const pic = $("#vpic"); const src = "/photo/" + p.id;
  if (!pic.src.endsWith(src)) pic.src = src;
  const list = visibleList(), i = list.findIndex((x) => x.id === p.id);
  $("#vinfo").innerHTML =
    `<span class="st">${p.keep ? "✔ Отобрано" : "✖ Не отобрано"}</span>` +
    `<span>${esc(p.name)} (${i + 1} из ${list.length})</span>` +
    `<span>эстетика ${num(p.aesthetic)}</span><span>эмоции ${num(p.emotion)}</span><span>балл ${num(p.score)}</span>` +
    `<span>лиц: ${p.n_faces ?? 0}</span><span>улыбка ${num(p.smile)}</span><span>моргание ${num(p.blink_frac)}</span><span>резкость ${num(p.sharp_center, 0)}</span>` +
    (p.reject ? `<span class="bad">брак: ${esc(p.reject_reason)}</span>` : "") +
    `<span>${esc(p.date_taken.replace("T", " "))}</span><span>${esc(p.camera_model || "камера неизвестна")}</span>` +
    `<span class="hint">← → · Пробел · Esc</span>`;
}
async function stepViewer(d) {
  const list = visibleList(); const i = list.findIndex((x) => x.id === S.vid);
  const n = list[i + d];
  if (n) { S.vid = n.id; setFocus(n.id); renderViewer(); return; }
  // край части: подгружаем соседнюю часть события и продолжаем просмотр
  if (d > 0 && S.part < S.parts) { await goPart(S.part + 1); const l = visibleList(); if (l.length) { S.vid = l[0].id; setFocus(S.vid); renderViewer(); } }
  else if (d < 0 && S.part > 1) { await goPart(S.part - 1); const l = visibleList(); if (l.length) { S.vid = l[l.length - 1].id; setFocus(S.vid); renderViewer(); } }
}
$("#vclose").onclick = closeViewer;
$("#vprev").onclick = () => stepViewer(-1);
$("#vnext").onclick = () => stepViewer(1);

// ---- клавиши
function moveFocus(dir) {
  const cards = [...document.querySelectorAll(".card")]; if (!cards.length) return;
  const curEl = S.focus != null ? document.querySelector(`.card[data-id="${S.focus}"]`) : null;
  if (!curEl) return setFocus(+cards[0].dataset.id);
  const i = cards.indexOf(curEl);
  if (dir === "ArrowLeft" || dir === "ArrowRight") {
    const n = cards[i + (dir === "ArrowRight" ? 1 : -1)]; if (n) setFocus(+n.dataset.id); return;
  }
  const r = curEl.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  let best = null, bd = Infinity;
  for (const c of cards) {
    const q = c.getBoundingClientRect(), dy = (q.top + q.height / 2) - cy;
    if (dir === "ArrowDown" ? dy <= r.height / 2 : dy >= -r.height / 2) continue;
    const d = Math.abs(dy) * 1000 + Math.abs(q.left + q.width / 2 - cx);
    if (d < bd) { bd = d; best = c; }
  }
  if (best) setFocus(+best.dataset.id);
}
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && $("#ctx").classList.contains("open")) { closeMenu(); return; }
  if ($("#dlg").open || $("#settings").open || (ev.target.matches && ev.target.matches("input,textarea")) || ev.ctrlKey || ev.altKey || ev.metaKey) return;
  const k = ev.key;
  if (S.viewer) {
    if (k === "Escape") closeViewer();
    else if (k === "ArrowLeft") stepViewer(-1);
    else if (k === "ArrowRight") stepViewer(1);
    else if (k === " ") toggleKeep(S.photos.get(S.vid));
    else return;
    return ev.preventDefault();
  }
  if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(k)) { moveFocus(k); return ev.preventDefault(); }
  const p = S.focus != null ? S.photos.get(S.focus) : null;
  if (k === " " && p) { toggleKeep(p); ev.preventDefault(); }
  else if (k === "Enter" && ev.shiftKey) { ev.preventDefault(); markAndNext(); }
  else if (k === "Enter" && p) { ev.preventDefault(); openViewer(p.id); }
});

// ---------------------------------------------------------------- панель «Папки»
let treeTimer = null;
const STATUS_TIP = {gray: "Не проанализирована", blue: "Идёт анализ", yellow: "Проанализирована, не проверена", green: "Отработана", excluded: "Исключена из анализа"};

function nodeTip(n) {
  if (n.excluded) return `${n.path}\nИсключена из анализа. Оригиналы не затронуты. Правый клик → «Вернуть в анализ».`;
  if (!n.accessible) return `${n.path}\nНет доступа к папке`;
  return `${n.path}\n${STATUS_TIP[n.status]}${n.manual ? " (отмечена вручную)" : ""}\nФото: ${nf(n.total)} · проанализировано: ${nf(n.analyzed)} · отработано: ${nf(n.done)}` +
    (n.pending ? `\nНовых или изменённых файлов: ${nf(n.pending)}` : "");
}
function rowHTML(n, depth, isRoot) {
  const open = isRoot || S.expanded.has(n.path);
  const arrow = isRoot && S.roots.length === 1 ? "" : (n.has_dirs && !n.excluded ? (open ? "▾" : "▸") : "");
  let mk;
  if (n.status === "green") mk = '<span class="tick">✓</span>';
  else mk = `<span class="sdot st-${n.status}"></span>`;
  let label = "";
  if (n.excluded) label = "исключена";
  else if (n.status !== "green") {
    if (n.sub_total > 0 && n.status !== "gray") label = `отработано ${n.sub_done} из ${n.sub_total} подпапок`;
    else if (n.analyzed > 0) label = `${n.done} из ${n.total}`;
  }
  const name = isRoot && S.roots.length === 1 ? "Вся главная папка" : n.name;
  const cls = "frow" + (S.folder === n.path ? " sel" : "") + (n.excluded || !n.accessible ? " dim" : "");
  return `<div class="${cls}" data-path="${esc(n.path)}" data-root="${isRoot ? 1 : 0}" style="padding-left:${6 + depth * 14}px" title="${esc(nodeTip(n))}">` +
    `<span class="arrow" data-arrow="1">${arrow}</span><span class="mk">${mk}</span><div class="fmain"><div class="fl1"><span class="fname">${esc(name)}</span>` +
    `<span class="fcnt">${n.total ? nf(n.total) : ""}</span>` +
    (n.dup_checked ? `<span class="dchk" title="Дубли проверены: ${esc(String(n.dup_checked).slice(0, 16).replace("T", " "))}">⧉✓</span>` : "") +
    (n.dup_prio > 0 ? '<span class="dchk" title="Приоритетная папка: при выборе оригинала дубликата предпочтение отдаётся ей">⬆</span>' : "") +
    (n.dup_prio < 0 ? '<span class="warn" title="Папка «скорее копии»: оригинал дубликата предпочтительно брать не отсюда">⬇</span>' : "") +
    `${n.dot ? '<span class="odot" title="Появились новые или изменённые файлы после обработки"></span>' : ""}</div>` +
    (label ? `<div class="flab">${esc(label)}</div>` : "") + "</div></div>";
}
function treeHTML(path, depth) {
  const info = S.nodes[path]; if (!info) return "";
  let html = "";
  if (info.error) html += `<div class="hint" style="padding-left:${20 + depth * 14}px">${esc(info.error)}</div>`;
  if (info.slow) html += `<div class="hint" style="padding-left:${20 + depth * 14}px">Диск отвечает медленно — показаны сохранённые данные</div>`;
  for (const n of info.children) {
    html += rowHTML(n, depth, false);
    if (S.expanded.has(n.path) && n.has_dirs && !n.excluded) html += treeHTML(n.path, depth + 1);
  }
  return html;
}
function renderTree() {
  const box = $("#ftree");
  const single = S.roots.length === 1;
  let html = "";
  for (const r of S.rootNodes) {
    html += rowHTML(r, 0, true) + treeHTML(r.path, single ? 1 : 1);
  }
  box.innerHTML = html;
  const w = S.walk || {};
  $("#fwalk").textContent = w.running ? `Подсчёт фото на диске… (папок: ${nf(w.dirs)})` : (w.timed_out ? "Подсчёт прерван по таймауту (медленный диск) — данные неполные" : "");
  $("#fsetup").style.display = S.roots.length ? "none" : "";
  $("#analyzeNewBtn").disabled = !S.roots.length || S.job === "running";
}
async function loadTree() {
  if (!S.roots.length) { renderTree(); return; }
  try {
    const paths = [...S.roots, ...[...S.expanded]];
    const r = await api("POST", "/api/folders/query", {paths, view_folder: ""});
    S.nodes = r.nodes; S.rootNodes = r.roots; S.walk = r.walk; S.job = r.job;
    $("#toAnalyze").textContent = r.to_analyze ? "· осталось проанализировать: " + nf(r.to_analyze) : "";
    if (S.totals) { S.totals.all_photos = r.all_photos || S.totals.all_photos; S.totals.done_photos = r.done_photos; renderTotals(); }
    renderTree();
  } catch (e) { $("#fwalk").textContent = "Не удалось загрузить дерево: " + e.message; }
  clearTimeout(treeTimer);
  if ((S.walk && S.walk.running) || S.job === "running") treeTimer = setTimeout(loadTree, 2500);
}

$("#ftree").addEventListener("click", (ev) => {
  const row = ev.target.closest(".frow"); if (!row) return;
  const path = row.dataset.path;
  if (ev.target.dataset.arrow && ev.target.textContent.trim()) {
    if (S.expanded.has(path)) S.expanded.delete(path); else S.expanded.add(path);
    saveUi({expanded: [...S.expanded]});
    return loadTree();
  }
  selectFolder(path);
});
$("#ftree").addEventListener("contextmenu", (ev) => {
  const row = ev.target.closest(".frow"); if (!row) return;
  ev.preventDefault();
  const path = row.dataset.path;
  const n = findNode(path); if (n) showFolderMenu(ev.clientX, ev.clientY, n, row.dataset.root === "1");
});
function findNode(path) {
  for (const r of S.rootNodes) if (r.path === path) return r;
  for (const k in S.nodes) for (const n of S.nodes[k].children) if (n.path === path) return n;
  return null;
}
async function selectFolder(path) {
  S.folder = path; saveUi({selected: path});
  renderTree();
  if (S.tab === "dupes") { await loadDupes(); S.needReload = true; return; }
  await reloadEvents(false);
}

// ---- меню
function showMenu(x, y, items) {
  const m = $("#ctx");
  m.innerHTML = "";
  for (const it of items) {
    if (it === "-") { m.appendChild(document.createElement("hr")); continue; }
    const b = document.createElement("button"); b.textContent = it.label; b.title = it.tip || it.label;
    b.onclick = () => { closeMenu(); it.fn(); };
    m.appendChild(b);
  }
  m.classList.add("open");
  const w = m.offsetWidth, h = m.offsetHeight;
  m.style.left = Math.max(4, Math.min(x, innerWidth - w - 4)) + "px";
  m.style.top = Math.max(4, Math.min(y, innerHeight - h - 4)) + "px";
}
function closeMenu() { $("#ctx").classList.remove("open"); }
document.addEventListener("click", (ev) => { if (!ev.target.closest("#ctx") && ev.target.id !== "fmenuBtn") closeMenu(); });
window.addEventListener("blur", closeMenu);

function showFolderMenu(x, y, n, isRoot) {
  const items = [];
  if (!n.excluded) {
    items.push({label: "Анализировать", tip: "Проанализировать только новые и изменённые файлы этой папки (с подпапками)", fn: () => startJob(n.path)});
    items.push({label: n.manual ? "Снять отметку «отработана»" : "Отметить отработанной", tip: "Вручную отметить папку отработанной или снять эту отметку", fn: () => markFolder(n.path, !n.manual)});
  }
  items.push({label: "Открыть в Проводнике", tip: "Открыть папку в Проводнике Windows (только просмотр)", fn: () => api("POST", "/api/folders/open", {path: n.path, view_folder: ""}).catch((e) => alert(e.message))});
  if (!n.excluded) {
    items.push("-");
    items.push({label: "Искать дубли здесь", tip: "Открыть вкладку «Дубли» и запустить поиск дубликатов в этой папке", fn: () => { S.folder = n.path; saveUi({selected: n.path}); setTab("dupes"); startDupes(n.path); }});
    items.push({label: n.dup_prio > 0 ? "Снять «приоритетная папка»" : "Приоритетная папка (дубли)", tip: "При выборе оригинала среди дубликатов предпочитать файлы из этой папки", fn: () => dupPrio(n.path, n.dup_prio > 0 ? 0 : 1)});
    items.push({label: n.dup_prio < 0 ? "Снять «скорее копии»" : "Скорее копии (дубли)", tip: "Отметить папку как «скорее копии» (например, из WhatsApp или Телеграм): оригинал предпочтительно брать не отсюда", fn: () => dupPrio(n.path, n.dup_prio < 0 ? 0 : -1)});
  }
  if (!isRoot) {
    items.push("-");
    if (n.excluded) items.push({label: "Вернуть в анализ", tip: "Снова включить папку в сканирование и анализ", fn: () => excludeFolder(n, false)});
    else items.push({label: "Исключить из анализа…", tip: "Убрать фото папки из списков и не сканировать её дальше. Оригиналы и ваши отметки не затрагиваются", fn: () => excludeFolder(n, true)});
  }
  showMenu(x, y, items);
}
async function markFolder(path, done) {
  try { await api("POST", "/api/folders/mark", {path, done, view_folder: ""}); await loadTree(); applySummary(await api("GET", eventsUrl())); }
  catch (e) { alert(e.message); }
}
function excludeFolder(n, exclude) {
  const run = async () => {
    try {
      await api("POST", "/api/folders/exclude", {path: n.path, excluded: exclude, view_folder: ""});
      if (exclude && S.folder && (S.folder === n.path || S.folder.startsWith(n.path + "\\"))) { S.folder = S.root; saveUi({selected: S.root}); }
      await loadTree(); await reloadEvents(true);
    } catch (e) { alert(e.message); }
  };
  if (!exclude) return run();
  dialog(`Исключить папку «${n.name}» из анализа?\n\nОригиналы не затрагиваются, ваши отметки сохранятся. Фото папки исчезнут из списков, и папка больше не будет сканироваться.`,
    [["Отмена"], ["Исключить", run, true]]);
}

$("#fmenuBtn").onclick = (ev) => {
  const r = ev.currentTarget.getBoundingClientRect();
  showMenu(r.left, r.bottom + 2, [
    {label: "Сменить главную папку…", tip: "Выбрать другую главную папку с фото", fn: changeRoot},
    {label: "Обновить подсчёт фото", tip: "Заново пересчитать фото на диске и найти новые файлы", fn: () => api("POST", "/api/folders/refresh", {view_folder: ""}).then(loadTree)},
  ]);
  ev.stopPropagation();
};
async function chooseFolder() {
  const r = await api("POST", "/api/choose-folder", {view_folder: ""});
  return r.path || "";
}
async function setRoot(path) {
  try {
    const r = await api("POST", "/api/roots/set", {path, view_folder: ""});
    S.roots = r.roots; S.root = r.main; S.folder = r.main; S.expanded = new Set(); saveUi({expanded: [], selected: r.main});
    await loadTree(); await reloadEvents(false);
  } catch (e) { alert(e.message); }
}
async function changeRoot() {
  const p = await chooseFolder(); if (p) await setRoot(p);
}
$("#browseBtn").onclick = async () => { try { const p = await chooseFolder(); if (p) $("#rootInput").value = p; } catch (e) { alert(e.message); } };
$("#rootOk").onclick = () => { const v = $("#rootInput").value.trim(); if (v) setRoot(v); };
$("#rootInput").addEventListener("keydown", (ev) => { if (ev.key === "Enter") $("#rootOk").click(); });

// ---- фоновый анализ
let jobTimer = null;
const STAGES = {score: "Подготовка результатов", scan: "Сканирование файлов", faces: "Поиск лиц", iqa: "Оценка качества", analyze: "Анализ кадров",
  dup_list: "Поиск файлов", dup_hash: "Контрольные суммы", dup_phash: "Перцептивные хэши", dup_match: "Поиск совпадений", dup_clip: "Сравнение содержимого", dup_group: "Формирование групп"};
function fmtEta(s) { return s == null ? "" : s < 90 ? ` · осталось ~${Math.max(5, Math.round(s / 5) * 5)} с` : ` · осталось ~${Math.round(s / 60)} мин`; }
function renderJob(j) {
  const dup = j.kind === "dupes", box = dup ? $("#dJob") : $("#jobBox"), running = j.state === "running";
  S.job = dup ? "idle" : j.state;  // синий цвет папок — только при анализе, не при поиске дублей
  $("#analyzeNewBtn").disabled = running || !S.roots.length;
  if ($("#dFind")) $("#dFind").disabled = running || !S.roots.length;
  const other = dup ? $("#jobBox") : $("#dJob"); if (other && running) other.innerHTML = "";
  if (!box) return;
  let html = "";
  if (running) {
    let text, pct = 0;
    if (j.stage === "download") {
      const mb = j.models_mb || 0, tot = j.models_total_mb || 1;
      if (mb >= 0.9 * tot) { text = "Подготовка моделей…"; pct = 100; }  // модели уже скачаны — идёт загрузка в память
      else {
        text = `${j.msg || "Загрузка моделей"} · скачано ${mb} МБ из ~${tot} МБ (один раз, дальше работа офлайн)`;
        pct = Math.min(99, 100 * mb / tot);
      }
    } else {
      text = j.stage === "score" ? j.msg : `${j.part_no > 1 ? "Порция " + j.part_no + " · " : ""}${STAGES[j.stage] || "Обработка"}: ${j.done} из ${j.total || "…"}${fmtEta(j.eta)}`;
      pct = j.total ? 100 * j.done / j.total : 0;
    }
    html = `<div class="bar-outer"><div class="bar-inner" style="width:${pct}%"></div></div><div class="hint">${esc(text)}</div>` +
      '<button class="stopJob" style="margin-top:6px" title="Остановить. Уже посчитанное сохранено в базе, при повторном запуске работа продолжится с этого места">Остановить</button>';
  } else if (j.state === "error") {
    html = `<div class="err">${esc(j.error)}</div><div class="hint">Подробности: ${esc(j.log)}</div>`;
  } else if (j.state === "stopped") {
    html = '<div class="hint">Остановлено. Обработанное сохранено — при следующем запуске анализ продолжится с этого места.</div>';
  } else if (j.state === "done") {
    html = `<div class="hint">${dup ? "Поиск дубликатов завершён." : "Анализ завершён."}</div>`;
  }
  box.innerHTML = html;
}
async function pollJob() {
  clearInterval(jobTimer);
  const tick = async () => {
    try {
      const j = await api("GET", "/api/job");
      renderJob(j);
      if (j.state !== "running") {
        clearInterval(jobTimer);
        await loadTree();
        if (j.kind === "dupes") { if (S.tab === "dupes") await loadDupes(); }
        else if (j.state === "done" || j.state === "stopped") await reloadEvents(true);
      }
    } catch (e) { clearInterval(jobTimer); }
  };
  jobTimer = setInterval(tick, 1000); tick();
  loadTree();
}
async function startJob(folder) {
  try { await api("POST", "/api/job/start", {folder: folder || "", limit: +($("#analyzePart") ? $("#analyzePart").value : 0), auto: !!($("#analyzeAuto") && $("#analyzeAuto").checked), view_folder: ""}); pollJob(); }
  catch (e) { $("#jobBox").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
$("#analyzeNewBtn").onclick = () => startJob("");
document.addEventListener("click", (ev) => { if (ev.target.classList && ev.target.classList.contains("stopJob")) api("POST", "/api/job/stop", {view_folder: ""}).then(pollJob); });

// ---- раскладка панелей: сворачивание и перетаскиваемые разделители
function applyUi() {
  const u = S.ui;
  if (u.side_w) $("#side").style.setProperty("--side-w", u.side_w + "px");
  if (u.folders_h) $("#pfolders").style.setProperty("--fp-h", u.folders_h + "px");
  $("#pfolders").classList.toggle("collapsed", !!u.folders_collapsed && S.roots.length > 0);
  $("#pevents").classList.toggle("collapsed", !!u.events_collapsed);
  $("#fcollapse").textContent = $("#pfolders").classList.contains("collapsed") ? "▸" : "▾";
  $("#ecollapse").textContent = $("#pevents").classList.contains("collapsed") ? "▸" : "▾";
}
$("#fcollapse").onclick = () => { saveUi({folders_collapsed: !S.ui.folders_collapsed}); applyUi(); };
$("#ecollapse").onclick = () => { saveUi({events_collapsed: !S.ui.events_collapsed}); applyUi(); };
function dragger(el, onmove, onend) {
  el.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    document.body.style.userSelect = "none";
    const move = (e) => onmove(e);
    const up = () => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); document.body.style.userSelect = ""; onend(); };
    document.addEventListener("mousemove", move); document.addEventListener("mouseup", up);
  });
}
dragger($("#vsplit"), (e) => {
  const w = Math.max(200, Math.min(innerWidth * 0.7, e.clientX - $("#side").getBoundingClientRect().left));
  $("#side").style.setProperty("--side-w", w + "px"); S.ui.side_w = Math.round(w);
}, () => saveUi({side_w: S.ui.side_w}));
dragger($("#hsplit"), (e) => {
  const side = $("#side").getBoundingClientRect();
  const h = Math.max(90, Math.min(side.height - 90, e.clientY - $("#pfolders").getBoundingClientRect().top));
  $("#pfolders").style.setProperty("--fp-h", h + "px"); S.ui.folders_h = Math.round(h);
}, () => saveUi({folders_h: S.ui.folders_h}));

// ---- обучение на выборе
let trainTimer = null;
function watchTraining() {
  clearInterval(trainTimer);
  const el = $("#trainStatus");
  const tick = async () => {
    try {
      const t = await api("GET", "/api/train/status");
      if (t.running) { el.textContent = "Обучение…"; return; }
      clearInterval(trainTimer);
      if (t.ok === null) { el.textContent = ""; return; }
      el.textContent = t.ok ? "Обучение завершено" : "Обучение не удалось";
      await reloadEvents(true);
      if (!$("#dlg").open) dialog("Обучение на вашем выборе\n\n" + t.text, [["Закрыть", null, true]]);
    } catch (e) { clearInterval(trainTimer); }
  };
  trainTimer = setInterval(tick, 2000); tick();
}
$("#trainBtn").onclick = async () => {
  try {
    const r = await api("POST", "/api/train", {});
    if (r.note) { alert(r.note); return; }
    $("#trainStatus").textContent = r.started ? "Обучение…" : "Обучение уже идёт";
    watchTraining();
  } catch (e) { alert(e.message); }
};

// ---- выход
$("#quitBtn").onclick = async () => {
  if (!confirm("Остановить BestShots? Идущая обработка будет прервана (сделанное сохранено).")) return;
  try { await api("POST", "/api/shutdown", {}); } catch (e) {}
  document.body.innerHTML = '<div class="empty" style="margin:20vh auto">BestShots остановлен. Окно можно закрыть.</div>';
};

// ---- экспорт
function dialog(text, buttons) {
  $("#dlgText").textContent = text;
  const b = $("#dlgBtns"); b.innerHTML = "";
  for (const [label, fn, primary] of buttons) {
    const el = document.createElement("button"); el.textContent = label; if (primary) el.className = "primary";
    el.title = label === "Отмена" ? "Закрыть окно без действий" : label;
    el.onclick = () => { $("#dlg").close(); if (fn) fn(); }; b.appendChild(el);
  }
  $("#dlg").showModal();
}
$("#exportBtn").onclick = async () => {
  try {
    const p = await api("GET", "/api/export/preview");
    const head = `Копирование (оригиналы не изменяются и не перемещаются) в:\n${p.out}\nФайлы раскладываются по годам: ГГГГ\\имя_файла. Уже скопированные пропускаются.\n\n`;
    const notReviewed = p.events_total - p.events_reviewed;
    const unrev = p.unreviewed_count ? `\n\nНе проверено событий: ${notReviewed}. Отобранных в них кадров: ${nf(p.unreviewed_count)} (${fmtMB(p.unreviewed_bytes)}) — они НЕ копируются. Чтобы они попали в экспорт, откройте события и нажмите «Проверено».` : "";
    const allBtn = ["Копировать и непроверенные…", () => dialog(`Копировать также рекомендации системы по непроверенным событиям (${nf(p.unreviewed_count)} кадров)? Это кадры, которые вы не просматривали.`, [["Отмена"], ["Да, копировать все", () => runExport(true), true]])];
    if (!p.count) {
      dialog(head + `Проверенных событий нет (проверено ${p.events_reviewed} из ${p.events_total}), копировать пока нечего.` + unrev, p.unreviewed_count ? [["Закрыть"], allBtn] : [["Закрыть", null, true]]);
    } else {
      dialog(head + `Проверено событий: ${p.events_reviewed} из ${p.events_total}.\nВ экспорт попадут отобранные кадры только из проверенных событий: ${nf(p.count)} файлов, ${fmtMB(p.bytes)}.` + unrev,
        p.unreviewed_count ? [["Отмена"], allBtn, ["Копировать проверенные", () => runExport(false), true]] : [["Отмена"], ["Копировать", () => runExport(false), true]]);
    }
  } catch (e) { alert(e.message); }
};
async function runExport(includeUnreviewed) {
  dialog("Копирование… не закрывайте окно.", []);
  try {
    const r = await api("POST", "/api/export", {include_unreviewed: !!includeUnreviewed});
    $("#dlg").close();
    dialog(`Экспорт завершён.\nСкопировано: ${r.copied} (${fmtMB(r.bytes)})\nПропущено (уже есть): ${r.skipped}\nОшибок: ${r.errors.length}` +
      (r.errors.length ? "\n\n" + r.errors.slice(0, 8).join("\n") : "") + `\n\nПапка: ${r.out}` +
      `\n\nМеток из проверенных событий: ${r.labels}. ` + (r.training ? "Обучение личной модели запущено…" : (r.labels ? "Обучение уже идёт." : "Обучение не запускалось: нет проверенных событий.")), [["Закрыть", null, true]]);
    watchTraining();
    loadTree(); reloadEvents(true);
  } catch (e) { $("#dlg").close(); alert("Ошибка экспорта: " + e.message); }
}

// ---------------------------------------------------------------- вкладки и «Дубли»
S.tab = "select"; S.needReload = false;
S.part = 1; S.parts = 1; S.partSize = 150; S.choices = {analyze: [0], review: [150], review_default: 150};
function initChoices(ch) {
  S.choices = ch || S.choices;
  let ps = NaN; try { ps = parseInt(localStorage.getItem("bestshots.partsize"), 10); } catch (e) {}
  S.partSize = S.choices.review.includes(ps) ? ps : S.choices.review_default;
  const sel = $("#analyzePart");
  sel.innerHTML = S.choices.analyze.map((n) => `<option value="${n}">${n ? "порциями по " + nf(n) : "все сразу"}</option>`).join("");
  let a = 0; try { a = parseInt(localStorage.getItem("bestshots.analyzepart") || "0", 10) || 0; } catch (e) {}
  sel.value = S.choices.analyze.includes(a) ? a : 0;
  sel.onchange = () => { try { localStorage.setItem("bestshots.analyzepart", sel.value); } catch (e) {} };
  const auto = $("#analyzeAuto");
  try { auto.checked = localStorage.getItem("bestshots.analyzeauto") === "1"; } catch (e) {}
  auto.onchange = () => { try { localStorage.setItem("bestshots.analyzeauto", auto.checked ? "1" : "0"); } catch (e) {} };
}
const D = {kind: "", uncertain: false, offset: 0, total: 0, built: false, cmp: {}};
const HEADER_SELECT_ONLY = ["#sizes", "#trainStatus", "#trainBtn", "#exportBtn"];

function setTab(tab) {
  S.tab = tab;
  try { localStorage.setItem("bestshots.tab", tab); } catch (e) {}
  const dup = tab === "dupes";
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  $("#content").style.display = dup ? "none" : "";
  $("#dupes").style.display = dup ? "" : "none";
  $("#pevents").style.display = dup ? "none" : "";
  $("#hsplit").style.display = dup ? "none" : "";
  $("#pfolders").style.flex = dup ? "1 1 auto" : "";
  HEADER_SELECT_ONLY.forEach((q) => { const el = $(q); if (el) el.style.display = dup ? "none" : ""; });
  if (dup) { if (!D.built) buildDupes(); loadDupes(); }
  else if (S.needReload) { S.needReload = false; reloadEvents(true); }
}
$("#tabs").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (b) setTab(b.dataset.tab); });

const shortPath = (p) => (S.root && p.startsWith(S.root) ? p.slice(S.root.length).replace(/^[\\/]/, "") : p);

function buildDupes() {
  D.built = true;
  $("#dupes").innerHTML = `
    <div class="drow"><span>Папка поиска: <b id="dScope"></b></span><span class="hint" id="dLast"></span></div>
    <div class="drow">
      <button id="dFind" class="primary" title="Найти дубликаты в выбранной папке (по умолчанию — вся главная папка). Уже посчитанное берётся из кэша">Найти дубли</button>
      <span class="hint">Выберите папку в дереве слева. Оригиналы не удаляются: дубли перемещаются в карантин _Дубли и могут быть возвращены.</span>
    </div>
    <div id="dJob"></div>
    <div class="dsum" id="dSummary"></div>
    <div class="drow">
      <span class="dfilt"><label title="Какие группы показывать">Тип: <select id="dKind" title="Показать все группы, только точные копии или только «ту же фотографию»"><option value="">все</option><option value="exact">точные копии</option><option value="visual">та же фотография</option></select></label>
      <label title="Показать только группы, где автоматика не уверена (совпадение близко к порогу)"><input type="checkbox" id="dUnc"> только неуверенные</label></span>
      <span class="dact"><button id="dApplyConfident" class="primary" title="Одной кнопкой переместить в карантин дубли ВСЕХ визуальных групп, где автоматика уверена. Неуверенные остаются для ручной проверки. Оригиналы остаются на месте, всё можно вернуть">Переместить все уверенные</button>
      <button id="dApplyExact" title="Переместить дубли всех групп «точные копии» в карантин (оригинал в каждой группе остаётся)">Применить ко всем точным копиям</button>
      <button id="dApplyConf" title="Переместить дубли всех визуальных групп, которые вы подтвердили кнопкой «Подтвердить»">Применить ко всем проверенным</button></span>
    </div>
    <div id="dGroups"></div>
    <div class="drow"><button id="dMore" title="Показать следующие группы">Показать ещё</button><span class="hint" id="dCount"></span></div>
    <details id="dsame"><summary>Одинаковые имена, разные фото <span class="hint" id="dsameN"></span></summary><div class="hint">Только подсказка, без действий: камеры повторяют имена (DSC_0001…), это разные снимки.</div><div class="dlist" id="dsameBody"></div></details>
    <details id="drawj"><summary>Пары RAW + JPEG <span class="hint" id="drawjN"></span></summary><div class="hint">Один кадр в двух форматах — это не дубли, обе версии остаются.</div><div class="dlist" id="drawjBody"></div></details>
    <details id="derr"><summary>Ошибки чтения <span class="hint" id="derrN"></span></summary><div class="dlist" id="derrBody"></div></details>
    <details id="djr" open><summary>Карантин и журнал <span class="hint" id="djrN"></span></summary>
      <div class="drow"><button id="dRestoreSel" title="Вернуть на прежние места отмеченные файлы">Вернуть выбранные</button><button id="dRestoreAll" title="Вернуть из карантина все файлы по журналу">Вернуть все</button><span class="hint">Если на прежнем месте уже есть файл, он не перезаписывается. Удалять файлы из _Дубли нужно вручную самим.</span></div>
      <div class="dlist" id="djrBody"></div></details>`;
  $("#dFind").onclick = () => startDupes(S.folder || S.root);
  $("#dKind").onchange = () => { D.kind = $("#dKind").value; loadDupes(); };
  $("#dUnc").onchange = () => { D.uncertain = $("#dUnc").checked; loadDupes(); };
  $("#dMore").onclick = () => moreGroups();
  $("#dApplyExact").onclick = () => applyDupes("exact_all");
  $("#dApplyConfident").onclick = () => applyDupes("visual_confident");
  $("#dApplyConf").onclick = () => applyDupes("confirmed_all");
  $("#dRestoreAll").onclick = () => restoreDupes({scope: "all"}, "Вернуть из карантина все файлы?");
  $("#dRestoreSel").onclick = () => {
    const ids = [...document.querySelectorAll("#djrBody input[data-jid]:checked")].map((c) => +c.dataset.jid);
    if (!ids.length) return alert("Отметьте файлы в журнале.");
    restoreDupes({scope: "ids", ids}, `Вернуть выбранные файлы (${ids.length})?`);
  };
  $("#dGroups").addEventListener("click", onGroupClick);
  $("#dGroups").addEventListener("dblclick", (ev) => { const img = ev.target.closest(".dimg"); if (img) openDView([+img.closest(".dm").dataset.id]); });
  $("#djrBody").addEventListener("click", (ev) => { const b = ev.target.closest("button[data-rgroup]"); if (b) restoreDupes({scope: "group", group_id: +b.dataset.rgroup}, "Вернуть файлы этой группы?"); });
  $("#dviewClose").onclick = closeDView;
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && $("#dview").classList.contains("open")) closeDView(); });
}

function updateDupButtons(sm) {  // кнопки массовых действий: сколько групп затронут; неактивны, если делать нечего
  const setBtn = (id, label, n) => { const b = $(id); b.textContent = `${label} (${nf(n)})`; b.disabled = !n; };
  setBtn("#dApplyConfident", "Переместить все уверенные", sm.visual.groups - sm.uncertain);
  setBtn("#dApplyExact", "Применить ко всем точным копиям", sm.exact.groups);
  setBtn("#dApplyConf", "Применить ко всем проверенным", sm.confirmed_visual);
}

async function loadDupes() {
  if (!D.built) return;
  const scope = S.folder || S.root;
  $("#dScope").textContent = scope || "—";
  const node = findNode(scope);
  $("#dLast").textContent = node && node.dup_checked ? "дубли проверены: " + String(node.dup_checked).slice(0, 16).replace("T", " ") : "ещё не проверялась";
  $("#dFind").disabled = !S.roots.length || S.job === "running";
  const sm = await api("GET", dupUrl("/api/dupes/summary"));
  const freed = sm.exact.bytes + sm.visual.bytes;
  $("#dSummary").innerHTML =
    `<div class="dcard" title="Файлы с одинаковым содержимым (sha256)"><b>${nf(sm.exact.groups)}</b>групп точных копий · ${nf(sm.exact.files)} лишних файлов · ${fmtMB(sm.exact.bytes)}</div>` +
    `<div class="dcard" title="Пересохранённые, уменьшенные, повёрнутые копии одной фотографии. Серийные кадры сюда не попадают"><b>${nf(sm.visual.groups)}</b>групп «та же фотография» · ${nf(sm.visual.files)} файлов · ${fmtMB(sm.visual.bytes)}${sm.uncertain ? ` · ⚠ неуверенных: ${sm.uncertain}` : ""}</div>` +
    `<div class="dcard" title="Сколько места освободится после перемещения всех дублей в карантин"><b>${fmtMB(freed)}</b>освободится</div>` +
    `<div class="dcard" title="Файлы, перемещённые в _Дубли и ещё не возвращённые"><b>${nf(sm.quarantined.files)}</b>в карантине · ${fmtMB(sm.quarantined.bytes)}</div>`;
  updateDupButtons(sm);
  $("#dsameN").textContent = `(${sm.samename})`; $("#drawjN").textContent = `(${sm.rawjpeg})`; $("#derrN").textContent = `(${sm.errors})`;
  D.offset = 0; D.cmp = {}; $("#dGroups").innerHTML = "";
  await moreGroups();
  loadDupExtra(); loadJournal();
}
const dupUrl = (p, extra) => p + "?folder=" + encodeURIComponent(S.folder || S.root || "") + (extra || "");

async function moreGroups() {
  const r = await api("GET", dupUrl("/api/dupes/groups", `&kind=${D.kind}&uncertain=${D.uncertain ? 1 : 0}&offset=${D.offset}&limit=20`));
  D.total = r.total; D.offset += r.groups.length;
  $("#dGroups").insertAdjacentHTML("beforeend", r.groups.map(groupHTML).join(""));
  $("#dCount").textContent = `Показано ${Math.min(D.offset, D.total)} из ${D.total} групп`;
  $("#dMore").style.display = D.offset < D.total ? "" : "none";
  if (!D.total) $("#dGroups").innerHTML = '<div class="empty">Групп дубликатов нет. Нажмите «Найти дубли» (или выберите другой тип и папку).</div>';
}

function memberHTML(g, m) {
  const img = m.kind === "video" ? '<div class="vph" title="Видео: сравниваются только точные копии">🎞</div>' : `<img loading="lazy" src="/dthumb/${m.id}" alt="">`;
  const prio = m.prio > 0 ? " · ⬆приоритетная папка" : m.prio < 0 ? " · ⬇скорее копии" : "";
  return `<div class="dm${m.is_orig ? " orig" : ""}" data-gid="${g.id}" data-id="${m.id}" data-path="${esc(m.path)}">` +
    `<div class="dimg" title="Двойной клик — просмотр во весь размер">${img}${m.is_orig ? '<span class="badge" title="Оригинал: этот файл останется на месте">★ оригинал</span>' : ""}</div>` +
    `<div class="dinfo" title="${esc(m.path)}">${esc(shortPath(m.path))}</div>` +
    `<div class="dmeta">${fmtMB(m.size)} · ${m.width ? m.width + "×" + m.height : "—"} · ${m.date ? m.date.slice(0, 10) : "—"} · ${m.has_exif ? "EXIF" : "без EXIF"}${prio}</div>` +
    `<div class="dbtns"><button data-act="orig" ${m.is_orig ? "disabled" : ""} title="Сделать этот файл оригиналом группы (остальные станут дублями)">★ Оригинал</button>` +
    (m.kind === "video" ? "" : '<label title="Отметить для сравнения двух кадров рядом"><input type="checkbox" data-act="cmp"> сравнить</label>') + "</div></div>";
}
function groupHTML(g) {
  const tag = g.kind === "exact" ? "точные копии" : "та же фотография";
  return `<div class="dgroup" data-gid="${g.id}"><div class="dghead"><b>Группа ${g.id}</b><span>${tag} · ${g.members.length} файла · освободится ${fmtMB(g.wasted)}</span>` +
    (g.uncertain ? '<span class="warn" title="Совпадение близко к порогу — проверьте глазами (двойной клик или «Сравнить рядом»)">⚠ автоматика не уверена</span>' : "") + '<span class="dact">' +
    '<button data-act="compare" disabled title="Отметьте «сравнить» у двух кадров, чтобы посмотреть их рядом">Сравнить рядом</button>' +
    (g.kind === "visual" ? `<button data-act="confirm" class="${g.confirmed ? "on" : ""}" title="Отметить группу проверенной: тогда её дубли попадут под «Применить ко всем проверенным»">${g.confirmed ? "✓ Проверена" : "Подтвердить"}</button>` : "") +
    '<button data-act="apply" class="primary" title="Переместить дубли этой группы в карантин _Дубли (оригинал останется на месте)">Переместить дубли</button>' +
    '<button data-act="ignore" title="Скрыть группу: файлы останутся на месте">Пропустить</button></span></div>' +
    `<div class="dmembers">${g.members.map((m) => memberHTML(g, m)).join("")}</div></div>`;
}

async function onGroupClick(ev) {
  const t = ev.target;
  const act = t.dataset && t.dataset.act; if (!act) return;
  const gEl = t.closest(".dgroup"), gid = +gEl.dataset.gid;
  try {
    if (act === "cmp") {
      const checked = [...gEl.querySelectorAll('input[data-act="cmp"]:checked')];
      if (checked.length > 2) { t.checked = false; return; }
      gEl.querySelector('button[data-act="compare"]').disabled = checked.length !== 2;
    } else if (act === "compare") {
      openDView([...gEl.querySelectorAll('input[data-act="cmp"]:checked')].map((c) => +c.closest(".dm").dataset.id));
    } else if (act === "orig") {
      await api("POST", "/api/dupes/set-original", {group_id: gid, path: t.closest(".dm").dataset.path, view_folder: ""});
      await loadDupes();
    } else if (act === "confirm") {
      await api("POST", "/api/dupes/confirm", {group_id: gid, confirmed: !t.classList.contains("on"), view_folder: ""});
      t.classList.toggle("on"); t.textContent = t.classList.contains("on") ? "✓ Проверена" : "Подтвердить";
      updateDupButtons(await api("GET", dupUrl("/api/dupes/summary")));  // счётчик «проверенных» обновляется сразу
    } else if (act === "ignore") {
      await api("POST", "/api/dupes/ignore", {group_id: gid, view_folder: ""}); await loadDupes();
    } else if (act === "apply") {
      applyDupes("group", gid);
    }
  } catch (e) { alert(e.message); }
}

async function applyDupes(scope, gid) {
  try {
    const body = scope === "group" ? {scope, group_id: gid} : {scope};
    const plan = await api("POST", "/api/dupes/apply", {...body, dry_run: true});
    if (!plan.files) { alert("Нет файлов для перемещения."); return; }
    dialog(`Переместить ${nf(plan.files)} файлов (${fmtMB(plan.bytes)}) из ${plan.groups} групп в папку:\n${plan.dest.join("\n")}\n\nСтруктура подпапок сохраняется. Оригиналы остаются на месте. Ничего не удаляется — дубли можно вернуть кнопкой «Вернуть».`,
      [["Отмена"], ["Переместить", async () => {
        try {
          const r = await api("POST", "/api/dupes/apply", body);
          S.needReload = true;
          dialog(`Перемещено файлов: ${nf(r.moved)} (${fmtMB(r.bytes)}).\nОшибок: ${r.errors.length}` + (r.errors.length ? "\n\n" + r.errors.slice(0, 6).join("\n") : "") +
            "\n\nЖурнал: <главная папка>\\_Дубли\\журнал.csv", [["Закрыть", null, true]]);
          await loadDupes(); loadTree();
        } catch (e) { alert(e.message); }
      }, true]]);
  } catch (e) { alert(e.message); }
}

function restoreDupes(body, question) {
  dialog(question + "\n\nФайлы вернутся на прежние места. Если там уже есть файл с таким именем, он не будет перезаписан.", [["Отмена"], ["Вернуть", async () => {
    try {
      const r = await api("POST", "/api/dupes/restore", body);
      S.needReload = true;
      dialog(`Возвращено: ${nf(r.restored)}.` + (r.conflicts.length ? `\nНе возвращено — на исходном месте уже есть файл: ${r.conflicts.length}\n` + r.conflicts.slice(0, 5).join("\n") : "") +
        (r.missing.length ? `\nНе найдено в карантине: ${r.missing.length}` : "") + (r.errors.length ? `\nОшибок: ${r.errors.length}` : "") +
        "\n\nЧтобы вернувшиеся фото попали в «Отбор», запустите «Анализировать новое».", [["Закрыть", null, true]]);
      await loadDupes(); loadTree();
    } catch (e) { alert(e.message); }
  }, true]]);
}

async function startDupes(folder) {
  try { await api("POST", "/api/dupes/find", {folder: folder || "", view_folder: ""}); pollJob(); }
  catch (e) { if ($("#dJob")) $("#dJob").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
async function dupPrio(path, prio) {
  try { await api("POST", "/api/folders/dupe-prio", {path, prio, view_folder: ""}); await loadTree(); if (S.tab === "dupes") loadDupes(); }
  catch (e) { alert(e.message); }
}

async function loadDupExtra() {
  const row = (f) => `<div title="${esc(f.path)}">${esc(shortPath(f.path))} · ${fmtMB(f.size)}${f.width ? " · " + f.width + "×" + f.height : ""}</div>`;
  const same = await api("GET", dupUrl("/api/dupes/extra", "&kind=samename"));
  $("#dsameBody").innerHTML = same.items.slice(0, 200).map((g) => `<div><b>${esc(g.name)}</b></div>` + g.files.map(row).join("")).join("") || "Нет.";
  const raw = await api("GET", dupUrl("/api/dupes/extra", "&kind=rawjpeg"));
  $("#drawjBody").innerHTML = raw.items.slice(0, 300).map((p) => `<div>${p.files.map((f) => esc(shortPath(f.path))).join("  +  ")}</div>`).join("") || "Нет.";
  const err = await api("GET", dupUrl("/api/dupes/extra", "&kind=errors"));
  $("#derrBody").innerHTML = err.items.map((e) => `<div title="${esc(e.error)}">${esc(shortPath(e.path))} — ${esc(e.error)}</div>`).join("") || "Нет.";
}
async function loadJournal() {
  const j = await api("GET", "/api/dupes/journal");
  const byGroup = {};
  for (const m of j.items) (byGroup[m.group_id] = byGroup[m.group_id] || []).push(m);
  const active = j.items.filter((m) => !m.restored_at).length;
  $("#djrN").textContent = `(в карантине: ${active}, записей: ${j.items.length})`;
  $("#djrBody").innerHTML = Object.entries(byGroup).map(([gid, items]) => {
    const left = items.filter((m) => !m.restored_at).length;
    return `<div><b>Группа ${gid}</b> · ${items[0].kind === "exact" ? "точные копии" : "та же фотография"} · в карантине: ${left}` +
      (left ? ` <button data-rgroup="${gid}" title="Вернуть все файлы этой группы из карантина">Вернуть группу</button>` : "") + "</div>" +
      items.map((m) => `<div><label title="${esc(m.to_path)}">${m.restored_at ? "↩ " : `<input type="checkbox" data-jid="${m.id}"> `}${esc(shortPath(m.from_path))} · ${fmtMB(m.size)} · ${esc(m.moved_at.replace("T", " "))}${m.restored_at ? " · возвращён" : ""}</label> <span class="hint">оригинал: ${esc(shortPath(m.orig_path))}</span></div>`).join("");
  }).join("") || "Журнал пуст.";
}

// просмотр во весь размер и сравнение двух кадров
async function openDView(ids) {
  if (!ids.length) return;
  const info = {};
  document.querySelectorAll(".dm").forEach((el) => { info[el.dataset.id] = el; });
  $("#dviewBody").innerHTML = ids.map((id) => {
    const el = info[id]; const meta = el ? el.querySelector(".dmeta").textContent : "";
    return `<figure><img src="/dphoto/${id}" alt=""><figcaption>${esc(el ? el.dataset.path : "")}<br>${esc(meta)}</figcaption></figure>`;
  }).join("");
  $("#dview").classList.add("open");
}
function closeDView() { $("#dview").classList.remove("open"); $("#dviewBody").innerHTML = ""; }

// ---------------------------------------------------------------- «Настройки»: о программе и стек технологий
const SET = {data: null, sizes: {state: "running", sizes: {}}, timer: null};
const fmtB = (b) => (b == null ? "—" : fmtMB(b));
async function openSettings() {
  $("#settings").showModal();
  $("#setBody").innerHTML = '<div class="hint">Загрузка…</div>';
  try {
    SET.data = await api("GET", "/api/about"); SET.sizes = {state: "running", sizes: {}};
    renderSettings(); pollSizes();
  } catch (e) { $("#setBody").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
function pollSizes() {  // размеры больших папок считаются в фоне: окно открывается сразу
  clearInterval(SET.timer);
  const tick = async () => {
    try { SET.sizes = await api("GET", "/api/about/sizes"); renderSettings(); if (SET.sizes.state !== "running") clearInterval(SET.timer); }
    catch (e) { clearInterval(SET.timer); }
  };
  SET.timer = setInterval(tick, 1200); tick();
}
function settingsModel() {  // единая модель для показа и для копирования текстом
  const d = SET.data, sz = SET.sizes.sizes;
  const groups = d.groups.map((g) => ({title: g.title, rows: g.rows}));
  const sizeRows = Object.entries(d.size_labels).map(([k, name]) => [name, sz[k] === undefined ? "считаю…" : fmtB(sz[k])]);
  const known = Object.keys(d.size_labels).every((k) => sz[k] !== undefined);
  if (known) sizeRows.push(["Итого", fmtB(Object.keys(d.size_labels).reduce((a, k) => a + (sz[k] || 0), 0))]);
  return {groups, models: d.models, sizeRows};
}
function renderSettings() {
  if (!SET.data) return;
  const m = settingsModel();
  let html = "";
  m.groups.forEach((g, i) => {
    html += `<h3>${esc(g.title)}</h3><table>` + g.rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("") + "</table>";
    if (i === 0) html += '<div class="setbtns"><button data-open="app" title="Открыть папку программы в Проводнике">Открыть папку программы</button><button data-open="logs" title="Открыть папку с журналом работы (logs)">Открыть логи</button></div>';
  });
  html += "<h3>Модели на диске</h3><table><tr><th>Модель</th><th>Файл</th><th>Размер</th><th>Загружена</th></tr>" +
    m.models.map((x) => `<tr><td>${esc(x.name)}</td><td>${esc(x.file)}</td><td>${fmtB(x.size)}</td><td class="${x.loaded ? "ok" : "no"}">${x.loaded ? "да" : "нет"}</td></tr>`).join("") + "</table>";
  html += "<h3>Занимаемое место</h3><table>" + m.sizeRows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("") + "</table>";
  $("#setBody").innerHTML = html;
}
function settingsText() {
  const m = settingsModel();
  const lines = [`BestShots ${SET.data.version} — сведения о программе`, ""];
  m.groups.forEach((g) => { lines.push(`== ${g.title} ==`); g.rows.forEach(([k, v]) => lines.push(`${k}: ${v}`)); lines.push(""); });
  lines.push("== Модели на диске ==");
  m.models.forEach((x) => lines.push(`${x.name} | ${x.file} | ${fmtB(x.size)} | загружена: ${x.loaded ? "да" : "нет"}`));
  lines.push("", "== Занимаемое место ==");
  m.sizeRows.forEach(([k, v]) => lines.push(`${k}: ${v}`));
  return lines.join("\n");
}
$("#settingsBtn").onclick = openSettings;
$("#setClose").onclick = () => { clearInterval(SET.timer); $("#settings").close(); };
$("#settings").addEventListener("close", () => clearInterval(SET.timer));
$("#setBody").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-open]");
  if (b) api("POST", "/api/about/open", {what: b.dataset.open, view_folder: ""}).catch((e) => alert(e.message));
});
$("#setCopy").onclick = async () => {
  if (!SET.data) return;
  const text = settingsText(), btn = $("#setCopy");
  try { await navigator.clipboard.writeText(text); }
  catch (e) {  // запасной способ, если буфер обмена недоступен
    const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } finally { ta.remove(); }
  }
  btn.textContent = "Скопировано ✓"; setTimeout(() => { btn.textContent = "Скопировать сведения"; }, 1800);
};

// ---- запуск
(async () => {
  try {
    api("GET", "/api/train/status").then((t) => { if (t.running) watchTraining(); }).catch(() => {});
    const r = await api("GET", "/api/roots");
    S.roots = r.roots; S.root = r.main; S.ui = r.ui || {}; S.walk = r.walk;
    initChoices(r.choices);
    S.expanded = new Set((S.ui.expanded || []).filter((p) => S.roots.some((x) => p.startsWith(x))));
    S.folder = S.ui.selected && S.roots.some((x) => S.ui.selected === x || S.ui.selected.startsWith(x + "\\")) ? S.ui.selected : S.root;
    applyUi();
    if (!S.roots.length) {
      $("#fsetup").style.display = "";
      $("#content").innerHTML = '<div class="empty">Выберите главную папку с фото в панели «Папки» слева.</div>';
      renderTree();
    } else {
      await loadTree();
      await reloadEvents(false);
    }
    api("GET", "/api/job").then((j) => { renderJob(j); if (j.state === "running") pollJob(); }).catch(() => {});
    let tab = "select"; try { tab = localStorage.getItem("bestshots.tab") || "select"; } catch (e) {}
    if (tab === "dupes" && S.roots.length) setTab("dupes");
    // параметры адреса для документации и тестов: ?tab=dupes|select, ?size=S|M|L, ?settings=1
    const qp = new URLSearchParams(location.search);
    if (qp.get("tab") && S.roots.length) setTab(qp.get("tab") === "dupes" ? "dupes" : "select");
    if (["S", "M", "L"].includes(qp.get("size"))) setSize(qp.get("size"));
    if (qp.has("settings")) setTimeout(openSettings, 900);
  } catch (e) { $("#content").innerHTML = '<div class="empty">Ошибка: ' + esc(e.message) + "</div>"; }
})();
</script>
</body>
</html>
'''


def main():
    global DB_PATH, OUT_DIR, PORT
    power.prefer_performance()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="BestShots: интерфейс финального отбора")
    ap.add_argument("--db", default="bestshots.db")
    ap.add_argument("--out", default=str(ROOT / "Export"), help="папка экспорта")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="не открывать браузер автоматически")
    args = ap.parse_args()
    DB_PATH, OUT_DIR, PORT = str(Path(args.db).resolve()), Path(args.out).resolve(), args.port
    setup_logging()
    con = db()  # базы может ещё не быть — интерфейс предложит выбрать главную папку
    con.executescript(SCHEMA)
    con.executescript(folders.SCHEMA)
    dupes.ensure_schema(con)  # таблицы дублей (с миграцией)
    st = load_settings()
    folders.infer_root(con, st.get("root") or st.get("last_folder") or "")  # база, созданная до панели «Папки»
    con.close()
    folders.start_walk(DB_PATH)

    try:
        server = Server(("127.0.0.1", PORT), Handler)
    except OSError as e:
        log.error("порт %s занят: %s", PORT, e)
        sys.exit(f"Не удалось занять порт {PORT} (возможно, BestShots уже запущен): {e}")
    SERVER["obj"] = server
    url = f"http://127.0.0.1:{PORT}/"
    print(f"BestShots: {url}  (Ctrl+C — остановить)\nЭкспорт: {OUT_DIR}")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        stop_job()
        server.server_close()


if __name__ == "__main__":
    main()
