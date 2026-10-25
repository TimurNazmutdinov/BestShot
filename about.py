"""BestShots: «О программе и стек технологий» — сведения для окна настроек (только чтение).

Версии библиотек определяются через importlib.metadata при открытии окна (без импорта тяжёлых пакетов);
размеры больших папок считаются в фоновом потоке, чтобы окно открывалось мгновенно.
"""
import json
import os
import platform
import sqlite3
import sys
import threading
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import config

NOT_INSTALLED = "не установлен"


def version_of(*names):
    """Версия первого найденного установленного пакета из names."""
    for n in names:
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return NOT_INSTALLED


def webview2_version():
    """Версия среды Edge WebView2 (из реестра Windows) или «не найден»."""
    if sys.platform != "win32":
        return "—"
    try:
        import winreg
        guid = r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, guid) as k:
                    return winreg.QueryValueEx(k, "pv")[0]
            except OSError:
                continue
    except Exception:
        pass
    return "не найден"


def fmt_size(n):
    if n is None:
        return "—"
    for unit, div in (("ГБ", 1 << 30), ("МБ", 1 << 20), ("КБ", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}" if unit != "КБ" else f"{n / div:.0f} {unit}"
    return f"{n} Б"


def dir_size(path):
    """Размер папки (без перехода по ссылкам и junction)."""
    total, stack = 0, [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def path_size(p):
    p = Path(p)
    try:
        return p.stat().st_size if p.is_file() else (dir_size(p) if p.is_dir() else None)
    except OSError:
        return None


# ---------------------------------------------------------------- размеры папок в фоне

_SIZES = {"state": "idle", "sizes": {}, "started": 0.0}
_LOCK = threading.Lock()


def start_sizes(items):
    """items: [(ключ, путь)]. Считает размеры по очереди в фоне; повторный вызов в течение минуты ничего не пересчитывает."""
    with _LOCK:
        if _SIZES["state"] == "running" or (_SIZES["state"] == "done" and time.time() - _SIZES["started"] < 60):
            return
        _SIZES.update(state="running", sizes={}, started=time.time())

    def run():
        for key, p in items:
            size = path_size(p)
            with _LOCK:
                _SIZES["sizes"][key] = size
        with _LOCK:
            _SIZES["state"] = "done"

    threading.Thread(target=run, daemon=True, name="about-sizes").start()


def sizes_state():
    with _LOCK:
        return {"state": _SIZES["state"], "sizes": dict(_SIZES["sizes"])}


# ---------------------------------------------------------------- модель пользователя

def personal_status(db_path):
    """Статус личной модели по файлам-сопровождению (без загрузки sklearn): models/personal.json и last_training.json."""
    out = {"file": config.PERSONAL_MODEL.exists()}
    try:
        meta = json.loads((config.MODELS_DIR / "personal.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    try:
        report = json.loads((config.MODELS_DIR / "last_training.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    labels = None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        labels = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        con.close()
    except Exception:
        pass
    n = meta.get("n")
    out.update(labels_now=labels, n=n, trained_at=meta.get("trained_at"),
               weight=(n / (n + config.PERSONAL_K)) if n else None,
               cv_f1=meta.get("cv_f1"), base_f1=meta.get("base_f1"), report=report or None)
    if config.PERSONAL_MODEL.exists():
        out["file_time"] = datetime.fromtimestamp(config.PERSONAL_MODEL.stat().st_mtime).isoformat(timespec="minutes")
    return out


def personal_rows(db_path):
    ps = personal_status(db_path)
    if not ps["file"]:
        return [["Личная модель", "нет (появится после первого обучения на проверенных событиях)"]]
    rows = [["Личная модель", "включена"]]
    rows.append(["Меток при обучении / сейчас в базе",
                 f"{ps['n'] if ps['n'] is not None else '—'} / {ps['labels_now'] if ps['labels_now'] is not None else '—'}"])
    rows.append(["Вес личной модели в баллах", f"{ps['weight']:.2f}  (n / (n + {config.PERSONAL_K}))" if ps["weight"] is not None else "—"])
    when = (ps.get("trained_at") or ps.get("file_time") or "—").replace("T", " ")[:16]
    rows.append(["Последнее обучение", when])
    if ps.get("cv_f1") is not None:
        rows.append(["Точность (проверка по событиям, F1)", f"личная {ps['cv_f1']:.3f} против базовой {ps['base_f1']:.3f}"])
    rep = ps.get("report")
    if rep and "personal" in rep and "base" in rep:
        b, p = rep["base"], rep["personal"]
        rows.append(["Последний отчёт", f"{(rep.get('at') or '').replace('T', ' ')[:16]}: личная угадала {p['hit']} из {p['total']} "
                                        f"(лишних {p['extra']}), базовая {b['hit']} из {b['total']} (лишних {b['extra']}); "
                                        f"{'включена' if rep.get('enabled') else 'не включена'}"])
    elif rep and rep.get("message"):
        rows.append(["Последний отчёт", rep["message"]])
    else:
        rows.append(["Последний отчёт", "появится после следующего обучения"])
    return rows


# ---------------------------------------------------------------- сведения

def info(*, root, db_path, main_root, port, out_dir, log_file, exe_name="BestShots.exe"):
    root = Path(root)
    exe = root / exe_name
    exe_date = datetime.fromtimestamp(exe.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if exe.exists() else "BestShots.exe не найден"
    db_size = path_size(db_path)
    groups = []

    groups.append({"title": "О программе", "rows": [
        ["Название", "BestShots — разбор семейного фотоархива"],
        ["Версия", config.APP_VERSION],
        ["Дата сборки BestShots.exe", exe_date],
        ["Папка программы", str(root)],
        ["База данных", f"{db_path} ({fmt_size(db_size)})"],
        ["Главная папка фото", main_root or "не выбрана"],
        ["Папка экспорта", str(out_dir)],
        ["Журнал", str(log_file)],
    ]})
    groups.append({"title": "Окружение", "rows": [
        ["Python", f"{platform.python_version()} ({platform.python_implementation()}, {platform.architecture()[0]})"],
        ["Windows", platform.platform()],
        ["SQLite", sqlite3.sqlite_version],
        ["Режим", "офлайн: сеть нужна только для разовой загрузки моделей"],
    ]})
    groups.append({"title": "Чтение фото", "rows": [
        ["Pillow", version_of("Pillow", "pillow")],
        ["pillow-heif", version_of("pillow-heif", "pillow_heif")],
        ["Форматы", "JPG/JPEG, PNG, HEIC; RAW (NEF, CR2, ARW, DNG) — по встроенному JPEG-превью"],
    ]})
    cv = version_of("opencv-python-headless", "opencv-python", "opencv-contrib-python")
    groups.append({"title": "Модели и зрение", "rows": [
        ["PyTorch", version_of("torch")],
        ["open_clip", f"{version_of('open_clip_torch', 'open-clip-torch')} — CLIP ViT-L/14 (OpenAI): эмбеддинги кадров"],
        ["LAION Aesthetic Predictor v2", "эстетика: голова поверх CLIP"],
        ["pyiqa", f"{version_of('pyiqa')} — TOPIQ-IAA (эстетика), TOPIQ-NR (техническое качество)"],
        ["OpenCV", f"{cv} — YuNet (поиск лиц), резкость, свет"],
        ["MediaPipe", f"{version_of('mediapipe')} — Face Landmarker (улыбка, моргание)"],
    ]})
    groups.append({"title": "Дубли", "rows": [
        ["Точные копии", "размер → SHA-256 (hashlib)"],
        ["Та же фотография", "pHash и dHash (4 поворота), сравнение 16×16, дата съёмки, CLIP"],
        ["numpy", version_of("numpy")],
    ]})
    groups.append({"title": "Обучение на выборе", "rows": [
        ["scikit-learn", f"{version_of('scikit-learn')} — логистическая регрессия, PCA"],
        ["Проверка", "по событиям (событие не попадает в обучение и проверку одновременно)"],
    ] + personal_rows(db_path)})
    groups.append({"title": "Интерфейс и запуск", "rows": [
        ["Сервер", f"http.server (Python {platform.python_version()}), 127.0.0.1:{port}"],
        ["Интерфейс", "HTML/CSS/JS без фреймворков"],
        ["Окно", f"pywebview {version_of('pywebview')} (Edge WebView2 {webview2_version()})"],
        ["Запуск", f"BestShots.exe (PyInstaller {version_of('pyinstaller')}) → .venv\\Scripts\\pythonw.exe review.py"],
    ]})

    def model(name, path):
        p = Path(path)
        return {"name": name, "file": str(p.relative_to(root)) if root in p.parents else str(p),
                "size": path_size(p) if p.exists() else None, "loaded": p.exists()}

    def first(pattern_dir, glob):
        d = Path(pattern_dir)
        found = sorted(d.glob(glob)) if d.exists() else []
        return found[0] if found else d / glob

    models = [
        model("CLIP ViT-L/14 (OpenAI)", config.MODELS_DIR / "clip" / "models--timm--vit_large_patch14_clip_224.openai"),
        model("LAION Aesthetic Predictor v2", config.AESTHETIC_HEAD),
        model("TOPIQ-IAA (эстетика)", first(config.MODELS_DIR / "torch" / "hub" / "pyiqa", "cfanet_iaa_ava_swin-*.pth")),
        model("TOPIQ-NR (качество)", first(config.MODELS_DIR / "torch" / "hub" / "pyiqa", "cfanet_nr_koniq_res50-*.pth")),
        model("YuNet (лица)", config.YUNET_MODEL),
        model("Face Landmarker (MediaPipe)", config.FACE_MODEL),
        model("Личная модель", config.PERSONAL_MODEL),
    ]
    paths = [("models", config.MODELS_DIR), ("thumbs", config.THUMBS_DIR), ("venv", root / ".venv"), ("db", db_path)]
    start_sizes([(k, str(v)) for k, v in paths])
    return {"groups": groups, "models": models, "version": config.APP_VERSION,
            "size_labels": {"models": "Модели (models)", "thumbs": "Превью (thumbs)", "venv": "Окружение Python (.venv)",
                            "db": "База данных"}}
