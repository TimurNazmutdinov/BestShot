"""BestShots, этап 2: автоматическая оценка кадров (CPU, офлайн).

Читает bestshots.db (после scan.py), считает метрики и пишет таблицы analysis и scores.
Исходные файлы только читаются. Уже проанализированные файлы (тот же sha256) пропускаются.
"""
import argparse
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime

# модели скачиваются один раз; если кэш уже есть — работаем без сети
import config
import personal
import power

if (config.MODELS_DIR / "clip").exists() and any((config.MODELS_DIR / "clip").iterdir()):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", str(config.MODELS_DIR / "torch"))  # веса pyiqa/timm лежат в models/
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis (
    path TEXT PRIMARY KEY,         -- связь с files.path
    sha256 TEXT NOT NULL,          -- для кэша: изменился файл — пересчитываем
    thumb TEXT,
    sharp_center REAL,             -- дисперсия лапласиана по центру (превью 512px)
    sharp_global REAL,
    sharp_faces REAL,              -- по лицам (NULL, если лиц нет)
    over_frac REAL,                -- доля пересвета
    under_frac REAL,               -- доля провалов в тенях
    mean_luma REAL,
    n_faces INTEGER,
    face_method TEXT,              -- каким методом найдены лица (кэш)
    smile REAL,                    -- средняя улыбка по лицам, 0..1
    eyes_open REAL,                -- средняя «открытость» глаз, 0..1
    blink_frac REAL,               -- доля лиц с закрытыми глазами
    aesthetic REAL,                -- LAION Aesthetic v2, ~1..10
    topiq_iaa REAL,                -- pyiqa TOPIQ-IAA (эстетика), ~1..10
    topiq_nr REAL,                 -- pyiqa TOPIQ-NR (техническое качество), ~0..1
    embedding BLOB,                -- CLIP ViT-L/14, float32[768], L2-нормирован
    analyzed_at TEXT
);
CREATE TABLE IF NOT EXISTS scores (
    path TEXT PRIMARY KEY,
    event_id INTEGER,
    group_id INTEGER,
    group_size INTEGER,
    aesthetic_n REAL,
    clip_emotion REAL,             -- вероятность «тёплых» фраз в CLIP zero-shot, 0..1 (по текущим фразам)
    emotion REAL,
    score REAL,                    -- итоговый балл (смесь базового и личного)
    base_score REAL,               -- базовый балл без личной модели
    personal REAL,                 -- вероятность по личной модели (NULL, если модель не включена)
    reject INTEGER,
    reject_reason TEXT,
    selected INTEGER             -- рекомендован системой (брак не рекомендуется никогда)
);
"""


# ---------------------------------------------------------------- модели

def progress(stage, done=0, total=0, msg=""):
    """Строка прогресса для интерфейса (review.py разбирает её и не пишет в лог)."""
    print("PROGRESS " + json.dumps({"stage": stage, "done": done, "total": total, "msg": msg}, ensure_ascii=False),
          flush=True)


def ensure_models():
    """Скачивает недостающие файлы моделей (только при первом запуске)."""
    import urllib.request
    config.MODELS_DIR.mkdir(exist_ok=True)
    for path, url in config.MODEL_URLS.items():
        if path.exists():
            continue
        print(f"Скачиваю {path.name} ...")
        progress("download", msg=f"Загрузка модели {path.name}")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)


def create_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED,
        cache_dir=str(config.MODELS_DIR / "clip"))
    model.eval()
    return model, preprocess


def load_clip():
    import torch
    model, preprocess = create_clip()
    nn = torch.nn
    head = nn.Module()  # архитектура LAION improved-aesthetic-predictor (без нелинейностей, как в оригинале)
    head.layers = nn.Sequential(
        nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
    head.forward = head.layers.forward
    head.load_state_dict(torch.load(config.AESTHETIC_HEAD, map_location="cpu", weights_only=True))
    head.eval()
    return model, preprocess, head


def load_iqa():
    """pyiqa: topiq_iaa (эстетика) и topiq_nr (техническое качество), CPU."""
    import pyiqa
    return {name: pyiqa.create_metric(name, device="cpu") for name in ("topiq_iaa", "topiq_nr")}


def iqa_scores(iqa, pil):
    import torch
    import torchvision.transforms.functional as TF
    x = TF.to_tensor(pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return {k: float(m(x)) for k, m in iqa.items()}


def load_faces():
    """YuNet (OpenCV) находит лица, MediaPipe Face Landmarker оценивает выражение на увеличенной вырезке."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    detector = cv2.FaceDetectorYN.create(str(config.YUNET_MODEL), "", (320, 320), config.FACE_SCORE_MIN, 0.3, 5000)
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.FACE_MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_faces=3, output_face_blendshapes=True)
    # нативные библиотеки MediaPipe пишут служебные предупреждения прямо в stderr — глушим на время создания
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as nul:
            os.dup2(nul.fileno(), 2)
            landmarker = vision.FaceLandmarker.create_from_options(opts)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    return detector, mp, landmarker


# ---------------------------------------------------------------- метрики

def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def open_oriented(path, size):
    """Открывает файл только на чтение, применяет EXIF Orientation, уменьшает до size."""
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (size, size))  # быстрое декодирование с понижением масштаба
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    return im


def technical(thumb):
    gray = cv2.cvtColor(np.asarray(thumb), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    center = gray[h // 4: h * 3 // 4, w // 4: w * 3 // 4]
    return {
        "sharp_center": lap_var(center), "sharp_global": lap_var(gray),
        "over_frac": float((gray > config.CLIP_HIGH).mean()),
        "under_frac": float((gray < config.CLIP_LOW).mean()),
        "mean_luma": float(gray.mean()),
    }


def expression(mp, landmarker, arr, box):
    """Улыбка и моргание для одного лица: вырезка с запасом, увеличение, MediaPipe. None, если не получилось."""
    h, w = arr.shape[:2]
    x, y, bw, bh = box
    m = config.FACE_CROP_MARGIN
    x0, x1 = max(int(x - m * bw), 0), min(int(x + bw * (1 + m)), w)
    y0, y1 = max(int(y - m * bh), 0), min(int(y + bh * (1 + m)), h)
    crop = arr[y0:y1, x0:x1]
    scale = config.FACE_CROP_SIZE / max(crop.shape[:2])
    crop = np.ascontiguousarray(cv2.resize(crop, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    if not res.face_landmarks:
        return None
    best = min(zip(res.face_landmarks, res.face_blendshapes),  # лицо, ближайшее к центру вырезки
               key=lambda t: (np.mean([p.x for p in t[0]]) - 0.5) ** 2 + (np.mean([p.y for p in t[0]]) - 0.5) ** 2)
    sc = {c.category_name: c.score for c in best[1]}
    blink = (sc["eyeBlinkLeft"] + sc["eyeBlinkRight"]) / 2
    return (sc["mouthSmileLeft"] + sc["mouthSmileRight"]) / 2, blink


def face_metrics(faces_tools, work):
    detector, mp, landmarker = faces_tools
    arr = np.asarray(work)
    h, w = arr.shape[:2]
    detector.setInputSize((w, h))
    _, found = detector.detect(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    n, smiles, blinks, sharps = 0, [], [], []
    for f in (found if found is not None else []):
        x, y, bw, bh = [float(v) for v in f[:4]]
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + bw), w), min(int(y + bh), h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        n += 1
        sharps.append(lap_var(cv2.resize(gray[y0:y1, x0:x1], (128, 128), interpolation=cv2.INTER_AREA)))
        if min(x1 - x0, y1 - y0) >= config.FACE_MIN_PX:
            ex = expression(mp, landmarker, arr, (x, y, bw, bh))
            if ex:
                smiles.append(ex[0])
                blinks.append(ex[1])
    out = {"n_faces": n, "face_method": config.FACE_METHOD, "smile": None, "eyes_open": None,
           "blink_frac": None, "sharp_faces": float(np.mean(sharps)) if sharps else None}
    if smiles:
        out.update(smile=float(np.mean(smiles)), eyes_open=float(1 - np.mean(blinks)),
                   blink_frac=float(np.mean([b > config.BLINK_THRESHOLD for b in blinks])))
    return out


# ---------------------------------------------------------------- этап A: анализ файлов

def analyze_files(con, force, limit):
    todo, todo_faces, todo_iqa = [], [], []
    cached = {r[0]: (r[1], r[2], r[3], r[4]) for r in con.execute(
        "SELECT path, sha256, face_method, topiq_iaa, thumb FROM analysis")}
    for path, sha, kind in con.execute("SELECT path, sha256, kind FROM files ORDER BY date_taken, path"):
        if kind != "photo":
            continue
        if not sha:
            sys.exit("В files нет sha256 — сначала запустите scan.py (этап 1 обновлён).")
        if force or path not in cached or cached[path][0] != sha:
            todo.append((path, sha))
        else:
            if cached[path][1] != config.FACE_METHOD:
                todo_faces.append((path, sha))  # CLIP и техника посчитаны, пересчитываем только лица
            if cached[path][2] is None:
                todo_iqa.append((path, cached[path][3]))  # нет метрик pyiqa
    n_all = con.execute("SELECT COUNT(*) FROM files WHERE kind='photo'").fetchone()[0]
    n_todo = len(todo)
    if limit:
        todo = todo[:limit]
    print(f"Фото: {n_all}, уже посчитано: {n_all - n_todo - len(todo_faces)}, к анализу: {len(todo)}"
          + (f", лица пересчитать: {len(todo_faces)}" if todo_faces else "")
          + (f", pyiqa: {len(todo_iqa)}" if todo_iqa else ""))
    if not todo and not todo_faces and not todo_iqa:
        return

    ensure_models()
    tools = load_faces()
    progress("download", msg="Подготовка моделей (при первом запуске — загрузка, нужен интернет)")
    iqa = load_iqa() if todo or todo_iqa else None

    if todo_iqa:
        for i, (path, thumb) in enumerate(todo_iqa, 1):
            try:
                tp = config.ROOT / thumb if thumb else None
                img = Image.open(tp) if tp and tp.exists() else open_oriented(path, config.THUMB_SIZE)
                m = iqa_scores(iqa, img)
                con.execute("UPDATE analysis SET topiq_iaa=?, topiq_nr=? WHERE path=?", (m["topiq_iaa"], m["topiq_nr"], path))
            except Exception as e:
                print(f"  pyiqa: пропущен {path}: {type(e).__name__}: {e}", file=sys.stderr)
            if i % 5 == 0 or i == len(todo_iqa):
                con.commit()
                progress("iqa", i, len(todo_iqa))

    if todo_faces:
        for i, (path, sha) in enumerate(todo_faces, 1):
            try:
                m = face_metrics(tools, open_oriented(path, config.FACE_WORK_SIZE))
                cols = list(m)
                con.execute(f"UPDATE analysis SET {','.join(c + '=?' for c in cols)} WHERE path=?",
                            [m[c] for c in cols] + [path])
            except Exception as e:
                print(f"  лица: пропущен {path}: {type(e).__name__}: {e}", file=sys.stderr)
            if i % 5 == 0 or i == len(todo_faces):
                con.commit()
                progress("faces", i, len(todo_faces))
    if not todo:
        return

    import torch
    progress("download", msg="Подготовка модели CLIP (при первом запуске — загрузка ~1,7 ГБ)")
    clip, preprocess, head = load_clip()
    progress("analyze", 0, len(todo))
    config.THUMBS_DIR.mkdir(exist_ok=True)
    failed = 0

    def flush(batch):
        if not batch:
            return
        x = torch.stack([b["tensor"] for b in batch])
        with torch.no_grad():
            e = clip.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
            aes = head(e).squeeze(-1)
        for b, emb, a in zip(batch, e.numpy(), aes.tolist()):
            m = b["m"]
            m.update(aesthetic=a, embedding=emb.astype(np.float32).tobytes(),
                     analyzed_at=datetime.now().isoformat(timespec="seconds"))
            cols = list(m)
            con.execute(f"INSERT OR REPLACE INTO analysis ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [m[c] for c in cols])
        con.commit()

    prime_text_cache(clip)
    live = {"t": time.time(), "n": 0}

    def maybe_live_score(done_count):
        """События появляются в «Отборе» не в конце порции, а по мере анализа: раз в LIVE['every'] секунд."""
        if done_count - live["n"] < LIVE["min"] or time.time() - live["t"] < LIVE["every"]:
            return
        try:
            score_all(con, freeze_decided=True)  # проверенные события не трогаем
        except Exception as e:  # промежуточный пересчёт не должен ронять анализ
            print(f"  промежуточный пересчёт баллов не удался: {type(e).__name__}: {e}", file=sys.stderr)
        live.update(t=time.time(), n=done_count)

    batch = []
    for i, (path, sha) in enumerate(todo, 1):
        try:
            work = open_oriented(path, config.FACE_WORK_SIZE)
            thumb = work.copy()
            thumb.thumbnail((config.THUMB_SIZE, config.THUMB_SIZE), Image.LANCZOS)
            tpath = config.THUMBS_DIR / f"{sha[:16]}.jpg"
            thumb.save(tpath, quality=88)
            m = {"path": path, "sha256": sha, "thumb": str(tpath.relative_to(config.ROOT))}
            m.update(technical(thumb))
            m.update(face_metrics(tools, work))
            m.update(iqa_scores(iqa, thumb))
            batch.append({"m": m, "tensor": preprocess(thumb)})
        except Exception as e:
            failed += 1
            print(f"  пропущен {path}: {type(e).__name__}: {e}", file=sys.stderr)
        if len(batch) >= config.BATCH:
            flush(batch)
            batch = []
            maybe_live_score(i)
        if i % 3 == 0 or i == len(todo):
            progress("analyze", i, len(todo))
    flush(batch)
    if failed:
        print(f"Не удалось обработать: {failed}")


# ---------------------------------------------------------------- этап B: баллы, группы, выбор

def robust_norm(values):
    """Нормирует в 0..1 по 5–95 перцентилям (устойчиво к выбросам)."""
    v = np.asarray(values, dtype=float)
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    if hi - lo < 1e-9:
        return np.full_like(v, 0.5)
    return np.clip((v - lo) / (hi - lo), 0, 1)


_TEXT = {}  # текстовые признаки наборов фраз: считаются один раз на процесс
LIVE = {"every": config.LIVE_SCORE_EVERY_S, "min": config.LIVE_SCORE_MIN_NEW}


def _text_features(model, pos, neg):
    import open_clip
    import torch
    key = tuple(pos + neg)
    if key not in _TEXT:
        tok = open_clip.get_tokenizer(config.CLIP_MODEL)
        with torch.no_grad():
            t = model.encode_text(tok(list(key)))
            _TEXT[key] = (t / t.norm(dim=-1, keepdim=True), float(model.logit_scale.exp()))
    return _TEXT[key]


def prime_text_cache(model):
    """Текстовые признаки фраз по уже загруженной модели CLIP (анализ не грузит вторую копию модели)."""
    _text_features(model, config.POSITIVE_PROMPTS, config.NEGATIVE_PROMPTS)
    _text_features(model, config.PEOPLE_POSITIVE_PROMPTS, config.PEOPLE_NEGATIVE_PROMPTS)


def clip_probs(emb, pos, neg):
    """Доля «положительных» фраз в CLIP zero-shot по сохранённым эмбеддингам (модель грузится один раз на процесс)."""
    import torch
    if tuple(pos + neg) not in _TEXT:
        model, _ = create_clip()
        _text_features(model, pos, neg)
    t, scale = _text_features(None, pos, neg)
    with torch.no_grad():
        logits = scale * torch.from_numpy(emb) @ t.T
        return logits.softmax(dim=-1)[:, :len(pos)].sum(dim=-1).numpy()


def clip_emotion_probs(emb):
    return clip_probs(emb, config.POSITIVE_PROMPTS, config.NEGATIVE_PROMPTS)


def reject_reason(a):
    face_ok = a["sharp_faces"] is not None and a["sharp_faces"] >= config.BLUR_MAX
    reasons = []
    if a["sharp_center"] < config.BLUR_MAX and not face_ok:
        reasons.append("смаз")
    if a["mean_luma"] < config.DARK_LUMA_MAX or a["under_frac"] > config.DARK_FRAC_MAX:
        reasons.append("тёмный")
    if a["over_frac"] > config.BRIGHT_FRAC_MAX:
        reasons.append("пересвет")
    if a.get("topiq_nr") is not None and a["topiq_nr"] < config.TECH_NR_MIN:
        reasons.append("качество")
    return ",".join(reasons)


def make_groups(items):
    """items — кадры одного события по времени; связываем похожие и близкие по времени."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # кадры упорядочены по времени: каждый сравнивается только с соседями в окне GROUP_MAX_GAP_S (без матрицы N×N,
    # которая для события на тысячи кадров занимала гигабайты памяти)
    emb = np.stack([it["emb"] for it in items])
    ts = np.array([it["ts"] for it in items])
    for i in range(n):
        end = int(np.searchsorted(ts, ts[i] + config.GROUP_MAX_GAP_S, side="right"))
        if end > i + 1:
            sims = emb[i + 1:end] @ emb[i]
            for k in np.nonzero(sims >= config.SIM_THRESHOLD)[0]:
                parent[find(i + 1 + int(k))] = find(i)
    return [find(i) for i in range(n)]


def is_portrait(r):
    w, h = r["width"] or 0, r["height"] or 0
    if r["orientation"] in (5, 6, 7, 8):
        w, h = h, w
    return h > w


def select_in_event(items):
    """Рекомендация внутри события: брак исключён; лучший кадр каждой группы с повторами — всегда;
    остальные места (одиночные кадры и «разнообразные» кадры групп) — по баллу до цели TARGET_SELECT_FRACTION."""
    by_group = defaultdict(list)
    for r in items:
        r["selected"] = False
        if not r["reject"]:
            by_group[r["group_id"]].append(r)
    mandatory, pool = [], []
    for g in by_group.values():
        g.sort(key=lambda r: -r["score"])
        best = g[0]
        (mandatory if best["group_size"] > 1 else pool).append(best)
        chosen = [best]
        for c in g[1:]:
            if len(chosen) >= config.DIVERSITY_MAX_PER_GROUP or c["score"] < config.DIVERSITY_MIN_SCORE_RATIO * best["score"]:
                continue
            if all(float(c["emb"] @ x["emb"]) < config.DIVERSITY_SIM_MAX or is_portrait(c) != is_portrait(x) for x in chosen):
                chosen.append(c)
                pool.append(c)
    for r in mandatory:
        r["selected"] = True
    target = max(1, round(config.TARGET_SELECT_FRACTION * len(items)))
    taken = len(mandatory)
    for r in sorted(pool, key=lambda r: -r["score"]):
        if taken >= target:
            break
        r["selected"] = True
        taken += 1


def score_all(con, freeze_decided=False):
    q = """SELECT f.path, f.event_id, f.date_taken, f.width, f.height, f.orientation, a.* FROM files f JOIN analysis a ON a.path = f.path
           WHERE f.kind='photo' ORDER BY f.date_taken, f.path"""
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(q)]
    con.row_factory = None
    if not rows:
        return []
    # точные копии (тот же sha256 в одном событии) не должны соревноваться друг с другом и раздувать группы:
    # считаем по одному представителю, копии получают его результат
    seen, dups = {}, []
    uniq = []
    for r in rows:
        key = (r["event_id"], r["sha256"])
        if r["sha256"] and key in seen:
            dups.append((r, seen[key]))
        else:
            seen[key] = r
            uniq.append(r)
    rows_all, rows = rows, uniq
    for r in rows:
        r["emb"] = np.frombuffer(r["embedding"], dtype=np.float32)
        r["ts"] = datetime.fromisoformat(r["date_taken"]).timestamp()
        r["reason"] = reject_reason(r)
        r["reject"] = bool(r["reason"])
    for r, p in zip(rows, clip_emotion_probs(np.stack([r["emb"] for r in rows]))):
        r["clip_emotion"] = float(p)

    # группы похожих: внутри события
    events = defaultdict(list)
    for r in rows:
        events[r["event_id"]].append(r)
    gid = 0
    for items in events.values():
        ids = {}
        for r, root in zip(items, make_groups(items)):
            if root not in ids:
                gid += 1
                ids[root] = gid
            r["group_id"] = ids[root]
    members = defaultdict(list)
    for r in rows:
        members[r["group_id"]].append(r)
    for g in members.values():
        # кадры группы оцениваем одинаково: лица нашлись хотя бы в одном — вся группа «с лицами»
        has_faces = any(r["n_faces"] for r in g)
        for r in g:
            r["group_size"] = len(g)
            r["group_faces"] = has_faces

    if config.EXCLUDE_NO_PEOPLE:  # люди есть: лица в самом кадре, либо CLIP видит людей; иначе — «брак: без людей»
        # (лица группы не учитываем: в группу пейзажей попадает кадр с ложным «лицом» на статуе, и вся группа считалась бы с людьми)
        people = clip_probs(np.stack([r["emb"] for r in rows]), config.PEOPLE_POSITIVE_PROMPTS, config.PEOPLE_NEGATIVE_PROMPTS)
        for r, p in zip(rows, people):
            r["people"] = float(p)
            if not r["n_faces"] and p < config.PEOPLE_CLIP_MIN:
                r["reason"] = ",".join(x for x in (r["reason"], "без людей") if x)
                r["reject"] = True

    # нормировка по всему архиву среди не-брака (5–95 перцентили)
    ok = [r for r in rows if not r["reject"]] or rows

    def span(key, f=lambda v: v):
        vals = [f(r[key]) for r in ok if r.get(key) is not None]
        return tuple(np.percentile(vals, [5, 95])) if vals else (0.0, 1.0)

    lo_a, hi_a = span("aesthetic")
    lo_i, hi_i = span("topiq_iaa")
    lo_n, hi_n = span("topiq_nr")
    lo_c, hi_c = span("clip_emotion")
    lo_s, hi_s = span("sharp_center", math.log1p)

    def nz(v, lo, hi):
        if v is None:
            return 0.5
        return float(np.clip((v - lo) / (hi - lo), 0, 1)) if hi - lo > 1e-9 else 0.5

    base_w = {"laion": config.W_LAION, "iaa": config.W_TOPIQ_IAA, "nr": config.W_TOPIQ_NR}
    for r in rows:
        r["aesthetic_n"] = nz(r["aesthetic"], lo_a, hi_a)
        emo = nz(r["clip_emotion"], lo_c, hi_c)
        if r["smile"] is not None:  # улыбка — только бонус
            emo = min(1.0, emo + config.SMILE_BONUS * min(r["smile"] / 0.6, 1.0))
        if r["blink_frac"] is not None:
            emo -= config.BLINK_PENALTY * r["blink_frac"]
        r["emotion"] = max(emo, 0.0)
        w_emo = config.W_EMOTION if r["group_faces"] else config.W_EMOTION_NO_FACES
        k = (1 - w_emo) / sum(base_w.values())
        r["score"] = (k * base_w["laion"] * r["aesthetic_n"] + k * base_w["iaa"] * nz(r["topiq_iaa"], lo_i, hi_i)
                      + k * base_w["nr"] * nz(r["topiq_nr"], lo_n, hi_n) + w_emo * r["emotion"]
                      + config.W_SHARPNESS * nz(math.log1p(r["sharp_center"]), lo_s, hi_s))

    # личная модель: итоговый балл = (1 − w) · базовый + w · вероятность, w = n / (n + K)
    for r in rows:
        r["base_score"] = r["score"]
        r["personal"] = None
    bundle = personal.load_model()
    if bundle:
        w = personal.personal_weight(bundle["n"])
        for r, p in zip(rows, personal.predict(bundle, rows)):
            r["personal"] = float(p)
            r["score"] = (1 - w) * r["base_score"] + w * float(p)

    for items in events.values():
        select_in_event(items)

    for d, src in dups:
        keep = {k: d[k] for k in ("path", "date_taken", "width", "height", "orientation")}
        d.update(src)
        d.update(keep)
    rows = rows_all

    if freeze_decided:  # события с решениями пользователя не пересчитываем: оставляем прежнюю рекомендацию
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        decided = set()
        if "decisions" in tables:
            decided |= {r[0] for r in con.execute(
                "SELECT s.event_id FROM scores s JOIN decisions d ON d.path = s.path")}
        if "event_reviews" in tables:
            decided |= {r[0] for r in con.execute("SELECT event_id FROM event_reviews WHERE reviewed = 1")}
        old = dict(con.execute("SELECT path, selected FROM scores"))
        for r in rows:
            if r["event_id"] in decided and r["path"] in old:
                r["selected"] = bool(old[r["path"]])

    with con:
        con.execute("DELETE FROM scores WHERE path NOT IN (SELECT path FROM analysis)")
        con.executemany(
            "INSERT OR REPLACE INTO scores (path, event_id, group_id, group_size, aesthetic_n, clip_emotion, emotion, "
            "score, base_score, personal, reject, reject_reason, selected) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(r["path"], r["event_id"], r["group_id"], r["group_size"], r["aesthetic_n"], r["clip_emotion"],
              r["emotion"], r["score"], r["base_score"], r["personal"], int(r["reject"]), r["reason"],
              int(r["selected"])) for r in rows])
    return rows


def report(rows):
    n = len(rows)
    events = {r["event_id"] for r in rows}
    groups = {r["group_id"] for r in rows}
    multi = {r["group_id"] for r in rows if r["group_size"] > 1}
    rej = [r for r in rows if r["reject"]]
    reasons = defaultdict(int)
    for r in rej:
        for x in r["reason"].split(","):
            reasons[x] += 1
    print(f"\n=== Итоги этапа 2 ===")
    print(f"Кадров: {n}, событий: {len(events)}")
    print(f"Групп похожих: {len(groups)} (из них с повторами: {len(multi)})")
    print(f"Брак: {len(rej)}" + (" (" + ", ".join(f"{k}: {v}" for k, v in reasons.items()) + ")" if rej else ""))
    sel = sum(r["selected"] for r in rows)
    print(f"Рекомендовано: {sel} = {100 * sel / n:.0f}% кадров")
    aes = [r["aesthetic"] for r in rows]
    print(f"Эстетика: мин {min(aes):.2f} / медиана {float(np.median(aes)):.2f} / макс {max(aes):.2f}")


def main():
    power.prefer_performance()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="BestShots: оценка кадров")
    ap.add_argument("--db", default="bestshots.db")
    ap.add_argument("--force", action="store_true", help="пересчитать всё, игнорируя кэш")
    ap.add_argument("--limit", type=int, default=0, help="анализировать не больше N новых файлов")
    ap.add_argument("--keep-decided", action="store_true",
                    help="не менять рекомендацию в событиях, где есть решения пользователя")
    ap.add_argument("--score-only", action="store_true", help="не анализировать, только пересчитать баллы/группы")
    ap.add_argument("--live-score-every", type=float, default=config.LIVE_SCORE_EVERY_S,
                    help="период промежуточного пересчёта баллов во время анализа, с (0 — только в конце)")
    ap.add_argument("--live-score-min", type=int, default=config.LIVE_SCORE_MIN_NEW,
                    help="минимум новых фото между промежуточными пересчётами")
    args = ap.parse_args()
    LIVE.update(every=args.live_score_every if args.live_score_every > 0 else 10 ** 9, min=args.live_score_min)
    if not os.path.exists(args.db):
        sys.exit(f"Нет базы {args.db}: сначала запустите scan.py")
    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)
    if "clip_emotion" not in {r[1] for r in con.execute("PRAGMA table_info(scores)")}:
        con.execute("ALTER TABLE scores ADD COLUMN clip_emotion REAL")  # миграция базы прежней версии
    have = {r[1] for r in con.execute("PRAGMA table_info(analysis)")}
    for col, typ in (("face_method", "TEXT"), ("topiq_iaa", "REAL"), ("topiq_nr", "REAL")):
        if col not in have:  # миграция базы прежней версии: недостающие метрики досчитаются
            con.execute(f"ALTER TABLE analysis ADD COLUMN {col} {typ}")
    have = {r[1] for r in con.execute("PRAGMA table_info(scores)")}
    for col in ("base_score", "personal"):
        if col not in have:
            con.execute(f"ALTER TABLE scores ADD COLUMN {col} REAL")
    con.executescript("CREATE INDEX IF NOT EXISTS idx_scores_event ON scores(event_id);"
                      "CREATE INDEX IF NOT EXISTS idx_scores_group ON scores(group_id);")
    if not args.score_only:
        analyze_files(con, args.force, args.limit)
    # анализ новых фото не меняет рекомендации в проверенных событиях; полный пересчёт — только --score-only
    rows = score_all(con, args.keep_decided or not args.score_only)
    if rows:
        report(rows)
    con.close()


if __name__ == "__main__":
    main()
