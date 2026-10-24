"""BestShots: поиск дубликатов (фоновое задание, запускается из review.py).

Оригиналы только читаются. Результаты и кэш хэшей — в БД (таблицы dup_*), повторный запуск считает только новое.
  1. «Точные копии»: группировка по размеру → sha256 только у файлов одинакового размера (фото, RAW и видео).
  2. «Та же фотография»: перцептивные хэши (pHash + dHash по превью с учётом Orientation, с поворотами 90/180/270),
     поиск соседей векторизованным сравнением, подтверждение датой съёмки (±2 с) и косинусной близостью CLIP.
     Порог строгий: серийные кадры дублями не считаются.
  Видео — только точные копии. Совпадение имён — лишь подсказка; пары RAW + JPEG дублями не считаются.
"""
import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import dupes
import power

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

TAG_ORIENTATION, TAG_EXIF_IFD, TAG_DT_ORIGINAL = 0x0112, 0x8769, 0x9003


def progress(stage, done=0, total=0, msg=""):
    print("PROGRESS " + json.dumps({"stage": stage, "done": done, "total": total, "msg": msg}, ensure_ascii=False),
          flush=True)


# ---------------------------------------------------------------- хэши изображения

_DCT = None


def _dct_matrix(n=32):
    global _DCT
    if _DCT is None:
        k = np.arange(n)
        m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2.0 / n)
        m[0] /= np.sqrt(2.0)
        _DCT = m
    return _DCT


def _signed(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def phash_bits(gray64):
    """pHash 64 бита: DCT 32×32, низкие частоты 8×8 (без DC) против медианы."""
    a = gray64.astype(np.float64).reshape(32, 2, 32, 2).mean(axis=(1, 3))
    m = _dct_matrix()
    low = (m @ a @ m.T)[:8, :8].ravel()
    med = np.median(low[1:])
    bits = low > med
    return _signed(int("".join("1" if b else "0" for b in bits), 2))


def dhash_bits(gray64):
    """dHash 64 бита: разности соседних пикселей сетки 9×8."""
    small = np.asarray(Image.fromarray(gray64.astype(np.uint8)).resize((9, 8), Image.BOX), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return _signed(int("".join("1" if b else "0" for b in bits), 2))


def open_preview(path, ext, want=256):
    if ext in dupes.RAW_EXT:
        im = Image.open(io.BytesIO(dupes.embedded_jpeg(path)))
    else:
        im = Image.open(path)
    return im


def hash_file(path):
    """Хэши одного изображения. Возвращает словарь (или {'error': ...}); работает в процессах-воркерах."""
    try:
        ext = os.path.splitext(path)[1].lower()
        im = open_preview(path, ext)
        w, h = im.size
        orientation, date = None, None
        try:
            ex = im.getexif()
            orientation = int(ex.get(TAG_ORIENTATION) or 0) or None
            raw_date = ex.get_ifd(TAG_EXIF_IFD).get(TAG_DT_ORIGINAL)
            if raw_date:
                date = datetime.strptime(str(raw_date).strip()[:19], "%Y:%m:%d %H:%M:%S").isoformat()
        except Exception:
            pass
        if im.format == "JPEG":
            im.draft("L", (256, 256))
        im = ImageOps.exif_transpose(im)
        if orientation in (5, 6, 7, 8):
            w, h = h, w
        # квадрат 64×64 без сохранения пропорций: поворот на 90° тогда не меняет хэш, а лишь переставляет биты
        gray = np.asarray(im.convert("L").resize((64, 64), Image.BOX))
        out = {"width": w, "height": h, "exif_date": date, "has_exif": 1 if date else 0, "orientation": orientation,
               "tiny": np.rint(gray.reshape(16, 4, 16, 4).mean(axis=(1, 3))).astype(np.uint8).tobytes()}
        for r in range(4):
            a = np.rot90(gray, r)
            out[f"ph{r}"] = phash_bits(a)
            out[f"dh{r}"] = dhash_bits(a)
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_job(path):
    try:
        return path, sha256_of(path), None
    except OSError as e:
        return path, None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- сравнение хэшей

def hamming(a, b):
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


def find_hash_pairs(items, ph_max, dh_max, on_progress=None):
    """items: [(id, [ph0..ph3], [dh0..dh3])]. Возвращает [(i, j, rot, dph, ddh)] для i<j:
    хэш i, повёрнутый на rot·90°, близок к хэшу j. Векторизованно (numpy.bitwise_count)."""
    n = len(items)
    if n < 2:
        return []
    P0 = np.array([it[1][0] for it in items], dtype=np.int64).view(np.uint64)
    D0 = np.array([it[2][0] for it in items], dtype=np.int64).view(np.uint64)
    PQ = np.array([it[1] for it in items], dtype=np.int64).view(np.uint64)
    DQ = np.array([it[2] for it in items], dtype=np.int64).view(np.uint64)
    pairs = {}
    chunk = max(1, int(6_000_000 // n))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        idx = np.arange(start, stop)
        later = np.arange(n)[None, :] > idx[:, None]  # только пары i < j
        for r in range(4):
            dp = np.bitwise_count(PQ[start:stop, r][:, None] ^ P0[None, :])
            hit = (dp <= ph_max) & later
            ii, jj = np.nonzero(hit)
            if ii.size:
                dd = np.bitwise_count(DQ[start + ii, r] ^ D0[jj])
                for a, b, p, d in zip(ii, jj, dp[ii, jj], dd):
                    if d <= dh_max:
                        key = (int(start + a), int(b))
                        cur = pairs.get(key)
                        if cur is None or (int(p) + int(d)) < cur[3] + cur[4]:
                            pairs[key] = (key[0], key[1], r, int(p), int(d))
        if on_progress:
            on_progress(stop, n)
    return list(pairs.values())


def pixel_check(tiny_a, tiny_b, rot):
    """Корреляция и средняя абс. разность нормированных копий 16×16 (a повёрнут на rot·90°). Серийные кадры
    отличаются заметно, пересохранённые/уменьшенные копии — практически нет."""
    a = np.rot90(np.frombuffer(tiny_a, dtype=np.uint8).reshape(16, 16).astype(np.float64), rot)
    b = np.frombuffer(tiny_b, dtype=np.uint8).reshape(16, 16).astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean()), float(np.abs(a - b).mean())


# ---------------------------------------------------------------- CLIP (только для кандидатов)

def clip_embeddings(paths, on_progress=None, rotations=None):
    """Нормированные CLIP-эмбеддинги ViT-L/14 тех же весов, что и в анализе (для сравнения с analysis.embedding)."""
    import open_clip
    import torch
    import config
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED, cache_dir=str(config.MODELS_DIR / "clip"))
    model.eval()
    out = {}
    batch, names = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            e = model.encode_image(torch.stack(batch))
            e = e / e.norm(dim=-1, keepdim=True)
        for p, v in zip(names, e.numpy()):
            out[p] = v.astype(np.float32)
        batch.clear()
        names.clear()

    for i, p in enumerate(paths, 1):
        try:
            ext = os.path.splitext(p)[1].lower()
            im = open_preview(p, ext)
            if im.format == "JPEG":
                im.draft("RGB", (600, 600))
            im = ImageOps.exif_transpose(im).convert("RGB")
            if rotations and rotations.get(p):  # повёрнутая копия: сравниваем кадр, повёрнутый обратно
                im = im.rotate(90 * rotations[p], expand=True)
            im.thumbnail((512, 512), Image.LANCZOS)
            batch.append(preprocess(im))
            names.append(p)
        except Exception:
            continue
        if len(batch) >= 8:
            flush()
        if on_progress and i % 4 == 0:
            on_progress(i, len(paths))
    flush()
    if on_progress:
        on_progress(len(paths), len(paths))
    return out


# ---------------------------------------------------------------- основной конвейер

def main():
    power.prefer_performance()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="BestShots: поиск дубликатов")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--db", type=Path, default=Path("bestshots.db"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    folder = str(args.folder.resolve())
    if not os.path.isdir(folder):
        sys.exit(f"Папка не найдена: {folder}")

    con = sqlite3.connect(args.db, timeout=30)
    con.row_factory = sqlite3.Row
    dupes.ensure_schema(con)
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None

    # --- 1. список файлов
    excluded = dupes.excluded_paths(con)
    known = {r["path"]: r for r in con.execute("SELECT * FROM dup_files WHERE substr(path,1,?)=?",
                                               (len(dupes.prefix(folder)), dupes.prefix(folder)))}
    seen, upserts, last = set(), [], time.time()
    count = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not dupes.skip_dir(os.path.join(dirpath, d), d, excluded))
        for name in sorted(filenames):
            kind = dupes.kind_of(name)
            if not kind:
                continue
            p = os.path.join(dirpath, name)
            try:
                st = os.stat(p)
            except OSError as e:
                upserts.append((p, 0, 0.0, kind, f"{type(e).__name__}: {e}"))
                seen.add(p)
                continue
            seen.add(p)
            old = known.get(p)
            if old and old["size"] == st.st_size and abs(old["mtime"] - st.st_mtime) < 1e-3:
                continue  # без изменений — кэш хэшей сохраняется
            upserts.append((p, st.st_size, st.st_mtime, kind, None))
            count += 1
        if time.time() - last > 1:
            progress("dup_list", len(seen), 0, "Поиск файлов")
            last = time.time()
    with con:
        for p, size, mtime, kind, err in upserts:  # новая запись или файл изменился: хэши сбрасываются
            con.execute("DELETE FROM dup_files WHERE path = ?", (p,))
            stem, _ext = os.path.splitext(os.path.basename(p))
            con.execute("INSERT INTO dup_files (path, dir, name, stem, size, mtime, kind, read_error) VALUES (?,?,?,?,?,?,?,?)",
                        (p, os.path.dirname(p).lower(), os.path.basename(p).lower(), stem.lower(), size, mtime, kind, err))
        gone = [p for p in known if p not in seen]
        con.executemany("DELETE FROM dup_files WHERE path = ?", [(p,) for p in gone])
    progress("dup_list", len(seen), len(seen), "Поиск файлов")

    rows = [dict(r) for r in con.execute("SELECT * FROM dup_files WHERE substr(path,1,?)=? AND read_error IS NULL",
                                         (len(dupes.prefix(folder)), dupes.prefix(folder)))]
    errors_total = con.execute("SELECT COUNT(*) FROM dup_files WHERE substr(path,1,?)=? AND read_error IS NOT NULL",
                               (len(dupes.prefix(folder)), dupes.prefix(folder))).fetchone()[0]

    # --- 2. точные копии: sha256 только у файлов одинакового размера
    by_size = defaultdict(list)
    for r in rows:
        if r["size"] > 0:
            by_size[r["size"]].append(r)
    todo = [r for g in by_size.values() if len(g) > 1 for r in g if not r["sha256"]]
    try:  # sha256 уже посчитан этапом 1 основного сканера — не читаем файл второй раз
        db_sha = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT path, size, mtime, sha256 FROM files WHERE sha256 IS NOT NULL")}
    except sqlite3.OperationalError:
        db_sha = {}
    need = []
    for r in todo:
        k = db_sha.get(r["path"])
        if k and k[0] == r["size"] and abs((k[1] or 0) - r["mtime"]) < 1e-3:
            r["sha256"] = k[2]
            con.execute("UPDATE dup_files SET sha256=? WHERE path=?", (k[2], r["path"]))
        else:
            need.append(r)
    con.commit()
    done = 0
    progress("dup_hash", 0, len(need), "Контрольные суммы")
    it = pool.map(_sha_job, [r["path"] for r in need], chunksize=4) if pool and len(need) > 8 else map(_sha_job, [r["path"] for r in need])
    path_rows = {r["path"]: r for r in rows}
    for p, sha, err in it:
        con.execute("UPDATE dup_files SET sha256=?, read_error=? WHERE path=?", (sha, err, p))
        if sha:
            path_rows[p]["sha256"] = sha
        else:
            path_rows[p]["read_error"] = err
        done += 1
        if done % 25 == 0:
            con.commit()
            progress("dup_hash", done, len(need), "Контрольные суммы")
    con.commit()
    progress("dup_hash", len(need), len(need), "Контрольные суммы")
    rows = [r for r in rows if not r["read_error"]]

    exact_groups = defaultdict(list)  # sha256 -> файлы
    for r in rows:
        if r["sha256"]:
            exact_groups[r["sha256"]].append(r)
    exact_groups = {k: v for k, v in exact_groups.items() if len(v) > 1}
    in_exact = {r["path"] for v in exact_groups.values() for r in v}

    # --- 3. перцептивные хэши (фото и RAW): один представитель на группу точных копий
    reps = {}
    for r in rows:
        if r["kind"] not in ("photo", "raw"):
            continue
        key = r["sha256"] if r["path"] in in_exact else r["path"]
        if key not in reps:
            reps[key] = r
    need = [r for r in reps.values() if r.get("ph0") is None]
    progress("dup_phash", 0, len(need), "Перцептивные хэши")
    done = 0
    paths = [r["path"] for r in need]
    results = pool.map(hash_file, paths, chunksize=4) if pool and len(paths) > 8 else map(hash_file, paths)
    for p, res in zip(paths, results):
        if "error" in res:
            con.execute("UPDATE dup_files SET read_error=? WHERE path=?", (res["error"], p))
            path_rows[p]["read_error"] = res["error"]
        else:
            con.execute("UPDATE dup_files SET width=?, height=?, exif_date=?, has_exif=?, orientation=?, tiny=?, "
                        "ph0=?, ph1=?, ph2=?, ph3=?, dh0=?, dh1=?, dh2=?, dh3=? WHERE path=?",
                        (res["width"], res["height"], res["exif_date"], res["has_exif"], res["orientation"], res["tiny"],
                         res["ph0"], res["ph1"], res["ph2"], res["ph3"], res["dh0"], res["dh1"], res["dh2"], res["dh3"], p))
            path_rows[p].update({k: v for k, v in res.items()})
        done += 1
        if done % 20 == 0:
            con.commit()
            progress("dup_phash", done, len(need), "Перцептивные хэши")
    con.commit()
    progress("dup_phash", len(need), len(need), "Перцептивные хэши")
    if pool:
        pool.shutdown()
    # свежие данные из БД (включая ранее закэшированные)
    rows = [dict(r) for r in con.execute("SELECT * FROM dup_files WHERE substr(path,1,?)=? AND read_error IS NULL",
                                         (len(dupes.prefix(folder)), dupes.prefix(folder)))]
    path_rows = {r["path"]: r for r in rows}
    # у остальных членов группы точных копий — данные представителя (одинаковое содержимое = одинаковые хэши,
    # разрешение и EXIF), чтобы выбор оригинала не зависел от того, чей файл хэшировался
    fields = ["width", "height", "exif_date", "has_exif", "orientation", "tiny", "ph0", "ph1", "ph2", "ph3",
              "dh0", "dh1", "dh2", "dh3"]
    by_sha = defaultdict(list)
    for r in rows:
        if r["sha256"]:
            by_sha[r["sha256"]].append(r)
    exact_groups = {k: v for k, v in by_sha.items() if len(v) > 1}
    for members in exact_groups.values():
        rep = next((m for m in members if m.get("ph0") is not None), None)
        if rep:
            for m in members:
                if m.get("ph0") is None:
                    for k in fields:
                        m[k] = rep[k]
                    con.execute(f"UPDATE dup_files SET {','.join(k + '=?' for k in fields)} WHERE path=?",
                                [rep[k] for k in fields] + [m["path"]])
    con.commit()

    # --- 4. поиск совпадений по хэшам
    cand = [r for r in reps.values() if path_rows.get(r["path"], {}).get("ph0") is not None]
    cand = [path_rows[r["path"]] for r in cand]
    items = [(i, [r[f"ph{k}"] for k in range(4)], [r[f"dh{k}"] for k in range(4)]) for i, r in enumerate(cand)]
    progress("dup_match", 0, len(items), "Поиск совпадений")
    raw_pairs = find_hash_pairs(items, dupes.PH_MAX, dupes.DH_MAX,
                                on_progress=lambda d, t: progress("dup_match", d, t, "Поиск совпадений"))
    links = []
    for i, j, rot, dph, ddh in raw_pairs:
        a, b = cand[i], cand[j]
        if dupes.is_raw_jpeg_pair(a["path"], b["path"]):
            continue  # RAW + JPEG одного кадра — не дубль
        if a["exif_date"] and b["exif_date"]:  # дата съёмки ±2 с, если обе есть
            da = datetime.fromisoformat(a["exif_date"])
            db_ = datetime.fromisoformat(b["exif_date"])
            if abs((da - db_).total_seconds()) > dupes.DATE_TOLERANCE_S:
                continue
        if a.get("tiny") is None or b.get("tiny") is None:
            continue
        pcorr, pmad = pixel_check(a["tiny"], b["tiny"], rot)
        if pcorr < dupes.PIXEL_CORR_MIN or pmad > dupes.PIXEL_MAD_MAX:
            continue  # похожи, но заметно отличаются (серия, другой дубль съёмки) — не дубли
        links.append((a, b, rot, dph, ddh, pcorr))

    # --- 5. подтверждение по CLIP
    confirmed = []
    if links:
        need_paths = sorted({x["path"] for l in links for x in l[:2]})
        rotated = {l[0]["path"]: l[2] for l in links if l[2]}  # путь -> поворот (в четвертях)
        emb = {}
        try:
            for r in con.execute("SELECT path, emb FROM dup_files WHERE emb IS NOT NULL"):
                emb[r[0]] = np.frombuffer(r[1], dtype=np.float32)
            for r in con.execute("SELECT a.path, a.embedding FROM analysis a JOIN files f ON f.path = a.path "
                                 "WHERE a.embedding IS NOT NULL"):
                emb.setdefault(r[0], np.frombuffer(r[1], dtype=np.float32))
        except sqlite3.OperationalError:
            pass
        missing = [p for p in need_paths if p not in emb]
        progress("dup_clip", 0, len(missing), "Сравнение содержимого (CLIP)")
        if missing:
            new = clip_embeddings(missing, on_progress=lambda d, t: progress("dup_clip", d, t, "Сравнение содержимого (CLIP)"))
            with con:
                for p, v in new.items():
                    con.execute("UPDATE dup_files SET emb=? WHERE path=?", (v.tobytes(), p))
            emb.update(new)
        rot_emb = {}
        if rotated:
            progress("dup_clip", 0, len(rotated), "Сравнение повёрнутых копий (CLIP)")
            rot_emb = clip_embeddings(sorted(rotated), rotations=rotated)
        for a, b, rot, dph, ddh, pcorr in links:
            ea = rot_emb.get(a["path"]) if rot else emb.get(a["path"])
            eb = emb.get(b["path"])
            if ea is None or eb is None:
                continue  # без эмбеддинга подтвердить нельзя — не считаем дублем
            cos = float(ea @ eb)
            if cos >= dupes.CLIP_MIN:
                confirmed.append((a, b, rot, dph, ddh, cos, pcorr))

    # --- 6. группы, оригинал, служебные списки
    progress("dup_group", 0, 0, "Формирование групп")
    dupes.build_groups(con, folder, rows, exact_groups, confirmed, errors_total)
    dupes.mark_checked(con, folder)
    print(f"Готово: точных групп {len(exact_groups)}, визуальных связей {len(confirmed)}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
