"""Личная модель отбора: признаки, обучение (логистическая регрессия) и оценка по событиям.

Используется из train.py (обучение) и analyze.py (применение при подсчёте баллов).
"""
import math
import pickle
from collections import defaultdict

import numpy as np

import config

EMB_DIM = 768
TAB_NAMES = ["laion", "topiq_iaa", "topiq_nr", "emotion", "smile", "blink", "sharp", "n_faces", "group_size",
             "d_laion", "d_iaa", "d_nr", "d_base", "rank", "is_best"]


def build_features(rows):
    """rows — словари с полями: event_id, group_id, emb, aesthetic, topiq_iaa, topiq_nr, emotion,
    smile, blink_frac, sharp_center, n_faces, group_size, base_score. Возвращает матрицу [эмбеддинг | табличные]."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["group_id"]].append(i)
    n = len(rows)
    tab = np.zeros((n, len(TAB_NAMES)), dtype=np.float64)
    nz = lambda v, d=0.0: d if v is None else float(v)
    for idx in groups.values():
        order = sorted(idx, key=lambda i: -rows[i]["base_score"])
        best = rows[order[0]]
        for rank, i in enumerate(order):
            r = rows[i]
            tab[i] = [nz(r["aesthetic"]), nz(r["topiq_iaa"]), nz(r["topiq_nr"]), nz(r["emotion"]), nz(r["smile"]),
                      nz(r["blink_frac"]), math.log1p(nz(r["sharp_center"])), nz(r["n_faces"]), len(idx),
                      nz(r["aesthetic"]) - nz(best["aesthetic"]), nz(r["topiq_iaa"]) - nz(best["topiq_iaa"]),
                      nz(r["topiq_nr"]) - nz(best["topiq_nr"]), r["base_score"] - best["base_score"], rank,
                      1.0 if rank == 0 else 0.0]
    emb = np.stack([r["emb"] for r in rows]).astype(np.float64)
    return np.hstack([emb, tab])


def make_pipeline(n_samples):
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    k = max(2, min(config.PERSONAL_PCA, n_samples - 1, EMB_DIM))
    pre = ColumnTransformer([
        ("emb", Pipeline([("pca", PCA(n_components=k, random_state=0)), ("sc", StandardScaler())]), slice(0, EMB_DIM)),
        ("tab", StandardScaler(), slice(EMB_DIM, EMB_DIM + len(TAB_NAMES))),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=config.PERSONAL_C, max_iter=2000))])


def recommend_topk(rows, scores, fraction=None):
    """Рекомендация для оценки: в каждом событии top-k по оценке среди не-брака, k = доля × число кадров."""
    fraction = config.TARGET_SELECT_FRACTION if fraction is None else fraction
    by_event = defaultdict(list)
    for i, r in enumerate(rows):
        by_event[r["event_id"]].append(i)
    rec = np.zeros(len(rows), dtype=bool)
    for idx in by_event.values():
        k = max(1, round(fraction * len(idx)))
        cand = sorted((i for i in idx if not rows[i]["reject"]), key=lambda i: -scores[i])
        for i in cand[:k]:
            rec[i] = True
    return rec


def match_stats(rec, y):
    """угадано, всего отобрано пользователем, лишних, рекомендовано, F1."""
    tp = int((rec & (y == 1)).sum())
    total = int((y == 1).sum())
    n_rec = int(rec.sum())
    prec = tp / n_rec if n_rec else 0.0
    recall = tp / total if total else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return {"hit": tp, "total": total, "extra": n_rec - tp, "recommended": n_rec, "f1": f1}


def loeo(rows, y, X):
    """Проверка по событиям: вероятности для каждого кадра от модели, не видевшей его событие.
    Событий немного — по одному (leave-one-event-out); их сотни — блоками по PERSONAL_MAX_FOLDS, иначе обучение
    длится десятки минут."""
    events = np.array([r["event_id"] for r in rows])
    uniq = np.unique(events)
    probs = np.zeros(len(rows))
    if len(uniq) > config.PERSONAL_MAX_FOLDS:
        fold_of = dict(zip(np.random.default_rng(0).permutation(uniq), np.arange(len(uniq)) % config.PERSONAL_MAX_FOLDS))
        folds = np.array([fold_of[e] for e in events])
        masks = [folds == f for f in range(config.PERSONAL_MAX_FOLDS)]
    else:
        masks = [events == ev for ev in uniq]
    for test in masks:
        train = ~test
        if len(np.unique(y[train])) < 2:
            probs[test] = y[train].mean() if train.any() else 0.5
            continue
        model = make_pipeline(int(train.sum())).fit(X[train], y[train])
        probs[test] = model.predict_proba(X[test])[:, 1]
    return probs


def personal_weight(n):
    return n / (n + config.PERSONAL_K)


def load_model(path=None):
    path = path or config.PERSONAL_MODEL
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict(bundle, rows):
    X = build_features(rows)
    return bundle["model"].predict_proba(X)[:, 1]
