"""BestShots: обучение личной модели отбора на выборе пользователя.

Метки (таблица labels) — финальное состояние «отобрано / нет» кадров событий, которые пользователь экспортировал.
Модель включается только если при проверке по событиям угадывает лучше базовой оценки и предыдущей версии.
"""
import argparse
import json
import pickle
import shutil
import sqlite3
import sys
from datetime import datetime

import numpy as np

import config
import personal

SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    path TEXT PRIMARY KEY,
    keep INTEGER NOT NULL,
    at TEXT
);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


def save_labels(con, event_ids=None):
    """Записывает итоговое состояние (decisions, иначе рекомендация системы) всех кадров указанных событий."""
    con.executescript(SCHEMA)
    q = """SELECT s.path, COALESCE(d.keep, s.selected) FROM scores s
           LEFT JOIN decisions d ON d.path = s.path"""
    args = ()
    if event_ids is not None:
        event_ids = list(event_ids)
        q += f" WHERE s.event_id IN ({','.join('?' * len(event_ids))})"
        args = event_ids
    rows = con.execute(q, args).fetchall()
    with con:
        con.executemany("INSERT INTO labels (path, keep, at) VALUES (?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET keep=excluded.keep, at=excluded.at",
                        [(p, int(k), now()) for p, k in rows])
    return len(rows)


def load_dataset(con):
    con.row_factory = sqlite3.Row
    q = """SELECT s.path, s.event_id, s.group_id, s.group_size, s.emotion, s.base_score, s.reject, l.keep,
                  a.embedding, a.aesthetic, a.topiq_iaa, a.topiq_nr, a.smile, a.blink_frac, a.sharp_center, a.n_faces
           FROM labels l JOIN scores s ON s.path = l.path JOIN analysis a ON a.path = s.path
           WHERE s.base_score IS NOT NULL ORDER BY s.event_id, s.path"""
    rows = [dict(r) for r in con.execute(q)]
    con.row_factory = None
    for r in rows:
        r["emb"] = np.frombuffer(r["embedding"], dtype=np.float32)
    return rows


def fmt(s):
    return f"угадано {s['hit']} из {s['total']}, лишних {s['extra']}, рекомендовано {s['recommended']}"


def train(con, apply=True):
    """Возвращает словарь с отчётом. Модель включается только при улучшении."""
    con.executescript(SCHEMA)
    rows = load_dataset(con)
    events = {r["event_id"] for r in rows}
    report = {"labels": len(rows), "events": len(events), "enabled": False}
    if len(rows) < config.PERSONAL_MIN_LABELS or len(events) < 3:
        report["message"] = (f"Мало данных: {len(rows)} меток в {len(events)} событиях "
                             f"(нужно ≥ {config.PERSONAL_MIN_LABELS} и ≥ 3 события).")
        return report

    y = np.array([r["keep"] for r in rows])
    X = personal.build_features(rows)
    base_scores = np.array([r["base_score"] for r in rows])
    base = personal.match_stats(personal.recommend_topk(rows, base_scores), y)
    probs = personal.loeo(rows, y, X)
    new = personal.match_stats(personal.recommend_topk(rows, probs), y)
    prev = personal.load_model()
    prev_f1 = prev["cv_f1"] if prev else None
    report.update(base=base, personal=new, prev_f1=prev_f1, total_frames=len(rows),
                  weight=personal.personal_weight(len(rows)))
    better = new["f1"] > base["f1"] and (prev_f1 is None or new["f1"] > prev_f1)
    report["enabled"] = bool(better)
    if better and apply:
        model = personal.make_pipeline(len(rows)).fit(X, y)
        bundle = {"model": model, "n": len(rows), "cv_f1": new["f1"], "base_f1": base["f1"],
                  "trained_at": now(), "features": personal.TAB_NAMES}
        if config.PERSONAL_MODEL.exists():
            shutil.copy2(config.PERSONAL_MODEL, config.PERSONAL_PREV)
        config.MODELS_DIR.mkdir(exist_ok=True)
        with open(config.PERSONAL_MODEL, "wb") as f:
            pickle.dump(bundle, f)
        (config.MODELS_DIR / "personal.json").write_text(  # сведения о модели для окна «О программе» (без загрузки sklearn)
            json.dumps({"n": len(rows), "cv_f1": new["f1"], "base_f1": base["f1"], "trained_at": bundle["trained_at"]}),
            encoding="utf-8")
        import analyze  # пересчёт рекомендаций только для событий без решений пользователя
        rows_after = analyze.score_all(con, freeze_decided=True)
        sel = sum(r["selected"] for r in rows_after)
        report["recommended_pct"] = 100 * sel / len(rows_after)
    return report


def summary_text(rep):
    if "message" in rep:
        return rep["message"]
    b, p = rep["base"], rep["personal"]
    lines = [f"Меток: {rep['labels']} (событий: {rep['events']}), проверка по событиям",
             f"Базовая модель: {fmt(b)} ({100 * b['recommended'] / rep['total_frames']:.0f}% кадров)",
             f"Личная модель:  {fmt(p)} ({100 * p['recommended'] / rep['total_frames']:.0f}% кадров)"]
    if rep["enabled"]:
        lines.append(f"Личная модель включена (вес в баллах {rep['weight']:.2f}); "
                     f"рекомендовано после пересчёта: {rep.get('recommended_pct', 0):.0f}% кадров.")
    else:
        why = "не лучше базовой" if p["f1"] <= b["f1"] else "не лучше предыдущей версии"
        lines.append(f"Личная модель не включена ({why}).")
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="BestShots: обучение на выборе пользователя")
    ap.add_argument("--db", default="bestshots.db")
    ap.add_argument("--seed-labels", action="store_true",
                    help="заполнить labels текущим состоянием (decisions + scores) всех кадров")
    ap.add_argument("--json", action="store_true", help="последней строкой вывести отчёт в JSON")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    if args.seed_labels:
        print(f"Меток записано: {save_labels(con)}")
    rep = train(con)
    try:  # последний отчёт — для окна «О программе»
        (config.MODELS_DIR / "last_training.json").write_text(json.dumps({**rep, "at": now()}, default=str), encoding="utf-8")
    except OSError:
        pass
    print(summary_text(rep))
    if args.json:
        print("JSON:" + json.dumps(rep, ensure_ascii=False))
    con.close()


if __name__ == "__main__":
    main()
