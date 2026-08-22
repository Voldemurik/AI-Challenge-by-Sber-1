"""Собрать сабмит для отправки на aiijc.com.

    python scripts/make_submission.py

Метрика квалификационного этапа — MAE, балл = max(0, 1 - MAE / 1500).
Основной файл — submissions/submission_mae.csv.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import boosting
from src.data import load_all, sanity_check
from src.evaluate import compare_models, cross_validate, make_predictor, points
from src.model import RentModel

OUT_DIR = ROOT / "submissions"


def main() -> None:
    train, test, sample = load_all()

    print("=" * 74)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 74)
    sanity_check(train, test)

    print("\n" + "=" * 74)
    print("КРОСС-ВАЛИДАЦИЯ  (5 фолдов x 4 повтора)")
    print("=" * 74)
    print(compare_models(train).round(3).to_string())

    print("\n" + "=" * 74)
    print("ФИНАЛЬНАЯ МОДЕЛЬ")
    print("=" * 74)
    if not boosting.is_available():
        print("LightGBM не установлен — собираю сабмит одной параметрической моделью.")

    weight = boosting.BLEND_WEIGHT if boosting.is_available() else 0.0
    cv = cross_validate(train, make_predictor(3, "l1", weight))
    print(f"Ожидаемый MAE:  {cv['MAE']:.1f}")
    print(f"Ожидаемый балл: {cv['баллы']:.4f}   (формула max(0, 1 - MAE / 1500))")

    OUT_DIR.mkdir(exist_ok=True)

    # --- основной файл: под MAE ---
    median_model = RentModel(degree=3, loss="l1").fit(train)
    pred_mae = median_model.predict(test)
    if weight:
        pred_mae = (1 - weight) * pred_mae + weight * boosting.fit_predict(train, test)

    # --- запасной файл: под RMSE и R², если метрика где-то окажется другой ---
    pred_rmse = RentModel(degree=3, loss="l2").fit(train).predict(test)

    for filename, pred, note in (("submission_mae.csv", pred_mae, "ОСНОВНОЙ — под MAE"),
                                 ("submission_rmse.csv", pred_rmse, "запасной — под RMSE/R²")):
        out = pd.DataFrame({"id": test["id"], "rent_price": np.round(pred, 2)})
        out.to_csv(OUT_DIR / filename, index=False)

        assert list(out.columns) == list(sample.columns), "колонки не совпадают с sample_submission"
        assert (out["id"].values == test["id"].values).all(), "порядок id не совпадает с test"
        assert out["rent_price"].notna().all(), "есть пропуски в предсказаниях"
        assert len(out) == len(test), "неверное число строк"

        print(f"  {filename:22s} строк {len(out)}   медиана {np.median(pred):7.1f}   {note}")

    print("\nОтправлять на aiijc.com: submissions/submission_mae.csv")


if __name__ == "__main__":
    main()
