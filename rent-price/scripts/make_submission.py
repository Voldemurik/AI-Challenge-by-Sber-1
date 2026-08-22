"""Собрать сабмиты одной командой.

    python scripts/make_submission.py

Печатает кросс-валидацию и кладёт два файла в submissions/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data import load_all, sanity_check
from src.evaluate import baseline_scores, compare_degrees
from src.model import RentModel

OUT_DIR = ROOT / "submissions"


def main() -> None:
    train, test, sample = load_all()

    print("=" * 72)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 72)
    sanity_check(train, test)

    print("\n" + "=" * 72)
    print("КРОСС-ВАЛИДАЦИЯ  (5 фолдов x 4 повтора)")
    print("=" * 72)
    table = pd.concat([baseline_scores(train), compare_degrees(train)])
    print(table.sort_values("RMSE").round(3).to_string())

    print("\n" + "=" * 72)
    print("ФИНАЛЬНАЯ МОДЕЛЬ")
    print("=" * 72)
    model = RentModel().fit(train)
    print(f"Параметров: {len(model.coef)}   строк в обучении: {len(train)}")
    print(f"σ остатков: {model.sigma:.4f}")
    print(f"Потолок R² при этом уровне шума: {model.noise_ceiling_r2(train):.4f}")

    OUT_DIR.mkdir(exist_ok=True)
    for filename, target in (("submission_rmse.csv", "mean"),
                             ("submission_mae.csv", "median")):
        pred = model.predict(test, target)
        out = pd.DataFrame({"id": test["id"], "rent_price": pred.round(2)})
        out.to_csv(OUT_DIR / filename, index=False)

        assert list(out.columns) == list(sample.columns), "колонки не совпадают с sample_submission"
        assert (out["id"].values == test["id"].values).all(), "порядок id не совпадает с test"
        assert out["rent_price"].notna().all(), "есть пропуски в предсказаниях"

        print(f"  {filename:22s} строк {len(out)}   среднее {pred.mean():7.1f}")

    print("\nГотово. Файлы в submissions/ — какой отправлять, зависит от метрики лидерборда:")
    print("  RMSE или R²   -> submission_rmse.csv")
    print("  MAE или MAPE  -> submission_mae.csv")


if __name__ == "__main__":
    main()
