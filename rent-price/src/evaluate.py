"""Кросс-валидация, метрики и перевод MAE в баллы конкурса.

Метрика квалификационного этапа — MAE, а балл считается по формуле
    points = max(0, 1 - MAE / 1500)
то есть каждые 15 единиц MAE — это 0.01 балла. Остальные метрики считаем
заодно, чтобы видеть картину целиком.

Чтобы одна дорогая квартира в неудачном фолде не искажала оценку, используем
повторную кросс-валидацию: 5 фолдов x 4 повтора = 20 замеров, усредняем.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import boosting
from .data import PRICE_FLOOR, TARGET
from .model import RentModel

METRICS = ["MAE", "баллы", "RMSE", "R2", "MAPE"]
N_SPLITS, N_REPEATS, SEED = 5, 4, 7
MAE_SCALE = 1500.0  # из формулы баллов квалификационного этапа


def points(mae: float) -> float:
    """Балл конкурса по значению MAE: max(0, 1 - MAE / 1500)."""
    return max(0.0, 1.0 - mae / MAE_SCALE)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, PRICE_FLOOR, None)
    mae = mean_absolute_error(y_true, y_pred)
    return np.array([
        mae,
        points(mae),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred),
        np.mean(np.abs(y_true - y_pred) / y_true) * 100,
    ])


def splits(train: pd.DataFrame) -> list:
    return list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                              random_state=SEED).split(train))


def cross_validate(train: pd.DataFrame, predict_fn: Callable) -> pd.Series:
    """predict_fn(train_fold_df, valid_fold_df) -> предсказания на валидации."""
    y = train[TARGET].values
    folds = [score(y[va], predict_fn(train.iloc[tr], train.iloc[va]))
             for tr, va in splits(train)]
    return pd.Series(np.array(folds).mean(axis=0), index=METRICS)


def make_predictor(degree: int = 3, loss: str = "l1", blend: float = 0.0,
                   target: str | None = None) -> Callable:
    """Собрать функцию предсказания: параметрическая модель плюс, при желании, бустинг."""
    def predict_fn(train_fold, valid_fold):
        base = RentModel(degree=degree, loss=loss).fit(train_fold).predict(valid_fold, target)
        if blend <= 0 or not boosting.is_available():
            return base
        return (1 - blend) * base + blend * boosting.fit_predict(train_fold, valid_fold)
    return predict_fn


def compare_models(train: pd.DataFrame) -> pd.DataFrame:
    """Основная таблица сравнения — от baseline организаторов до финальной модели."""
    rows = {}

    rows["Константа (baseline организаторов)"] = cross_validate(
        train, lambda tr, va: np.full(len(va), tr[TARGET].mean()))
    rows["Медиана по району"] = cross_validate(
        train, lambda tr, va: va["district"].map(tr.groupby("district")[TARGET].median()).values)

    for degree in (1, 2, 3, 4):
        rows[f"МНК в логах, deg {degree}"] = cross_validate(
            train, make_predictor(degree, "l2"))
    rows["МНК в логах + медиана, deg 3"] = cross_validate(
        train, make_predictor(3, "l2", target="median"))

    for degree in (1, 2, 3, 4):
        rows[f"Медианная регрессия, deg {degree}"] = cross_validate(
            train, make_predictor(degree, "l1"))

    if boosting.is_available():
        rows[f"Медианная deg 3 + бустинг {boosting.BLEND_WEIGHT:.2f}"] = cross_validate(
            train, make_predictor(3, "l1", boosting.BLEND_WEIGHT))

    return pd.DataFrame(rows).T.sort_values("MAE")


# --- совместимость со старым API ноутбука ---

def cross_validate_model(train: pd.DataFrame, degree: int = 3, target: str = "median") -> pd.Series:
    loss = "l1" if target == "median" else "l2"
    return cross_validate(train, make_predictor(degree, loss))
