"""Кросс-валидация и метрики.

Метрика лидерборда заранее неизвестна, поэтому считаем сразу четыре. Чтобы одна
дорогая квартира в неудачном фолде не искажала оценку, используем повторную
кросс-валидацию: 5 фолдов x 4 повтора = 20 замеров, усредняем.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import PRICE_FLOOR, TARGET
from .model import RentModel

METRICS = ["RMSE", "MAE", "R2", "MAPE"]
N_SPLITS, N_REPEATS, SEED = 5, 4, 7


def score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, PRICE_FLOOR, None)
    return np.array([
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
        np.mean(np.abs(y_true - y_pred) / y_true) * 100,
    ])


def cross_validate(train: pd.DataFrame, predict_fn: Callable) -> pd.Series:
    """predict_fn(train_fold_df, valid_fold_df) -> предсказания на валидации."""
    y = train[TARGET].values
    splitter = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    folds = [score(y[va], predict_fn(train.iloc[tr], train.iloc[va]))
             for tr, va in splitter.split(train)]
    return pd.Series(np.array(folds).mean(axis=0), index=METRICS)


def cross_validate_model(train: pd.DataFrame, degree: int = 3, target: str = "mean") -> pd.Series:
    """Кросс-валидация основной модели."""
    def predict_fn(train_fold, valid_fold):
        return RentModel(degree=degree).fit(train_fold).predict(valid_fold, target)
    return cross_validate(train, predict_fn)


def compare_degrees(train: pd.DataFrame, degrees=(1, 2, 3, 4)) -> pd.DataFrame:
    """Подобрать степень полинома и способ обратного преобразования по CV."""
    rows = {}
    for degree in degrees:
        for target in ("mean", "median"):
            rows[f"deg {degree}, {target}"] = cross_validate_model(train, degree, target)
    return pd.DataFrame(rows).T.sort_values("RMSE")


def baseline_scores(train: pd.DataFrame) -> pd.DataFrame:
    """Ориентиры, с которыми сравнивается основная модель."""
    def constant(train_fold, valid_fold):
        return np.full(len(valid_fold), train_fold[TARGET].mean())

    def district_median(train_fold, valid_fold):
        medians = train_fold.groupby("district")[TARGET].median()
        return valid_fold["district"].map(medians).values

    return pd.DataFrame({
        "Константа (среднее train)": cross_validate(train, constant),
        "Медиана по району": cross_validate(train, district_median),
    }).T
