"""LightGBM с функцией потерь L1 — вспомогательная модель для смеси.

Сам по себе бустинг эту задачу решает хуже параметрической модели: зависимость
гладкая, а шума много, и деревья тратят ёмкость на аппроксимацию плавных кривых
ступеньками. Но ошибки у него другие по структуре, поэтому небольшая добавка
к основной модели ошибку всё же снижает — на кросс-валидации примерно -2 к MAE
при весе 0.15.

LightGBM не обязателен: если он не установлен, is_available() вернёт False,
и сборка сабмита обойдётся одной параметрической моделью.
"""

import numpy as np
import pandas as pd

from .data import PRICE_FLOOR, TARGET

BLEND_WEIGHT = 0.15  # доля бустинга в смеси, подобрана по кросс-валидации

PARAMS = dict(
    objective="l1",          # оптимизируем модуль ошибки — под метрику конкурса
    n_estimators=900,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    verbose=-1,
    random_state=0,
)


def is_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["area", "floor", "distance_to_center"]].copy()
    X["log_area"] = np.log(df["area"])
    X["district"] = df["district"].astype("category")
    return X


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Обучить на train, предсказать цену для test. Работает в логарифмах."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(_features(train), np.log(train[TARGET].values))
    return np.clip(np.exp(model.predict(_features(test))), PRICE_FLOOR, None)
