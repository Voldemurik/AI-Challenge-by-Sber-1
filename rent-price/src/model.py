"""Блочная лог-линейная модель — основное решение.

Идея. Цена собирается как произведение факторов: район задаёт базовый уровень,
площадь входит степенью, расстояние — экспонентой. После логарифмирования всё
становится линейным:

    log(price) = a_d + P_d(log area) - b_d * distance + c_d * log(1 + floor) + шум

Индекс d означает, что у каждого района свой набор коэффициентов, а P_d — полином
степени `degree`. Это и есть то самое взаимодействие district x area, без которого
задача не решается.

Способ подгонки выбирается под метрику соревнования:

    loss="l2"  метод наименьших квадратов -> оценивает СРЕДНЕЕ log(price).
               Обратное преобразование exp(mu + sigma^2/2) даёт условное среднее
               цены — оптимум для RMSE и R².

    loss="l1"  медианная (квантильная) регрессия -> оценивает МЕДИАНУ log(price).
               Поскольку медиана сохраняется при монотонном преобразовании,
               exp(mu) сразу даёт условную медиану цены — оптимум для MAE и MAPE.

Разница не косметическая: остатки в логарифмах скошены влево (из-за обрезки цены
снизу на 350), поэтому среднее и медиана log(price) заметно расходятся. На
кросс-валидации переход с l2 на l1 даёт примерно -5 к MAE.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import PRICE_FLOOR, TARGET

LOG_AREA_CENTER = 4.2  # ≈ log(66 м²): центрируем, чтобы свободный член был осмысленным
DEFAULT_DEGREE = 3     # подобрано по кросс-валидации, см. evaluate.py


def make_basis(df: pd.DataFrame, districts: list[str], degree: int = DEFAULT_DEGREE) -> np.ndarray:
    """Блочная матрица признаков: каждый признак умножается на one-hot района.

    В результате обычный МНК оценивает отдельные коэффициенты для каждого района —
    ровно то, что нужно, раз связь цены с площадью в районах разная.
    """
    onehot = (pd.get_dummies(df["district"])
              .reindex(columns=districts, fill_value=0)
              .astype(float).values)

    log_area = np.log(df["area"].values) - LOG_AREA_CENTER
    distance = df["distance_to_center"].values
    floor = np.log1p(df["floor"].values)

    blocks = [onehot]
    for power in range(1, degree + 1):
        blocks.append(onehot * (log_area ** power)[:, None])
    blocks.append(onehot * distance[:, None])
    blocks.append(onehot * floor[:, None])
    return np.hstack(blocks)


def fit_l1(design: np.ndarray, y: np.ndarray, max_iter: int = 60, eps: float = 1e-3) -> np.ndarray:
    """Медианная регрессия методом итеративно перевзвешенных наименьших квадратов.

    Минимизируется сумма модулей остатков, а не квадратов. Приём простой: на каждом
    шаге решаем обычный МНК с весами 1/|остаток| — точки, от которых модель далеко,
    получают меньший вес, и решение сходится к медианной регрессии.
    """
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(max_iter):
        weights = 1.0 / np.maximum(np.abs(y - design @ beta), eps)
        root_w = np.sqrt(weights)
        beta_new = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            return beta_new
        beta = beta_new
    return beta


@dataclass
class RentModel:
    """Обучение — одна задача наименьших квадратов, предсказание — умножение матриц."""

    degree: int = DEFAULT_DEGREE
    loss: str = "l1"                      # "l1" под MAE (метрика конкурса), "l2" под RMSE
    districts: list[str] = field(default_factory=list)
    coef: np.ndarray | None = None
    sigma: float = 0.0

    def fit(self, train: pd.DataFrame) -> "RentModel":
        if self.loss not in ("l1", "l2"):
            raise ValueError("loss должен быть 'l1' или 'l2'")

        self.districts = sorted(train["district"].unique())
        design = make_basis(train, self.districts, self.degree)
        log_price = np.log(train[TARGET].values)

        if self.loss == "l1":
            self.coef = fit_l1(design, log_price)
        else:
            self.coef, *_ = np.linalg.lstsq(design, log_price, rcond=None)

        self.sigma = float((log_price - design @ self.coef).std())
        return self

    def predict(self, df: pd.DataFrame, target: str | None = None) -> np.ndarray:
        """target='median' -> exp(mu),             оптимум для MAE и MAPE
        target='mean'   -> exp(mu + sigma^2/2), оптимум для RMSE и R²

        По умолчанию согласуется с loss: l1 -> медиана, l2 -> среднее.
        """
        if self.coef is None:
            raise RuntimeError("Модель не обучена — сначала вызовите fit().")
        if target is None:
            target = "median" if self.loss == "l1" else "mean"
        if target not in ("mean", "median"):
            raise ValueError("target должен быть 'mean' или 'median'")

        mu = make_basis(df, self.districts, self.degree) @ self.coef
        if target == "mean":
            mu = mu + 0.5 * self.sigma ** 2
        return np.clip(np.exp(mu), PRICE_FLOOR, None)

    def coefficients_table(self) -> pd.DataFrame:
        """Читаемая таблица коэффициентов (для degree=1 — прямая интерпретация)."""
        k = len(self.districts)
        table = pd.DataFrame(index=self.districts)
        table["базовая ставка (66 м²)"] = np.exp(self.coef[:k]).round(0)
        table["α — эластичность по площади"] = self.coef[k:2 * k].round(2)
        table["β — % цены за км"] = ((np.exp(self.coef[-2 * k:-k]) - 1) * 100).round(1)
        table["γ — этаж"] = self.coef[-k:].round(3)
        return table.sort_values("базовая ставка (66 м²)")

    def noise_ceiling_r2(self, train: pd.DataFrame) -> float:
        """Максимальный R², достижимый при данном уровне шума.

        Если log(price) = mu + N(0, sigma), то цена логнормальна и её условная
        дисперсия равна exp(2*mu + sigma^2) * (exp(sigma^2) - 1). Идеальная модель
        оставила бы после себя ровно эту дисперсию.
        """
        design = make_basis(train, self.districts, self.degree)
        log_price = np.log(train[TARGET].values)
        mu = design @ self.coef

        residual = log_price - mu
        sigma_d = train["district"].map(
            pd.Series(residual).groupby(train["district"].values).std()
        ).values

        cond_var = np.exp(2 * mu + sigma_d ** 2) * (np.exp(sigma_d ** 2) - 1)
        return float(1 - cond_var.mean() / np.var(train[TARGET].values))
