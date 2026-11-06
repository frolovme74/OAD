import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Обработка выбросов

    Существует [ГОСТ Р ИСО 16269-4-2017 Статистические методы. Статистическое представление данных. Часть 4. Выявление и обработка выбросов](http://docs.cntd.ru/document/1200146680) посвященный обработке выбросов.

    В соответствии с ГОСТ, **выброс (outlier)** - это элемент маломощного подмножества выборки, существенно отличающийся от остальных элементов выборки.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Почему необходимо устранять выбросы?** Выбросы негативно влияют на построение модели.

    **Причины появления выбросов:**
    - Ошибки в измерениях, например, часть значений признака "расстояние" была измерена не в километрах, а в метрах.
    - Технические ошибки форматирования данных.

    **Основные задачи обработки выбросов:**
    - Обнаружение выбросов.
    - Устранение (удаление или замена) выбросов (в зависимости от требований задачи).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Загрузка и предобработка данных
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from IPython.display import Image
    # '%matplotlib inline' command supported automatically in marimo
    sns.set(style="ticks")
    return Image, np, pd, plt, sns, stats


@app.cell
def _():
    x_col_list = ['RM', 'LSTAT', 'CRIM']
    return (x_col_list,)


@app.cell
def _(pd, x_col_list):
    boston_df = pd.read_csv("data/BostonHousing.csv")

    boston_df.columns = boston_df.columns.str.upper()

    data = boston_df[x_col_list].copy()

    data["Y"] = boston_df["MEDV"]

    data.shape
    return (data,)


@app.cell
def _(plt, sns, stats):
    def diagnostic_plots(df, variable, title):
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.subplot(2, 2, 1)  # гистограмма
        df[variable].hist(bins=30)
        plt.subplot(2, 2, 2)
        stats.probplot(df[variable], dist='norm', plot=plt)  ## Q-Q plot
        plt.subplot(2, 2, 3)
        sns.violinplot(x=df[variable])
        plt.subplot(2, 2, 4)  # ящик с усами
        sns.boxplot(x=df[variable])
        fig.suptitle(title)
        plt.show()  # ящик с усами

    return (diagnostic_plots,)


@app.cell
def _(data, diagnostic_plots):
    diagnostic_plots(data, 'RM', 'RM - original')
    return


@app.cell
def _(data, diagnostic_plots):
    diagnostic_plots(data, 'LSTAT', 'LSTAT - original')
    return


@app.cell
def _(data, diagnostic_plots):
    diagnostic_plots(data, 'CRIM', 'CRIM - original')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Обнаружение выбросов
    """)
    return


@app.cell
def _(Image):
    # Нормальное распределение
    Image('img/normal_dist.png', width='50%')
    return


@app.cell
def _(Image):
    # Асимметричное (положительно скошенное) распределение
    Image('img/center_skewed.png', width='30%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    По определению гистограммы распределения, выбросы - это значения на краях гистограммы (очень большие или очень маленькие по сравнению со всей выборкой).

    Задача обнаружения выбросов - это задача выделения элементов, находящихся на краях гистограммы.

    ### Подход в случае нормального распределения

    Или распределения, похожего на нормальное.

    #### Использование правила трех сигм
    """)
    return


@app.cell
def _(Image):
    Image('img/out_sigma.png', width='50%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$ outlier < mean(x) - 3 \cdot std(x) $$
    $$ outlier > mean(x) + 3 \cdot std(x) $$

    где $ \sigma = std(x) $

    #### Использование 5% и 95% квантилей

    95% данных располагаются выше 5% квантиля и 95% данных располагаются ниже 95% квантиля. Значения ниже 5% квантиля и выше 95% квантиля можно считать выбросами.
    """)
    return


@app.cell
def _(Image):
    Image('img/out_quantile.png', width='50%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$ outlier < x.quantile(5\%) $$
    $$ outlier > x.quantile(95\%) $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Подход в случае асимметричного распределения
    """)
    return


@app.cell
def _(Image):
    Image('img/out_skewed.png', width='30%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Использование межквартильного размаха

    Межквартильный размах IQR (interquartile range, IQR) - это разность третьего квартиля и первого квартиля:

    $$ IQR = Q3(x) - Q1(x) $$

    Тогда:

    $$  outlier < Q1(x) - K \cdot IQR $$
    $$  outlier > Q3(x) + K \cdot IQR $$

    Значение $K$ обычно выбирается равным $1,5$.
    """)
    return


@app.cell
def _():
    # Тип вычисления верхней и нижней границы выбросов
    from enum import Enum
    class OutlierBoundaryType(Enum):
        SIGMA = 1
        QUANTILE = 2
        IRQ = 3

    return (OutlierBoundaryType,)


@app.cell
def _(OutlierBoundaryType):
    # Функция вычисления верхней и нижней границы выбросов
    def get_outlier_boundaries(df, col, outlier_boundary_type: OutlierBoundaryType):
        if outlier_boundary_type == OutlierBoundaryType.SIGMA:
            K1 = 3
            _lower_boundary = df[col].mean() - K1 * df[col].std()
            _upper_boundary = df[col].mean() + K1 * df[col].std()
        elif outlier_boundary_type == OutlierBoundaryType.QUANTILE:
            _lower_boundary = df[col].quantile(0.05)
            _upper_boundary = df[col].quantile(0.95)
        elif outlier_boundary_type == OutlierBoundaryType.IRQ:
            K2 = 1.5
            IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
            _lower_boundary = df[col].quantile(0.25) - K2 * IQR
            _upper_boundary = df[col].quantile(0.75) + K2 * IQR
        else:
            raise NameError('Unknown Outlier Boundary Type')
        return (_lower_boundary, _upper_boundary)

    return (get_outlier_boundaries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Устранение выбросов

    - Устранение предполагает удаление или замену выбросов.
    - Для замены могут использоваться найденные верхняя и нижняя границы.
    - Найденные выбросы можно считать "псевдо-пропущенными значениями" и использовать для их замены методы обработки пропусков.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Удаление выбросов
    """)
    return


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _(
    OutlierBoundaryType,
    data,
    diagnostic_plots,
    get_outlier_boundaries,
    np,
    x_col_list,
):
    for _col in x_col_list:
        for _obt in OutlierBoundaryType:
            _lower_boundary, _upper_boundary = get_outlier_boundaries(data, _col, _obt)  # Вычисление верхней и нижней границы
            outliers_temp = np.where(data[_col] > _upper_boundary, True, np.where(data[_col] < _lower_boundary, True, False))
            data_trimmed = data.loc[~outliers_temp,]  # Флаги для удаления выбросов
            _title = 'Поле-{}, метод-{}, строк-{}'.format(_col, _obt, data_trimmed.shape[0])
            diagnostic_plots(data_trimmed, _col, _title)  # Удаление данных на основе флага
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Замена выбросов

    Для замены используем найденные верхнюю и нижнюю границы.
    """)
    return


@app.cell
def _(
    OutlierBoundaryType,
    data,
    diagnostic_plots,
    get_outlier_boundaries,
    np,
    x_col_list,
):
    for _col in x_col_list:
        for _obt in OutlierBoundaryType:
            _lower_boundary, _upper_boundary = get_outlier_boundaries(data, _col, _obt)  # Вычисление верхней и нижней границы
            data[_col] = np.where(data[_col] > _upper_boundary, _upper_boundary, np.where(data[_col] < _lower_boundary, _lower_boundary, data[_col]))
            _title = 'Поле-{}, метод-{}'.format(_col, _obt)  # Изменение данных
            diagnostic_plots(data, _col, _title)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Обработка выбросов должна производиться **ТОЛЬКО В ОБУЧАЮЩЕЙ ВЫБОРКЕ**, чтобы улучшить точность построения модели.
    """)
    return


if __name__ == "__main__":
    app.run()
