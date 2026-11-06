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
    # Масштабирование признаков
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Масштабирование — это изменение диапазона измерения признака с целью улучшения качества построения модели.

    **Почему необходимо масштабировать признаки?**

    Алгоритмы, основанные на градиентной оптимизации, чувствительны к масштабу признаков, поскольку величина градиента зависит от масштаба входных данных. Кроме того, метрические алгоритмы также чувствительны к масштабу признаков: признаки с большей дисперсией могут доминировать при вычислении расстояний и определять значение метрики, снижая вклад признаков с меньшим диапазоном значений.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Методы машинного обучения (как с учителем, так и без учителя), **ЗАВИСЯЩИЕ** от масштабирования признаков:

    - Метод ближайших соседей
    - Линейная регрессия
    - Логистическая регрессия
    - Метод опорных векторов (SVM)
    - Нейронные сети
    - Некоторые алгоритмы кластеризации (K-means)
    - Анализ главных компонент (Principal Component Analysis, PCA)

    Методы машинного обучения, **НЕ ЗАВИСЯЩИЕ** от масштабирования признаков:

    - Деревья решений и другие алгоритмы на их основе:
        - Случайный лес
        - Градиентный бустинг
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Признаки нужно масштабировать до или после деления на обучающую и тестовую выборку?**

    - Предположим, что мы разделили данные на обучающую и тестовую выборки, и взяли данные для масштабирования только из обучающей выборки.
    - В этом случае наличие выбросов в тестовой выборке может нарушить схему масштабирования.
    - Традиционным является подход, при котором данные делятся на обучающую и тестовую выборки ДО масштабирования. Параметры масштабирования (например, среднее значение, дисперсия) берутся только из обучающей выборки и затем применяются к тестовой выборке.
    - Если выбросы в тестовой выборке мешают реализации этого подхода, то данные делятся на обучающую и тестовую выборки ПОСЛЕ масштабирования.
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
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MaxAbsScaler

    return (
        MaxAbsScaler,
        MinMaxScaler,
        RobustScaler,
        StandardScaler,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    boston_dataset = pd.read_csv("data/BostonHousing.csv")
    boston_dataset.columns = boston_dataset.columns.str.upper()
    data = boston_dataset.drop(columns="MEDV").copy()
    data["Y"] = boston_dataset["MEDV"]
    data.shape
    return (data,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    # Нужно ли масштабирование
    data.describe()
    return


@app.cell
def _(data):
    # DataFrame не содержащий целевой признак
    X_ALL = data.drop('Y', axis=1)
    return (X_ALL,)


@app.cell
def _(X_ALL, pd):
    # Функция для восстановления датафрейма
    # на основе масштабированных данных
    def arr_to_df(arr_scaled):
        res = pd.DataFrame(arr_scaled, columns=X_ALL.columns)
        return res

    return (arr_to_df,)


@app.cell
def _(X_ALL, arr_to_df, data, train_test_split):
    # Разделим выборку на обучающую и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X_ALL, data['Y'],
                                                        test_size=0.2,
                                                        random_state=1)
    # Преобразуем массивы в DataFrame
    X_train_df = arr_to_df(X_train)
    X_test_df = arr_to_df(X_test)

    X_train_df.shape, X_test_df.shape
    return X_test, X_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Масштабирование данных на основе [Z-оценки](https://ru.wikipedia.org/wiki/Z-%D0%BE%D1%86%D0%B5%D0%BD%D0%BA%D0%B0)

    $$ x' = \frac{x - \mu(x) }{\sigma(x)} $$

    где $x$ - признак, $\mu(x)=mean(x)$ - среднее значение, $\sigma(x)=std(x)$ - среднеквадратичное отклонение.

    Особенности метода:
    - Среднее значение приводится к 0.
    - Среднеквадратичное отклонение приводится к 1.
    - Форма исходного распределения сохраняется.
    - Максимальные и минимальные значения могут варьироваться.
    - Выбросы сохраняются.

    Метод реализован с использованием класса [StandardScaler.](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    """)
    return


@app.cell
def _(StandardScaler, X_ALL, arr_to_df):
    # Обучаем StandardScaler на всей выборке и масштабируем
    cs11 = StandardScaler()
    data_cs11_scaled_temp = cs11.fit_transform(X_ALL)
    # формируем DataFrame на основе массива
    data_cs11_scaled =  arr_to_df(data_cs11_scaled_temp)
    data_cs11_scaled
    return (data_cs11_scaled,)


@app.cell
def _(data_cs11_scaled):
    data_cs11_scaled.describe()
    return


@app.cell
def _(plt, sns):
    # Построение плотности распределения
    def draw_kde(col_list, df1, df2, label1, label2):
        fig, (ax1, ax2) = plt.subplots(
            ncols=2, figsize=(12, 5))
        # первый график
        ax1.set_title(label1)
        sns.kdeplot(data=df1[col_list], ax=ax1)
        # второй график
        ax2.set_title(label2)
        sns.kdeplot(data=df2[col_list], ax=ax2)
        plt.show()

    return (draw_kde,)


@app.cell
def _(data, data_cs11_scaled, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data, data_cs11_scaled, 'до масштабирования', 'после масштабирования')
    return


@app.cell
def _(StandardScaler, X_test, X_train, arr_to_df):
    # Обучаем StandardScaler на обучающей выборке 
    # и масштабируем обучающую и тестовую выборки
    cs12 = StandardScaler()
    cs12.fit(X_train)
    data_cs12_scaled_train_temp = cs12.transform(X_train)
    data_cs12_scaled_test_temp = cs12.transform(X_test)
    # формируем DataFrame на основе массива
    data_cs12_scaled_train = arr_to_df(data_cs12_scaled_train_temp)
    data_cs12_scaled_test = arr_to_df(data_cs12_scaled_test_temp)
    return data_cs12_scaled_test, data_cs12_scaled_train


@app.cell
def _(data_cs12_scaled_train):
    data_cs12_scaled_train.describe()
    return


@app.cell
def _(data_cs12_scaled_test):
    data_cs12_scaled_test.describe()
    return


@app.cell
def _(data_cs12_scaled_test, data_cs12_scaled_train, draw_kde):
    # распределения для обучающей и тестовой выборки немного отличаются
    draw_kde(['AGE', 'DIS', 'NOX'], data_cs12_scaled_train, data_cs12_scaled_test, 'обучающая', 'тестовая')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Масштабирование "Mean Normalisation"

    $$ x' = \frac{x - \mu(x) }{max(x)-min(x)} $$

    где $x$ - признак, $\mu(x)=mean(x)$ - среднее значение.

    Особенности метода:
    - Среднее значение приводится к 0.
    - Среднеквадратичное отклонение может варьироваться.
    - Форма исходного распределения может изменяться.
    - Максимальные и минимальные значения в диапазоне [-1;1].
    - Выбросы сохраняются.

    Нет стандартного класса, возможна реализация с использованием Pandas.
    """)
    return


@app.cell
def _(X_train):
    class MeanNormalisation:

        def fit(self, param_df):
            self.means = X_train.mean(axis=0)
            maxs = X_train.max(axis=0)
            mins = X_train.min(axis=0)
            self.ranges = maxs - mins

        def transform(self, param_df):
            param_df_scaled = (param_df - self.means) / self.ranges
            return param_df_scaled

        def fit_transform(self, param_df):
            self.fit(param_df)
            return self.transform(param_df)

    return (MeanNormalisation,)


@app.cell
def _(MeanNormalisation, X_ALL):
    sc21 = MeanNormalisation()
    data_cs21_scaled = sc21.fit_transform(X_ALL)
    data_cs21_scaled.describe()
    return (data_cs21_scaled,)


@app.cell
def _(MeanNormalisation, X_test, X_train):
    cs22 = MeanNormalisation()
    cs22.fit(X_train)
    data_cs22_scaled_train = cs22.transform(X_train)
    data_cs22_scaled_test = cs22.transform(X_test)
    return data_cs22_scaled_test, data_cs22_scaled_train


@app.cell
def _(data_cs22_scaled_train):
    data_cs22_scaled_train.describe()
    return


@app.cell
def _(data_cs22_scaled_test):
    data_cs22_scaled_test.describe()
    return


@app.cell
def _(data, data_cs21_scaled, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data, data_cs21_scaled, 'до масштабирования', 'после масштабирования')
    return


@app.cell
def _(data_cs22_scaled_test, data_cs22_scaled_train, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data_cs22_scaled_train, data_cs22_scaled_test, 'обучающая', 'тестовая')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MinMax-масштабирование

    $$ x' = \frac{x - min(x)}{max(x)-min(x)} $$

    Особенности метода:
    - Среднее значение может варьироваться.
    - Среднеквадратичное отклонение может варьироваться.
    - Форма исходного распределения может изменяться.
    - Максимальные и минимальные значения в диапазоне [0;1].
    - Выбросы сохраняются.

    Метод реализован с использованием класса [MinMaxScaler.](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    """)
    return


@app.cell
def _(MinMaxScaler, X_ALL, arr_to_df):
    # Обучаем StandardScaler на всей выборке и масштабируем
    cs31 = MinMaxScaler()
    data_cs31_scaled_temp = cs31.fit_transform(X_ALL)
    # формируем DataFrame на основе массива
    data_cs31_scaled = arr_to_df(data_cs31_scaled_temp)
    data_cs31_scaled.describe()
    return (data_cs31_scaled,)


@app.cell
def _(MinMaxScaler, X_test, X_train, arr_to_df):
    cs32 = MinMaxScaler()
    cs32.fit(X_train)
    data_cs32_scaled_train_temp = cs32.transform(X_train)
    data_cs32_scaled_test_temp = cs32.transform(X_test)
    # формируем DataFrame на основе массива
    data_cs32_scaled_train = arr_to_df(data_cs32_scaled_train_temp)
    data_cs32_scaled_test = arr_to_df(data_cs32_scaled_test_temp)
    return data_cs32_scaled_test, data_cs32_scaled_train


@app.cell
def _(data, data_cs31_scaled, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data, data_cs31_scaled, 'до масштабирования', 'после масштабирования')
    return


@app.cell
def _(data_cs32_scaled_test, data_cs32_scaled_train, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data_cs32_scaled_train, data_cs32_scaled_test, 'обучающая', 'тестовая')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Масштабирование по медиане

    $$ x' = \frac{x - median(x)}{IQR} $$

    где

    $$ IQR = Q3(x) - Q1(x) $$

    $IQR -$ разность между 1 и 3 квартилями.

    Особенности метода:
    - Медиана приводится к 0.
    - Среднеквадратичное отклонение может варьироваться.
    - Форма исходного распределения может изменяться.
    - Максимальные и минимальные значения могут варьироваться.
    - Устраняются выбросы.

    Метод реализован с использованием класса [RobustScaler.](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
    """)
    return


@app.cell
def _(RobustScaler, X_ALL, arr_to_df):
    cs41 = RobustScaler()
    data_cs41_scaled_temp = cs41.fit_transform(X_ALL)
    # формируем DataFrame на основе массива
    data_cs41_scaled = arr_to_df(data_cs41_scaled_temp)
    data_cs41_scaled.describe()
    return (data_cs41_scaled,)


@app.cell
def _(RobustScaler, X_test, X_train, arr_to_df):
    cs42 = RobustScaler()
    cs42.fit(X_train)
    data_cs42_scaled_train_temp = cs42.transform(X_train)
    data_cs42_scaled_test_temp = cs42.transform(X_test)
    # формируем DataFrame на основе массива
    data_cs42_scaled_train = arr_to_df(data_cs42_scaled_train_temp)
    data_cs42_scaled_test = arr_to_df(data_cs42_scaled_test_temp)
    return data_cs42_scaled_test, data_cs42_scaled_train


@app.cell
def _(data, data_cs41_scaled, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data, data_cs41_scaled, 'до масштабирования', 'после масштабирования')
    return


@app.cell
def _(data_cs42_scaled_test, data_cs42_scaled_train, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data_cs42_scaled_train, data_cs42_scaled_test, 'обучающая', 'тестовая')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Масштабирование по максимальному значению

    $$ x' = \frac{x}{max(|x|)} $$

    Особенности метода:
    - Среднее значение НЕ ЦЕНТРИРУЕТСЯ. Предполагается что оно уже отцентрировано исходя их характера данных.
    - Среднеквадратичное отклонение не масштабируется.
    - Форма исходного распределения может изменяться.
    - Максимальные и минимальные значения в диапазоне [-1;1].

    Метод реализован с использованием класса [MaxAbsScaler.](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)

    Для центрирования распределения возможна комбинация с другими методами, например со StandardScaler.
    """)
    return


@app.cell
def _(MaxAbsScaler, X_ALL, arr_to_df):
    cs51 = MaxAbsScaler()
    data_cs51_scaled_temp = cs51.fit_transform(X_ALL)
    # формируем DataFrame на основе массива
    data_cs51_scaled = arr_to_df(data_cs51_scaled_temp)
    data_cs51_scaled.describe()
    return (data_cs51_scaled,)


@app.cell
def _(MaxAbsScaler, StandardScaler, X_test, X_train, arr_to_df):
    cs52_mas = MaxAbsScaler()
    cs52_mean = StandardScaler(with_mean=True, with_std=False)

    cs52_mas.fit(X_train)
    cs52_mean.fit(X_train)

    data_cs52_scaled_train_temp = cs52_mas.transform(cs52_mean.transform(X_train))
    data_cs52_scaled_test_temp = cs52_mas.transform(cs52_mean.transform(X_test))
    # формируем DataFrame на основе массива
    data_cs52_scaled_train = arr_to_df(data_cs52_scaled_train_temp)
    data_cs52_scaled_test = arr_to_df(data_cs52_scaled_test_temp)
    return data_cs52_scaled_test, data_cs52_scaled_train


@app.cell
def _(data, data_cs51_scaled, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data, data_cs51_scaled, 'до масштабирования', 'после масштабирования')
    return


@app.cell
def _(data_cs52_scaled_test, data_cs52_scaled_train, draw_kde):
    draw_kde(['AGE', 'DIS', 'NOX'], data_cs52_scaled_train, data_cs52_scaled_test, 'обучающая', 'тестовая')
    return


if __name__ == "__main__":
    app.run()
