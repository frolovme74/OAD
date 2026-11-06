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
    # Устранение пропусков в данных
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Проблема.** Если в данных есть пропуски, то большинство алгоритмов машинного обучения не будет с ними работать. Даже корреляционная матрица не будет строиться корректно.

    Существуют различные способы устранения пропусков в данных, которые связаны с удалением или заполнением пропусков.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Загрузка и первичный анализ данных

    Используем данные из соревнования [House Prices: Advanced Regression Techniques.](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.impute import SimpleImputer
    from sklearn.impute import MissingIndicator
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from IPython.display import Image
    # '%matplotlib inline' command supported automatically in marimo
    sns.set(style="ticks")
    return (
        GridSearchCV,
        Image,
        IterativeImputer,
        KNNImputer,
        Lasso,
        MissingIndicator,
        Pipeline,
        RandomForestRegressor,
        SimpleImputer,
        StandardScaler,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(pd):
    # Будем использовать только обучающую выборку
    hdata_loaded = pd.read_csv(
        "data/houseprices.csv",
        keep_default_na=False,
        na_values=["NA"]
    )
    return (hdata_loaded,)


@app.cell
def _(hdata_loaded):
    hdata_loaded.shape
    return


@app.cell
def _(hdata_loaded):
    hdata = hdata_loaded
    return (hdata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Удаление пропущенных значений

    **Допущение:** пропуски распределены случайным образом.

    **Когда рекомендуется использовать?**
    - Если пропущенных данных слишком много и возникает опасность нарушить распределение исходных данных при заполнении пропусков. Рекомендуется удалять признак (колонку) целиком.
    - Если датасет большой и пропущенных данных относительно немного, то рекомендуется удалять строки, содержащие пропуски в данных.
    - Под "немного" в идеальном случае понимается 5% от выборки.

    **Преимущества:**
    - Простота реализации.
    - При случайном распределении пропусков сохраняются параметры распределения исходных данных.

    **Недостатки:**
    - Может быть удален большой фрагмент данных при неудачном распределении пропусков в нескольких столбцах.
    - Если пропуски распределены не случайно, то можно удалить значимые данные.
    """)
    return


@app.cell
def _(hdata):
    list(zip(hdata.columns, [i for i in hdata.dtypes]))
    return


@app.cell
def _(hdata):
    # Колонки с пропусками
    hcols_with_na = [c for c in hdata.columns if hdata[c].isnull().sum() > 0]
    hcols_with_na
    return (hcols_with_na,)


@app.cell
def _(hdata):
    hdata.shape
    return


@app.cell
def _(hcols_with_na, hdata):
    # Количество пропусков
    [(c, hdata[c].isnull().sum()) for c in hcols_with_na]
    return


@app.cell
def _(hcols_with_na, hdata):
    # Доля (процент) пропусков
    [(c, hdata[c].isnull().mean()) for c in hcols_with_na]
    return


@app.cell
def _():
    # Колонки для которых удаляются пропуски
    hcols_with_na_temp = ['LotFrontage', 'GarageYrBlt', 'BsmtQual', 'MasVnrArea']
    return (hcols_with_na_temp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Удаление колонок, содержащих пустые значения
    `res = data.dropna(axis=1, how='any')`

    Удаление строк, содержащих пустые значения
    `res = data.dropna(axis=0, how='any')`

    [Документация](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html)

    **Удаление может производиться для группы строк или колонок.**
    """)
    return


@app.cell
def _(hcols_with_na_temp, hdata):
    # Удаление пропусков
    hdata_drop = hdata[hcols_with_na_temp].dropna()
    hdata_drop.shape
    return (hdata_drop,)


@app.cell
def _(plt):
    def plot_hist_diff(old_ds, new_ds, cols):
        """
        Разница между распределениями до и после устранения пропусков
        """
        for c in cols:   
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.title.set_text('Поле - ' + str(c))
            old_ds[c].hist(bins=50, ax=ax, density=True, color='green')
            new_ds[c].hist(bins=50, ax=ax, color='blue', density=True)
            plt.show()

    return (plot_hist_diff,)


@app.cell
def _(hcols_with_na_temp, hdata, hdata_drop, plot_hist_diff):
    plot_hist_diff(hdata, hdata_drop, hcols_with_na_temp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Заполнение значений для одного признака

    В этом случае данные которые находятся в соседних признаках  (колонках) не учитываются при заполнении.

    **Заполнение (внедрение) значений или импьютация (imputation)** - это заполнение пропущенных значений их статистическими оценками.

    **Для числовых признаков:**
    - Заполнение [показателями центра распределения](https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%BA%D0%B0%D0%B7%D0%B0%D1%82%D0%B5%D0%BB%D0%B8_%D1%86%D0%B5%D0%BD%D1%82%D1%80%D0%B0_%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F).
    - Заполнение константой. *Полезно в случае "неслучайного" распределения пропусков.*
    - Заполнение "хвостом распределения".

    **Для категориальных признаков:**
    - Заполнение наиболее распространенным значением категории (аналогом моды).
    - Введение отдельного значения категории для пропущенных значений.

    **Для числовых и категориальных признаков:**
    - Добавления флага пропусков.
    - Заполнение случайным значением признака. *Метод обычно применяют на больших выборках. Преимуществом является то, что он сохраняет дисперсию исходной выборки.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Заполнение [показателями центра распределения](https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%BA%D0%B0%D0%B7%D0%B0%D1%82%D0%B5%D0%BB%D0%B8_%D1%86%D0%B5%D0%BD%D1%82%D1%80%D0%B0_%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F) и константой

    **Для числовых признаков.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В случае нормального распределения математическое ожидание, медиана и мода совпадают:
    """)
    return


@app.cell
def _(Image):
    Image('img/normal_dist.png', width='50%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Для ассиметричных распределений эти показатели различаются:
    """)
    return


@app.cell
def _(Image):
    Image('img/center_skewed.png', width='30%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Когда рекомендуется использовать?**
    - Если пропуски распределены случайным образом.
    - В идеальном случае пропусков не более 5% от выборки.

    **Какой показатель центра распределения лучше использовать?**
    - Если распределение одномодальное, то лучше использовать моду, иначе математическое ожидание или медиану.
    - Не существует однозначного предпочтения между математическим ожиданием или медианой. Но медиана более устойчива к выбросам в данных.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Для внедрения значений может быть использован класс [SimpleImputer.](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

    Для фильтрации пропущенных значений может быть использован класс [MissingIndicator.](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html)
    """)
    return


@app.cell
def _(MissingIndicator, np):
    # Пример работы MissingIndicator
    temp_x1 = np.array([[np.nan, 1, 3], [4, 0, np.nan], [8, 1, 0]])
    print('Исходный массив:')
    print(temp_x1)
    _indicator = MissingIndicator()
    temp_x1_transformed = _indicator.fit_transform(temp_x1)
    print('Маска пропущенных значений:')
    print(temp_x1_transformed)
    return


@app.cell
def _(MissingIndicator, SimpleImputer):
    def impute_column(dataset, column, strategy_param, fill_value_param=None):
        """
        Заполнение пропусков в одном признаке
        """
        temp_data = dataset[[column]].values
        size = temp_data.shape[0]
        _indicator = MissingIndicator()
        mask_missing_values_only = _indicator.fit_transform(temp_data)
        imputer = SimpleImputer(strategy=strategy_param, fill_value=fill_value_param)
        all_data = imputer.fit_transform(temp_data)
        missed_data = temp_data[mask_missing_values_only]
        filled_data = all_data[mask_missing_values_only]
        return (all_data.reshape((size,)), filled_data, missed_data)

    return (impute_column,)


@app.cell
def _(hdata, impute_column):
    all_data, filled_data, missed_data = impute_column(hdata, 'MasVnrArea', 'mean')
    all_data
    return filled_data, missed_data


@app.cell
def _(filled_data):
    filled_data
    return


@app.cell
def _(missed_data):
    missed_data
    return


@app.cell
def _(impute_column, pd, sns):
    def research_impute_numeric_column(dataset, num_column, const_value=None):
        strategy_params = ['mean', 'median', 'most_frequent', 'constant']
        strategy_params_names = ['Среднее', 'Медиана', 'Мода']
        strategy_params_names.append('Константа = ' + str(const_value))

        original_temp_data = dataset[[num_column]].values
        size = original_temp_data.shape[0]
        original_data = original_temp_data.reshape((size,))

        new_df = pd.DataFrame({'Исходные данные': original_data})

        for i in range(len(strategy_params)):
            strategy = strategy_params[i]
            col_name = strategy_params_names[i]

            if (strategy != 'constant') or (
                strategy == 'constant' and const_value is not None
            ):
                if strategy == 'constant':
                    temp_data, _, _ = impute_column(
                        dataset,
                        num_column,
                        strategy,
                        fill_value_param=const_value
                    )
                else:
                    temp_data, _, _ = impute_column(
                        dataset,
                        num_column,
                        strategy
                    )

                new_df[col_name] = temp_data

        ax = sns.kdeplot(data=new_df)
        return ax

    return (research_impute_numeric_column,)


@app.cell
def _(hdata, research_impute_numeric_column):
    research_impute_numeric_column(hdata, 'GarageYrBlt', 1950)
    return


@app.cell
def _(hdata, research_impute_numeric_column):
    research_impute_numeric_column(hdata, 'LotFrontage')
    return


@app.cell
def _(hdata, research_impute_numeric_column):
    research_impute_numeric_column(hdata, 'MasVnrArea')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Заполнение "хвостом распределения"

    **Для числовых признаков.**

    **Допущение:** пропуски распределены НЕ случайным образом. Мы хотим выделить пропущенные значения из остальных значений.

    **Преимущества:**
    - Простота реализации.
    - Выделяет пропущенные значения из остальных значений.

    **Недостатки:**
    - Нарушение параметров исходного распределения.
    - Поскольку значения на краю распределения фактически являются аномалиями (выбросами), то данный подход может пересекаться с алгоритмами поиска аномалий.

    **Как вычислить "хвост распределения"?**

    Если распределение данных признака $f$ напоминает нормальное:

    $$ extreme\_value = mean(f) + 3 \cdot std(f) $$

    Для ассиметричного распределения:

    $$ IQR = Q3-Q1 $$

    $IQR -$ interquartile range.

    $$ extreme\_value = Q3 + K \cdot IQR $$

    Значение $K$ обычно выбирается равным $1,5$. Но для экстремальных выбросов выбирают $K=3$.
    """)
    return


@app.cell
def _(hdata):
    # Похоже на нормальное
    LotFrontage_ev = hdata['LotFrontage'].mean() + 3*hdata['LotFrontage'].std()
    LotFrontage_ev
    return (LotFrontage_ev,)


@app.cell
def _(LotFrontage_ev, hdata, research_impute_numeric_column):
    research_impute_numeric_column(hdata, 'LotFrontage', LotFrontage_ev)
    return


@app.cell
def _(hdata):
    # Ассиметричное
    IQR = hdata['MasVnrArea'].quantile(0.75) - hdata['MasVnrArea'].quantile(0.25)
    MasVnrArea_ev1 = hdata['MasVnrArea'].quantile(0.75) + 3*IQR
    print('IQR={}, extreme_value={}'.format(IQR, MasVnrArea_ev1))
    return IQR, MasVnrArea_ev1


@app.cell
def _(MasVnrArea_ev1, hdata, research_impute_numeric_column):
    research_impute_numeric_column(hdata, 'MasVnrArea', MasVnrArea_ev1)
    return


@app.cell
def _(IQR, hdata):
    MasVnrArea_ev2 = hdata['MasVnrArea'].quantile(0.75) + 1.5*IQR
    print('IQR={}, extreme_value={}'.format(IQR, MasVnrArea_ev2))
    return (MasVnrArea_ev2,)


@app.cell
def _(MasVnrArea_ev2, hdata, research_impute_numeric_column):
    research_impute_numeric_column(hdata, 'MasVnrArea', MasVnrArea_ev2)
    return


@app.cell
def _(hdata):
    IQR_lf = hdata['LotFrontage'].quantile(0.75) - hdata['LotFrontage'].quantile(0.25)
    LotFrontage_ev1 = hdata['LotFrontage'].quantile(0.75) + 1.5*IQR_lf
    LotFrontage_ev2 = hdata['LotFrontage'].quantile(0.75) + 3*IQR_lf
    return LotFrontage_ev1, LotFrontage_ev2


@app.cell
def _(LotFrontage_ev1, hdata, research_impute_numeric_column):
    research_impute_numeric_column(hdata, 'LotFrontage', LotFrontage_ev1)
    return


@app.cell
def _(LotFrontage_ev2, hdata, research_impute_numeric_column):
    research_impute_numeric_column(hdata, 'LotFrontage', LotFrontage_ev2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Заполнение наиболее распространенным значением категории

    **Для категориальных признаков.**

    **Допущение:** пропуски распределены случайным образом. Заполнение пропусков наиболее распространенным значением категории в наименьшей степени повлияет на исходное распределение.
    """)
    return


@app.cell
def _(hdata):
    hdata_cat_cols = ['GarageType', 'PoolQC', 'Fence']
    hdata_cat_new = hdata[hdata_cat_cols].copy()
    return hdata_cat_cols, hdata_cat_new


@app.cell
def _(hdata_cat_new, impute_column):
    GarageType_cat_new_temp, _, _ = impute_column(hdata_cat_new, 'GarageType', 'most_frequent')
    PoolQC_cat_new_temp, _, _ = impute_column(hdata_cat_new, 'PoolQC', 'most_frequent')
    Fence_cat_new_temp, _, _ = impute_column(hdata_cat_new, 'Fence', 'most_frequent')
    return Fence_cat_new_temp, GarageType_cat_new_temp, PoolQC_cat_new_temp


@app.cell
def _(
    Fence_cat_new_temp,
    GarageType_cat_new_temp,
    PoolQC_cat_new_temp,
    hdata_cat_new,
):
    hdata_cat_new['GarageType'] = GarageType_cat_new_temp
    hdata_cat_new['PoolQC'] = PoolQC_cat_new_temp
    hdata_cat_new['Fence'] = Fence_cat_new_temp
    return


@app.cell
def _(hdata, hdata_cat_cols, hdata_cat_new, plot_hist_diff):
    plot_hist_diff(hdata, hdata_cat_new, hdata_cat_cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Введение отдельного значения категории для пропущенных значений

    **Для категориальных признаков.**

    **Основное преимущество подхода** состоит в том, что не дается никаких предположений о распределении  пропущенных значений.
    """)
    return


@app.cell
def _(hdata, hdata_cat_cols):
    hdata_cat_na = hdata[hdata_cat_cols].copy()
    return (hdata_cat_na,)


@app.cell
def _(hdata_cat_na, impute_column):
    GarageType_cat_na_temp, _, _ = impute_column(hdata_cat_na, 'GarageType', 'constant', fill_value_param='NA')
    PoolQC_cat_na_temp, _, _ = impute_column(hdata_cat_na, 'PoolQC', 'constant', fill_value_param='NA')
    Fence_cat_na_temp, _, _ = impute_column(hdata_cat_na, 'Fence', 'constant', fill_value_param='NA')
    return Fence_cat_na_temp, GarageType_cat_na_temp, PoolQC_cat_na_temp


@app.cell
def _(
    Fence_cat_na_temp,
    GarageType_cat_na_temp,
    PoolQC_cat_na_temp,
    hdata_cat_na,
):
    hdata_cat_na['GarageType'] = GarageType_cat_na_temp
    hdata_cat_na['PoolQC'] = PoolQC_cat_na_temp
    hdata_cat_na['Fence'] = Fence_cat_na_temp
    return


@app.cell
def _(hdata, hdata_cat_cols, hdata_cat_na, plot_hist_diff):
    plot_hist_diff(hdata, hdata_cat_na, hdata_cat_cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Добавления флага пропусков

    **Для любых признаков.**

    Для каждой колонки данных вводится дополнительная бинарная колонка, в которой пустым значениям признака соответствует 1.

    **Преимущества:**
    - Модель получает дополнительную информацию о том, насколько мы уверены в наших данных. Особенно хорошо этот подход работает для деревьев решений и производных моделей.

    **Недостатки:**
    - Расширяется признаковое пространство.
    - Заполнять пропуски для исходных колонок все равно необходимо.
    - Флаги для разных колонок могут сильно коррелировать между собой.
    """)
    return


@app.cell
def _(hdata):
    hdata_mis = hdata[['PoolQC']].copy()
    hdata_mis.head()
    return (hdata_mis,)


@app.cell
def _(MissingIndicator, hdata_mis):
    _indicator = MissingIndicator()
    PoolQC_missing = _indicator.fit_transform(hdata_mis[['PoolQC']])
    PoolQC_missing
    return (PoolQC_missing,)


@app.cell
def _(PoolQC_missing):
    PoolQC_missing_int = [1 if i==True else 0 for i in PoolQC_missing]
    PoolQC_missing_int[:10]
    return (PoolQC_missing_int,)


@app.cell
def _(PoolQC_missing_int, hdata_mis):
    hdata_mis['PoolQC_missing'] = PoolQC_missing_int
    hdata_mis.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Заполнение значений для нескольких признаков

    В этом случае данные которые находятся в соседних признаках (колонках) учитываются при заполнении.

    Идея состоит в том, что признаки могут зависеть между собой и такие зависимости необходимо использовать при заполнении пропусков.

    В этом случае мы решаем отдельную задачу машинного обучения, рассматривая пропущенный признак как целевой (y), а остальные признаки как исходные (X).

    Для решения задачи можно использовать различные методы машинного обучения. На практике чаще всего используется **метод ближайших соседей**.

    Также проблема состоит в том, что практически все признаки могут содержать пропуски и для их заполнения другие признаки необходимо предварительно импьютировать известными методами. (возникает подобие циклических ссылок).

    Для решения этой задачи используется метод
    **MICE (multivariate Imputation of Chained Equations)**. Существует расширение этого метода **MissForest** в котором используется случайный лес.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Метод  MICE (multivariate Imputation of Chained Equations)

    Описание метода и его реализация в пакете [statsmodels.](https://www.statsmodels.org/stable/imputation.html)

    **Основные шаги метода:**
    1. Используем произвольную импьютацию для всех колонок (например, для числовых колонок - средним значением).
    1. Для первой колонки подставляем обратно пустые значения.
    1. Пропущенные значения в первой колонке предсказываются с помощью модели машинного обучения  на основе остальных значений.
    1. Предсказанные значения подставляются в первую колонку.
    1. Действия выше выполняются в цикле для всех колонок.
    1. Действия выше выполняются заданное количество раз. Предполагается что на каждой итерации мы постепенно уточняем значения, соответствующие пропускам.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Рассмотрим реализацию [MissForest](https://towardsdatascience.com/missforest-the-best-missing-data-imputation-algorithm-4d01182aed3) с использованием [IterativeImputer.](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)
    """)
    return


@app.cell
def _(hdata):
    mice_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
        'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',
        'SalePrice'
    ]

    mice_hdata = hdata[mice_cols].copy()
    return (mice_hdata,)


@app.cell
def _(mice_hdata):
    mice_hdata.head()
    return


@app.cell
def _(mice_hdata):
    mice_hdata.isnull().sum()
    return


@app.cell
def _(IterativeImputer, RandomForestRegressor):
    imputer_missForest = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        max_iter=10,
        random_state=0)
    return (imputer_missForest,)


@app.cell
def _(imputer_missForest, mice_hdata):

    import time
    start = time.time()
    missForest_hdata_imputed_temp = imputer_missForest.fit_transform(mice_hdata)
    end = time.time()
    print(f"Время выполнения: {end - start} секунд")
    return (missForest_hdata_imputed_temp,)


@app.cell
def _(mice_hdata, missForest_hdata_imputed_temp, pd):
    missForest_hdata_imputed = pd.DataFrame(
        missForest_hdata_imputed_temp,
        columns=mice_hdata.columns
    )
    return (missForest_hdata_imputed,)


@app.cell
def _(missForest_hdata_imputed):
    missForest_hdata_imputed.isnull().sum()
    return


@app.cell
def _(mice_hdata, missForest_hdata_imputed, pd, sns):
    LotFrontage_df = pd.DataFrame({
        'original': mice_hdata['LotFrontage'].values,
        'MissForest': missForest_hdata_imputed['LotFrontage'].values
    })

    sns.kdeplot(data=LotFrontage_df)
    return (LotFrontage_df,)


@app.cell
def _(mice_hdata, missForest_hdata_imputed, pd, sns):
    GarageYrBlt_df = pd.DataFrame({
        'original': mice_hdata['GarageYrBlt'].values,
        'MissForest': missForest_hdata_imputed['GarageYrBlt'].values
    })

    sns.kdeplot(data=GarageYrBlt_df)
    return (GarageYrBlt_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Импьютация с использованием метода  ближайших соседей

    Используется класс [KNNImputer.](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
    """)
    return


@app.cell
def _():
    knnimpute_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
        'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
        'WoodDeckSF',  'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',
        'SalePrice'
    ]
    return (knnimpute_cols,)


@app.cell
def _(hdata, knnimpute_cols):
    knnimpute_hdata = hdata[knnimpute_cols].copy()
    knnimpute_hdata.head()
    return (knnimpute_hdata,)


@app.cell
def _(knnimpute_hdata):
    # Признаки с пропусками
    knnimpute_hdata.isnull().sum()
    return


@app.cell
def _(KNNImputer, knnimpute_hdata, pd):
    knnimputer = KNNImputer(
        n_neighbors=5, 
        weights='distance', 
        metric='nan_euclidean', 
        add_indicator=False, 
    )
    knnimpute_hdata_imputed_temp = knnimputer.fit_transform(knnimpute_hdata)
    knnimpute_hdata_imputed = pd.DataFrame(knnimpute_hdata_imputed_temp, columns=knnimpute_hdata.columns)
    knnimpute_hdata_imputed.head()
    return (knnimpute_hdata_imputed,)


@app.cell
def _(knnimpute_hdata_imputed):
    # Пропуски заполнены
    knnimpute_hdata_imputed.isnull().sum()
    return


@app.cell
def _(LotFrontage_df, knnimpute_hdata_imputed, sns):
    LotFrontage_df['KNN_5'] = knnimpute_hdata_imputed['LotFrontage'].values

    sns.kdeplot(data=LotFrontage_df)
    return


@app.cell
def _(GarageYrBlt_df, knnimpute_hdata_imputed, sns):
    GarageYrBlt_df['KNN_5'] = knnimpute_hdata_imputed['GarageYrBlt'].values

    sns.kdeplot(data=GarageYrBlt_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Подбор гиперпараметров

    Так как для импьютации используется модель KNN, то возникает необходимость подбора гиперпараметров. В этом случае необходимо создать полный пайплайн машинного обучения и оптимизировать параметры всего пайплайна.
    """)
    return


@app.cell
def _(KNNImputer, Lasso, Pipeline, StandardScaler):
    pipe = Pipeline(steps=[
        ('imputer', KNNImputer(
            n_neighbors=5,
            weights='distance',
            add_indicator=False)),
        ('scaler', StandardScaler()),
        ('regressor', Lasso(max_iter=2000)),
    ])
    return (pipe,)


@app.cell
def _():
    param_grid = {
        'imputer__n_neighbors': [3,5,10],
        'imputer__weights': ['uniform', 'distance'],
        'imputer__add_indicator': [True, False],
        'regressor__alpha': [10, 100, 200],
    }
    return (param_grid,)


@app.cell
def _(GridSearchCV, param_grid, pipe):
    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring='r2')
    return (grid_search,)


@app.cell
def _(grid_search, hdata, knnimpute_hdata):
    grid_search.fit(knnimpute_hdata, hdata['SalePrice'])
    return


@app.cell
def _(grid_search):
    grid_search.best_params_
    return


@app.cell
def _(KNNImputer, grid_search, knnimpute_hdata, pd):
    knnimputer2 = KNNImputer(
        n_neighbors=grid_search.best_params_['imputer__n_neighbors'], 
        weights=grid_search.best_params_['imputer__weights'], 
        metric='nan_euclidean', 
        add_indicator=False, 
    )
    knnimpute_hdata_imputed_temp2 = knnimputer2.fit_transform(knnimpute_hdata)
    knnimpute_hdata_imputed2 = pd.DataFrame(knnimpute_hdata_imputed_temp2, columns=knnimpute_hdata.columns)
    knnimpute_hdata_imputed2.head()
    return (knnimpute_hdata_imputed2,)


@app.cell
def _(LotFrontage_df, knnimpute_hdata_imputed2, sns):
    LotFrontage_df['KNN_after_gridsearch'] = knnimpute_hdata_imputed2['LotFrontage']
    sns.kdeplot(data=LotFrontage_df)
    return


@app.cell
def _(GarageYrBlt_df, knnimpute_hdata_imputed2, sns):
    GarageYrBlt_df['KNN_after_gridsearch'] = knnimpute_hdata_imputed2['GarageYrBlt']
    sns.kdeplot(data=GarageYrBlt_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Методы KNNImputer и MissForest также реализованы в библиотеке [missingpy.](https://github.com/epsilon-machine/missingpy)
    """)
    return


if __name__ == "__main__":
    app.run()
