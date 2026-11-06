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
    # Кодирование категориальных признаков
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import Image
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    sns.set(style="ticks")
    return Image, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Загрузка и предобработка данных

    Используем данные из соревнования [Titanic](https://www.kaggle.com/c/titanic/)
    """)
    return


@app.cell
def _(pd):
    # Будем использовать только обучающую выборку
    data_loaded = pd.read_csv('data/titanic.csv', sep=",")
    return (data_loaded,)


@app.cell
def _(data_loaded):
    # размер набора данных
    data_loaded.shape
    return


@app.cell
def _(data_loaded):
    data_loaded.head()
    return


@app.cell
def _(data_loaded):
    data_features = list(zip(
    # признаки
    [i for i in data_loaded.columns],
    zip(
        # типы колонок
        [str(i) for i in data_loaded.dtypes],
        # проверим есть ли пропущенные значения
        [i for i in data_loaded.isnull().sum()]
    )))
    # Признаки с типом данных и количеством пропусков
    data_features
    return


@app.cell
def _(data_loaded):
    # Используем только некоторые признаки
    cols_filter = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'Sex', 'Cabin', 'Embarked', 'Survived']
    data = data_loaded[cols_filter]
    data.head()
    return (data,)


@app.cell
def _(data):
    # Заполним пропуски
    data.dropna(subset=['Fare', 'Embarked'], inplace=True)
    return


@app.cell
def _(data):
    # От каюты оставляет только первую букву 
    # и убираем каюты типа Т так как их мало
    data["Cabin"] = data["Cabin"].str[0].fillna("n")
    data_1 = data[data["Cabin"] != "T"]
    return (data_1,)


@app.cell
def _(data_1):
    # Заполним пропуски возраста средними значениями
    def impute_na(df, variable, value):
        df[variable] = df[variable].fillna(value)

    impute_na(data_1, "Age", data_1["Age"].mean())
    return


@app.cell
def _(data_1):
    # Убедимся что нет пустых значений
    data_1.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Кодирование категорий целочисленными значениями - [label encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Label Encoding (LE), который также называют integer encoding, предполагает, что значения категорий заменяются целыми числами в случайном порядке.

    #### Преимущества:
    - Простота реализации.
    - Не расширяется признаковое пространство (не появляется дополнительных колонок).

    #### Недостатки:
    - Не использует информацию о распределении значений категорий.
    - Не подходит для линейных моделей, так как создает фиктивное отношение порядка между значениями.
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import LabelEncoder

    return (LabelEncoder,)


@app.cell
def _(LabelEncoder, data_1):
    le = LabelEncoder()
    cat_enc_le = le.fit_transform(data_1['Cabin'])
    return cat_enc_le, le


@app.cell
def _(data_1):
    data_1['Cabin'].unique()
    return


@app.cell
def _(cat_enc_le, np):
    np.unique(cat_enc_le)
    return


@app.cell
def _(le):
    le.inverse_transform([0, 1, 2, 3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Для целочисленного кодирования также можно использовать [метод ordinal библиотеки Category Encoders.](https://contrib.scikit-learn.org/category_encoders/ordinal.html) Для того чтобы учесть возможное отношение порядка между значениями категорий, в качестве параметра mapping можно передавать словарь соответствий между значениями  категорий и кодирующими их числами.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Кодирование категорий наборами бинарных значений - [one-hot encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One-hot encoding предполагает, что значение категории заменяется на отдельную колонку, которая содержит бинарные значения.

    #### Преимущества:
    - Простота реализации.
    - Подходит для любых моделей, так как НЕ создает фиктивное отношение порядка между значениями.

    #### Недостатки:
    - Расширяется признаковое пространство.
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import OneHotEncoder

    return (OneHotEncoder,)


@app.cell
def _(OneHotEncoder, data_1):
    ohe = OneHotEncoder(drop='first')
    cat_enc_ohe = ohe.fit_transform(data_1[['Cabin']])
    cat_enc_ohe
    return (cat_enc_ohe,)


@app.cell
def _(cat_enc_ohe):
    cat_enc_ohe.todense()[0:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [Pandas get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) - быстрый вариант one-hot кодирования

    Преимущество в том, что возвращается Pandas DataFrame.
    """)
    return


@app.cell
def _(data_1, pd):
    pd.get_dummies(data_1[['Cabin']], drop_first=True).head()
    return


@app.cell
def _(data_1, pd):
    # Добавление отдельной колонки, признака пустых значений
    pd.get_dummies(data_1[['Cabin']], dummy_na=True, drop_first=True).head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Использование библиотеки [Сategory Encoders](https://contrib.scikit-learn.org/category_encoders/index.html)

    В данной реализации не предусмотрен параметр удаления одной из категорий (drop='first'). В результате после one-hot кодирования все бинарные признаки сохраняются, что приводит к линейной зависимости между ними
    """)
    return


@app.cell
def _():
    from category_encoders.one_hot import OneHotEncoder as ce_OneHotEncoder

    return (ce_OneHotEncoder,)


@app.cell
def _(ce_OneHotEncoder, data_1):
    ce_OneHotEncoder1 = ce_OneHotEncoder()
    data_OHE = ce_OneHotEncoder1.fit_transform(data_1[data_1.columns.difference(['Survived'])])
    return (data_OHE,)


@app.cell
def _(data_OHE):
    data_OHE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Count (frequency) encoding
    """)
    return


@app.cell
def _(Image):
    Image('img/count_frequency_encoding.png', width='50%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Count encoding предполагает что значение категории заменяется на количество раз, которое оно встречается в категории.

    В случае frequency encoding вместо количества используется доля (процент) от количества записей.

    #### Преимущества:
    - Простота реализации.
    - Не расширяется признаковое пространство.

    #### Недостатки:
    - Если два значения встречаются одинаковое количество раз, то они будут заменены на одно и то же количество и становятся неразличимы.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Использование библиотеки [Сategory Encoders](https://contrib.scikit-learn.org/category_encoders/index.html)
    """)
    return


@app.cell
def _():
    from category_encoders.count import CountEncoder as ce_CountEncoder

    return (ce_CountEncoder,)


@app.cell
def _(ce_CountEncoder, data_1):
    ce_CountEncoder1 = ce_CountEncoder()
    data_COUNT_ENC = ce_CountEncoder1.fit_transform(data_1[data_1.columns.difference(['Survived'])])
    return (data_COUNT_ENC,)


@app.cell
def _(data_COUNT_ENC):
    data_COUNT_ENC
    return


@app.cell
def _(data_1):
    data_1['Cabin'].unique()
    return


@app.cell
def _(data_COUNT_ENC):
    data_COUNT_ENC['Cabin'].unique()
    return


@app.cell
def _(ce_CountEncoder, data_1):
    ce_CountEncoder2 = ce_CountEncoder(normalize=True)
    data_FREQ_ENC = ce_CountEncoder2.fit_transform(data_1[data_1.columns.difference(['Survived'])])
    return (data_FREQ_ENC,)


@app.cell
def _(data_FREQ_ENC):
    data_FREQ_ENC
    return


@app.cell
def _(data_FREQ_ENC):
    data_FREQ_ENC['Cabin'].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Target (Mean) encoding
    """)
    return


@app.cell
def _(Image):
    Image('img/target_encoding.png', width='50%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Target encoding предполагает что значение категории заменяется на ранг, соответствующий среднему значению целевого признака для этой категории.

    В случае Mean encoding в кодируемый признак подставляется усредненное значение целевого признака.

    Фактически, ранг зависит от вероятности появления целевого признака, при условии, что кодируемый признак принимает определенное значение.

    Кодирование признака управляется значением целевого признака, возникает монотонная зависимость между кодируемым признаком и целевым признаком.

    #### Преимущества:
    - Наличие монотонной зависимости.
    - Не расширяется признаковое пространство.

    #### Недостатки:
    - Может привести к переобучению.
    - Сложность использования с библиотеками кросс-валидации.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Использование библиотеки [Сategory Encoders](https://contrib.scikit-learn.org/category_encoders/index.html)
    """)
    return


@app.cell
def _():
    # На самом деле этот метод реализует Mean encoding
    from category_encoders.target_encoder import TargetEncoder as ce_TargetEncoder

    return (ce_TargetEncoder,)


@app.cell
def _(ce_TargetEncoder, data_1):
    ce_TargetEncoder1 = ce_TargetEncoder()
    data_MEAN_ENC = ce_TargetEncoder1.fit_transform(data_1[data_1.columns.difference(['Survived'])], data_1['Survived'])
    return (data_MEAN_ENC,)


@app.cell
def _(data_MEAN_ENC):
    data_MEAN_ENC
    return


@app.cell
def _(data_1):
    # Проверка для поля "Пол"
    data_1['Sex'].unique()
    return


@app.cell
def _(data_MEAN_ENC):
    data_MEAN_ENC['Sex'].unique()
    return


@app.cell
def _(data_1):
    def check_mean_encoding(field):
        for s in data_1[field].unique():
            data_filter = data_1[data_1[field] == s]
            if data_filter.shape[0] > 0:
                prob = sum(data_filter['Survived']) / data_filter.shape[0]
                print(s, '-', prob)

    return (check_mean_encoding,)


@app.cell
def _(check_mean_encoding):
    check_mean_encoding('Sex')
    return


@app.cell
def _(data_MEAN_ENC):
    # Проверка для поля "Каюта"
    data_MEAN_ENC['Cabin'].unique()
    return


@app.cell
def _(check_mean_encoding):
    check_mean_encoding('Cabin')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Weight of evidence (WoE) encoding

    $$ WoE = \ln\Big(\frac{\text{Пропорция хороших событий}}{\text{Пропорция плохих  событий}}\Big) $$

    $$ \text{Пропорция хороших (плохих) событий} = \frac{\text{Количество хороших (плохих) событий для заданного значения категории}}{\text{Общее количество хороших (плохих) событий}} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Кодирование признака управляется значением целевого признака, возникает монотонная зависимость между кодируемым признаком и целевым признаком.

    #### Преимущества:
    - Наличие монотонной зависимости.
    - Хорошо подходит для логистической регресии.

    #### Недостатки:
    - Может привести к переобучению.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Использование библиотеки [Сategory Encoders](https://contrib.scikit-learn.org/category_encoders/index.html)
    """)
    return


@app.cell
def _():
    from category_encoders.woe import WOEEncoder as ce_WOEEncoder

    return (ce_WOEEncoder,)


@app.cell
def _(ce_WOEEncoder, data_1):
    ce_WOEEncoder1 = ce_WOEEncoder()
    data_WOE_ENC = ce_WOEEncoder1.fit_transform(data_1[data_1.columns.difference(['Survived'])], data_1['Survived'])
    return (data_WOE_ENC,)


@app.cell
def _(data_WOE_ENC):
    data_WOE_ENC
    return


@app.cell
def _(data_1):
    # Проверка для поля "Пол"
    data_1['Sex'].unique()
    return


@app.cell
def _(data_WOE_ENC):
    data_WOE_ENC['Sex'].unique()
    return


@app.cell
def _(data_1, np):
    def check_woe_encoding(field):
        data_ones = data_1[data_1['Survived'] == 1].shape[0]
        data_zeros = data_1[data_1['Survived'] == 0].shape[0]
        for s in data_1[field].unique():
            data_filter = data_1[data_1[field] == s]
            if data_filter.shape[0] > 0:
                filter_data_ones = data_filter[data_filter['Survived'] == 1].shape[0]
                filter_data_zeros = data_filter[data_filter['Survived'] == 0].shape[0]
                good = filter_data_ones / data_ones
                bad = filter_data_zeros / data_zeros
                woe = np.log(good / bad)
                print(s, '-', woe)

    return (check_woe_encoding,)


@app.cell
def _(check_woe_encoding):
    check_woe_encoding('Sex')
    return


@app.cell
def _(data_WOE_ENC):
    # Проверка для поля "Каюта"
    data_WOE_ENC['Cabin'].unique()
    return


@app.cell
def _(check_woe_encoding):
    check_woe_encoding('Cabin')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.  Helmert encoding

    Используется для упорядоченных категориальных переменных.

    Cравнивает каждый уровень категориальной переменной со средним значением последующих уровней.

    [Детальное пояснение алгоритма кодирования.](https://stats.stackexchange.com/questions/411134/how-to-calculate-helmert-coding)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Использование библиотеки [Сategory Encoders](https://contrib.scikit-learn.org/category_encoders/index.html)
    """)
    return


@app.cell
def _():
    from category_encoders.helmert import HelmertEncoder as ce_HelmertEncoder

    return (ce_HelmertEncoder,)


@app.cell
def _(ce_HelmertEncoder, data_1):
    ce_HelmertEncoder1 = ce_HelmertEncoder()
    data_HELM_ENC = ce_HelmertEncoder1.fit_transform(data_1[data_1.columns.difference(['Survived'])], data_1['Survived'])
    return (data_HELM_ENC,)


@app.cell
def _(data_HELM_ENC):
    data_HELM_ENC
    return


if __name__ == "__main__":
    app.run()
