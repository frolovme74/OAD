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
    # Нормализация числовых признаков

    Нормализация числового признака предполагает что на основе существующего признака мы создаем новый признак, который в идеале имеет нормальное распределение.

    Многие модели машинного обучения лучше работают с нормализованными признаками, особенно линейные модели.

    Какие функции чаще всего используются для нормализации:

    - логарифм - np.log(X)
    - обратная зависимость - 1 / X
    - квадратный корень
    - возведение в степень
    - [преобразование Бокса-Кокса (Box-Cox transformation)](http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%91%D0%BE%D0%BA%D1%81%D0%B0-%D0%9A%D0%BE%D0%BA%D1%81%D0%B0)
    - [преобразование Йео-Джонсона (Yeo-Johnson transformation) - расширение метода Бокса-Кокса для работы с произвольными величинами](https://www.stat.umn.edu/arc/yjpower.pdf)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Как визуально определить, является ли распределение нормальным?**

    1. Гистограмма распределения.

    1. [График квантиль-квантиль](http://statistica.ru/glossary/general/grafiki-kvantil-kvantil/) или [Q-Q plot.](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)
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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    return np, pd, plt, stats


@app.cell
def _(plt, stats):
    def diagnostic_plots(df, variable):
        plt.figure(figsize=(15,6))
        # гистограмма
        plt.subplot(1, 2, 1)
        df[variable].hist(bins=30)
        ## Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(df[variable], dist="norm", plot=plt)
        plt.show()

    return (diagnostic_plots,)


@app.cell
def _(pd):
    # Будем использовать только обучающую выборку
    data = pd.read_csv(
        "data/houseprices.csv",
        keep_default_na=False,
        na_values=["NA"]
    )
    return (data,)


@app.cell
def _(data, plt):
    data.hist(figsize=(20,20))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Исходное распределение

    Можно видеть, что оно отличается от нормального.
    """)
    return


@app.cell
def _(data, diagnostic_plots):
    diagnostic_plots(data, 'GrLivArea')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Логарифмическое преобразование

    Более приближено к нормальному.
    """)
    return


@app.cell
def _(data, diagnostic_plots, np):
    data['GrLivArea_log'] = np.log(data['GrLivArea'])
    diagnostic_plots(data, 'GrLivArea_log')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Обратное преобразование

    Не очень хороший результат.
    """)
    return


@app.cell
def _(data, diagnostic_plots):
    data['GrLivArea_reciprocal'] = 1 / (data['GrLivArea']) 
    diagnostic_plots(data, 'GrLivArea_reciprocal')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Квадратный корень
    """)
    return


@app.cell
def _(data, diagnostic_plots):
    data['GrLivArea_sqr'] = data['GrLivArea']**(1/2) 
    diagnostic_plots(data, 'GrLivArea_sqr')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Возведение в степень
    """)
    return


@app.cell
def _(data, diagnostic_plots):
    data['GrLivArea_exp1'] = data['GrLivArea']**(1/1.5)
    diagnostic_plots(data, 'GrLivArea_exp1')
    return


@app.cell
def _(data, diagnostic_plots):
    data['GrLivArea_exp2'] = data['GrLivArea']**(2)
    diagnostic_plots(data, 'GrLivArea_exp2')
    return


@app.cell
def _(data, diagnostic_plots):
    data['GrLivArea_exp3'] = data['GrLivArea']**(0.333)
    diagnostic_plots(data, 'GrLivArea_exp3')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Преобразование Бокса-Кокса

    Очень хороший результат.
    """)
    return


@app.cell
def _(data, diagnostic_plots, stats):
    data['GrLivArea_boxcox'], _param = stats.boxcox(data['GrLivArea'])
    print('Оптимальное значение λ = {}'.format(_param))
    diagnostic_plots(data, 'GrLivArea_boxcox')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Преобразование Йео-Джонсона

    Очень хороший результат.
    """)
    return


@app.cell
def _(data, diagnostic_plots, stats):
    # Необходимо преобразовать данные к действительному типу
    data['GrLivArea'] = data['GrLivArea'].astype('float')
    data['GrLivArea_yeojohnson'], _param = stats.yeojohnson(data['GrLivArea'])
    print('Оптимальное значение λ = {}'.format(_param))
    diagnostic_plots(data, 'GrLivArea_yeojohnson')
    return


if __name__ == "__main__":
    app.run()
