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
    # Подбор гиперпараметров модели и кросс-валидация
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Рассмотрим задачу регрессии с одним признаком $x$.

    Зададим линейную модель в виде полинома степени $d$:

    $$
    \hat y = w_0 + w_1x + w_2x^2 + \ldots + w_dx^d.
    $$

    Для выборки из $\ell$ объектов матрица признаков имеет вид

    $$
    X =
    \begin{pmatrix}
    1 & x_1 & x_1^2 & \cdots & x_1^d \\
    1 & x_2 & x_2^2 & \cdots & x_2^d \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & x_\ell & x_\ell^2 & \cdots & x_\ell^d
    \end{pmatrix}.
    $$

    Будем обучать модель с использованием квадратичной функции потерь. Как известно, аналитическое решение при данных условиях имеет вид:
    $$
    w^* = \left(X^TX\right)^{-1}X^T y.
    $$

    Степень полинома $d$ не определяется алгоритмом обучения и должна быть задана заранее. Следовательно, $d$ является **гиперпараметром модели**.
    """)
    return


@app.cell
def _():
    from IPython.display import Image
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedKFold
    from sklearn.metrics import mean_squared_error 
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.model_selection import learning_curve, validation_curve
    import seaborn as sns
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    sns.set(style="ticks")
    return (
        GridSearchCV,
        Image,
        KFold,
        LeaveOneOut,
        LeavePOut,
        LinearRegression,
        PolynomialFeatures,
        RandomizedSearchCV,
        RepeatedKFold,
        ShuffleSplit,
        StratifiedKFold,
        cross_val_score,
        cross_validate,
        learning_curve,
        make_pipeline,
        mean_squared_error,
        np,
        pd,
        plt,
        train_test_split,
        validation_curve,
    )


@app.cell
def _(np):
    RANDOM_STATE = 42
    rng = np.random.default_rng(RANDOM_STATE)

    # Число объектов
    l_samples = 30

    # Исходный признак
    x = np.linspace(-1, 1, l_samples)

    # Небольшое случайное смещение координат x
    x += rng.normal(0, 0.015, size=l_samples)
    x = np.clip(x, -1, 1)
    x = np.sort(x)


    # Истинная зависимость
    def f_true(x):
        return (
            0.9 * x**3
            + 0.15 * x**2
            - 1.0 * x
            + 0.15
        )


    # Добавляем шум
    noise_std = 0.07

    y_true = f_true(x)
    y = y_true + rng.normal(0, noise_std, size=l_samples)
    return x, y


@app.cell
def _(np, plt, x, y):
    x_grid = np.linspace(-1, 1, 500)

    plt.figure(figsize=(8, 5))

    plt.scatter(
        x,
        y,
        label="Наблюдения"
    )

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Регрессионная выборка")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell
def _(LinearRegression, PolynomialFeatures, make_pipeline):
    def polynomial_regression(degree):
        return make_pipeline(
            PolynomialFeatures(
                degree=degree,
                include_bias=True
            ),
            LinearRegression(
                fit_intercept=False
            )
        )

    return (polynomial_regression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Недообучение и [переобучение](https://ru.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B5%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5) моделей
    - [Детальная статья про переобучение](http://www.machinelearning.ru/wiki/index.php?title=%D0%9F%D0%B5%D1%80%D0%B5%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5)
    - [Раздел про переобучение из scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
    - [Дилемма смещения–дисперсии](https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D0%BB%D0%B5%D0%BC%D0%BC%D0%B0_%D1%81%D0%BC%D0%B5%D1%89%D0%B5%D0%BD%D0%B8%D1%8F%E2%80%93%D0%B4%D0%B8%D1%81%D0%BF%D0%B5%D1%80%D1%81%D0%B8%D0%B8)
    """)
    return


@app.cell
def _(Image):
    Image('img/bv1.png', width='50%')
    return


@app.cell
def _(x, y):
    X = x.reshape(-1, 1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return (X,)


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=7
    )

    print("Размер всей выборки:", len(X))
    print("Размер train:", len(X_train))
    print("Размер test:", len(X_test))
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, plt, y_test, y_train):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        X_train[:, 0],
        y_train,
        label="Train"
    )

    plt.scatter(
        X_test[:, 0],
        y_test,
        marker="x",
        s=70,
        label="Test"
    )

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Разделение выборки на train и test")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell
def _(X_train, mean_squared_error, polynomial_regression, y_train):
    degrees = [1, 3, 20]

    models = {}
    train_mse = {}

    for degree in degrees:
        model = polynomial_regression(degree)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        models[degree] = model
        train_mse[degree] = mean_squared_error(y_train, y_train_pred)

    for degree in degrees:
        print(
            f"d = {degree:2d} | "
            f"Train MSE = {train_mse[degree]:.10e}"
        )
    return degrees, models


@app.cell
def _(X, X_train, degrees, models, np, plt, y_train):
    X_grid = np.linspace(
        X.min(),
        X.max(),
        1000
    ).reshape(-1, 1)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        X_train[:, 0],
        y_train,
        s=45,
        label="Train"
    )

    for _degree in degrees:
        y_grid_pred = models[_degree].predict(X_grid)

        plt.plot(
            X_grid[:, 0],
            y_grid_pred,
            linewidth=2,
            label=f"d = {_degree}"
        )

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim(-0.5, 1.0)
    plt.title("Модели различной сложности")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell
def _(
    X_test,
    X_train,
    degrees,
    mean_squared_error,
    models,
    pd,
    y_test,
    y_train,
):
    results_df = pd.DataFrame([
        {
            "degree": _degree,
            "train_MSE": mean_squared_error(y_train, models[_degree].predict(X_train)),
            "test_MSE": mean_squared_error(y_test, models[_degree].predict(X_test)),
        }
        for _degree in degrees
    ])

    results_df
    return (results_df,)


@app.cell
def _(plt, results_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["degree"],
        results_df["train_MSE"],
        marker="o",
        label="Train MSE"
    )

    plt.plot(
        results_df["degree"],
        results_df["test_MSE"],
        marker="o",
        label="Test MSE"
    )

    plt.yscale("log")

    plt.xlabel("Степень полинома $d$")
    plt.ylabel("MSE")
    plt.title("Ошибка на обучающей и тестовой выборках")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell
def _(X, X_test, degrees, models, np, plt, y, y_test):
    X_grid_test = np.linspace(
        X.min(),
        X.max(),
        1000
    ).reshape(-1, 1)

    plt.figure(figsize=(10, 6))

    # Показываем только тестовые объекты
    plt.scatter(
        X_test[:, 0],
        y_test,
        s=60,
        marker="x",
         color="red",
        label="Test"
    )

    for _degree in degrees:
        y_grid_pred_test = models[_degree].predict(X_grid_test)

        plt.plot(
            X_grid_test[:, 0],
            y_grid_pred_test,
            linewidth=2,
            label=f"d = {_degree}"
        )

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Поведение обученных моделей на тестовой выборке")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.ylim(
        y.min() - 0.2,
        y.max() + 0.2
    )

    plt.show()
    return


@app.cell
def _(degrees, models):
    for _degree in degrees:
        weights = models[_degree].named_steps["linearregression"].coef_

        print(f"d = {_degree}")
        for i, w in enumerate(weights):
            print(f"w_{i} = {w:.6f}")

        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Таким образом, сложность модели никак не ограничена: при достаточно большой степени полинома модель может практически идеально подстроиться под обучающую выборку, подобрав соответствующие значения коэффициентов, однако при этом потерять обобщающую способность.

    Более того, если подбирать степень $d$ по обучающей выборке, то такой подход будет поощрять всё более сложные модели. В нашем случае при $\ell_{\mathrm{train}}$ различных обучающих объектах полином степени $\ell_{\mathrm{train}}−1$ уже [способен интерполировать](https://ru.wikipedia.org/wiki/Интерполяция_алгебраическими_многочленами) все точки и дать практически нулевую ошибку на train.

    Поэтому подбирать гиперпараметры по обучающей выборке **недопустимо**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Схема обучения
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ранее использовалась следующая (упрощённая) схема обучения модели:
    """)
    return


@app.cell
def _(Image):
    Image("img/grid_search_workflow_simple.png", width="50%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Исходный набор данных делится на обучающую и тестовую выборку (например, с помощью функции train_test_split)
    1. Обучающая выборка используется для обучения модели.
    1. Тестовая выборка используется для оценки качества модели на основе выбранных метрик качества.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **К основным недостаткам** упрощенной модели можно отнести:
    - Ручная установка гиперпараметров. Гиперпараметры модели устанавливаются случайным образом и непонятно как найти комбинацию гиперпараметров, которая будет лучшей для модели.
    - Однократное разбиение на обучающую и тестовую выборку приводит к вычислению "неточной" (смещенной) оценки метрики качества.

    Для того, чтобы подобрать гиперпараметры, необходимо перебрать все возможные варианты гиперпараметров и данных для всего исходного датасета (не только для обучающей выборки).

    Таким образом, необходимо решить две задачи:

    1) [Оптимизация гиперпараметров.](https://ru.wikipedia.org/wiki/%D0%9E%D0%BF%D1%82%D0%B8%D0%BC%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F_%D0%B3%D0%B8%D0%BF%D0%B5%D1%80%D0%BF%D0%B0%D1%80%D0%B0%D0%BC%D0%B5%D1%82%D1%80%D0%BE%D0%B2) Наиболее простой метод - Grid Search (поиск по решетке, решетчатый поиск).

    2) Оптимизацию гиперпараметров необходимо выполнять на полном датасете. Таким образом, исходный датасет необходимо разбить на обучающие и тестовые выборки в различных комбинациях, чтобы получить несмещенную оценку вычисляемой метрики. Этот процесс называют [кросс-валидация](https://ru.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B5%D0%BA%D1%80%D1%91%D1%81%D1%82%D0%BD%D0%B0%D1%8F_%D0%BF%D1%80%D0%BE%D0%B2%D0%B5%D1%80%D0%BA%D0%B0) (скользящий контроль, перекрестная проверка), англ. [cross-validation.](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Для того, чтобы избежать недообучения и переобучения используется следующая схема обучения (иллюстрация из описания библиотеки scikit-learn):
    """)
    return


@app.cell
def _(Image):
    Image('img/grid_search_workflow_2.png', width='80%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # [Кросс-валидация](https://ru.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B5%D0%BA%D1%80%D1%91%D1%81%D1%82%D0%BD%D0%B0%D1%8F_%D0%BF%D1%80%D0%BE%D0%B2%D0%B5%D1%80%D0%BA%D0%B0) (скользящий контроль, перекрестная проверка), [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
    - [Раздел по кросс-валидации в scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В Википедии смысл перекрестной проверки объясняется следующим образом:

    Перекрестная проверка — метод оценки аналитической модели и её поведения на независимых данных. При оценке модели имеющиеся в наличии данные разбиваются на k частей. Затем на k−1 частях данных производится обучение модели, а оставшаяся часть данных используется для тестирования. Процедура повторяется k раз; в итоге каждая из k частей данных используется для тестирования. В результате получается оценка эффективности выбранной модели с наиболее равномерным использованием имеющихся данных.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    К данному определению необходимо сделать следующие примечания:
    - Каждую из k частей принято называть fold (фолд).
    - Количество фолдов при кросс-валидации обычно обозначают параметром cv.
    - В результате каждого обучения и проверки модели формируются указанные разработчиком метрики качества модели.

    Например, в результате кросс-валидации для 3 фолдов при решении задачи регрессии можно получить 3 значения MSE, которые были вычислены для каждой комбинации фолдов. Полученные метрики обычно усредняют.

    <table style="text-align:left">
        <tr>
            <td>Fold 1</td>
            <td>Fold 2</td>
            <td>Fold 3</td>
            <td>MSE</td>
        </tr>

        <tr>
            <td style="background-color:red">Fold 1</td>
            <td>Fold 2</td>
            <td>Fold 3</td>
            <td>MSE1</td>
        </tr>

        <tr>
            <td>Fold 1</td>
            <td style="background-color:red">Fold 2</td>
            <td>Fold 3</td>
            <td>MSE2</td>
        </tr>

        <tr>
            <td>Fold 1</td>
            <td>Fold 2</td>
            <td style="background-color:red">Fold 3</td>
            <td>MSE3</td>
        </tr>

        <tr>
            <td colspan="4">
                MSE = np.mean([MSE1, MSE2, MSE3])
            </td>
        </tr>
    </table>

    - Существуют различные способы (стратегии) разбиения выборки на k частей, которые называют стратегиями кросс-валидации (cross-validation splitting strategy).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Наиболее простым способом кросс-валидации является вызов функции [cross_val_score.](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score) В этом случае стратегия кросс-валидации определяется автоматически.
    """)
    return


@app.cell
def _(X_train, cross_val_score, polynomial_regression, y_train):
    scores_1 = -cross_val_score(
        polynomial_regression(degree=3),
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    return (scores_1,)


@app.cell
def _(scores_1):
    # Значение метрики для 3 фолдов
    scores_1
    return


@app.cell
def _(np, scores_1):
    # Усредненное значение метрики для 3 фолдов
    np.mean(scores_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В отличие от функции cross_val_score, функция [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) позволяет использовать для оценки несколько метрик и возращает более детальную информацию.
    """)
    return


@app.cell
def _():
    scoring = {
        "MSE": "neg_mean_squared_error",
        "RMSE": "neg_root_mean_squared_error"
    }
    return (scoring,)


@app.cell
def _(X_train, cross_validate, polynomial_regression, scoring, y_train):
    scores_2 = cross_validate(
        polynomial_regression(degree=3),
        X_train,
        y_train,
        cv=5,
        scoring=scoring
    )
    return (scores_2,)


@app.cell
def _(scores_2):
    mse_2 = -scores_2["test_MSE"]
    rmse_2 = -scores_2["test_RMSE"]

    print("MSE:", mse_2)
    print("RMSE:", rmse_2)
    return mse_2, rmse_2


@app.cell
def _(mse_2, np, rmse_2):
    print("Mean MSE:", np.mean(mse_2))
    print("Mean RMSE:", np.mean(rmse_2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Стратегии кросс-валидации

    Все стратегии кросс-валидации основаны на том, что разработчик задает или количество разбиений (в этом случае автоматически определяются размеры обучающей и тестовой выборок для каждого разбиения) или размер обучающей или тестовой выборки для одного разбиения (в этом случае автоматически определяется количество разбиений).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [K-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold)

    Данная стратегия работает в соответствии с определением кросс-валидации.

    Каждой стратегии в scikit-learn ставится в соответствии специальный класс-итератор, который может быть указан в качестве параметра cv функций cross_val_score и cross_validate.
    """)
    return


@app.cell
def _(KFold):
    # Возвращаются индексы элементов
    _X = ['a', 'b', 'c']
    _kf = KFold(n_splits=3)
    for _train, _test in _kf.split(_X):
        print('%s %s' % (_train, _test))
    return


@app.cell
def _(KFold):
    _X = range(12)
    _kf = KFold(n_splits=3)
    for _train, _test in _kf.split(_X):
        print('%s %s' % (_train, _test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Все элементы данных по очереди фигурируют и в обучающей и в тестовой выборках.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Далее итератор может быть использован в функциях cross_val_score и cross_validate:
    """)
    return


@app.cell
def _(KFold, X_train, cross_val_score, polynomial_regression, y_train):
    _kf = KFold(n_splits=5)
    scores_3 = -cross_val_score(
        polynomial_regression(degree=3),
        X_train,
        y_train,
        cv=_kf,
        scoring="neg_mean_squared_error"
    )
    scores_3
    return


@app.cell
def _(X_train, cross_validate, polynomial_regression, scoring, y_train):
    scores_4 = cross_validate(
        polynomial_regression(degree=3),
        X_train,
        y_train,
        cv=5,
        scoring=scoring
    )
    mse_4 = -scores_4["test_MSE"]
    rmse_4 = -scores_4["test_RMSE"]

    print("MSE:", mse_4)
    print("RMSE:", rmse_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [Repeated K-Fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold)

    Повторяет стратегию K-Fold заданное количество раз.
    """)
    return


@app.cell
def _(RepeatedKFold):
    _X = range(12)
    _kf = RepeatedKFold(n_splits=3, n_repeats=2)
    for _train, _test in _kf.split(_X):
        print('%s %s' % (_train, _test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [Leave One Out (LOO)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut)

    В тестовую выборку помещается единственный элемент (One Out). Количество фолдов в этом случае определяется автоматически и равняется количеству элементов.

    Данный метод более ресурсоемкий чем KFold.

    Существует эмпирическое правило, что вместо Leave One Out лучше использовать KFold на 5 или 10 фолдов.
    """)
    return


@app.cell
def _(LeaveOneOut):
    _X = range(12)
    # Эквивалент KFold(n_splits=n)
    _kf = LeaveOneOut()
    for _train, _test in _kf.split(_X):
        print('%s %s' % (_train, _test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [Leave P Out (LPO)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html#sklearn.model_selection.LeavePOut)

    В тестовую выборку помещается P элементов (P Out), полностью перебираются все уникальные комбинации из P элементов. Количество фолдов в этом случае определяется автоматически.
    """)
    return


@app.cell
def _(LeavePOut):
    _X = range(12)
    # Эквивалент KFold(n_splits=n)
    _kf = LeavePOut(2)
    for _train, _test in _kf.split(_X):
        print('%s %s' % (_train, _test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit)

    Генерируется N случайных перемешиваний данных, в каждом перемешивании заданная доля помещается в тестовую выборку.
    """)
    return


@app.cell
def _(ShuffleSplit):
    _X = range(12)
    _kf = ShuffleSplit(n_splits=5, test_size=0.25)
    for _train, _test in _kf.split(_X):
        print('%s %s' % (_train, _test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Модели со стратификацией используются для дисбалансированных выборок

    Разбиения производятся таким образом, что сохраняется баланс классов целевого признака.

    - [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)
    - [StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit)
    """)
    return


@app.cell
def _(StratifiedKFold, np):
    _X = np.ones(10)
    _y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    _skf = StratifiedKFold(n_splits=3)
    for _train, _test in _skf.split(_X, _y):
        print('%s %s' % (_train, _test))
    return


@app.cell
def _(StratifiedKFold, np):
    _X = np.ones(10)
    _y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    _skf = StratifiedKFold(n_splits=5)
    for _train, _test in _skf.split(_X, _y):
        print('%s %s' % (_train, _test))
    return


@app.cell
def _(StratifiedKFold, np):
    _X = np.ones(10)
    _y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    _skf = StratifiedKFold(n_splits=5)
    for _train, _test in _skf.split(_X, _y):
        print('%s %s' % (_train, _test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # [Оптимизация гиперпараметров](https://scikit-learn.org/stable/modules/grid_search.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) (решетчатый поиск)
    """)
    return


@app.cell
def _(np):
    degree_range = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20])

    tuned_parameters = [
        {"polynomialfeatures__degree": degree_range}
    ]
    tuned_parameters
    return degree_range, tuned_parameters


@app.cell
def _(GridSearchCV, X_train, polynomial_regression, tuned_parameters, y_train):
    model_gs = GridSearchCV(
        polynomial_regression(degree=1),  
        tuned_parameters,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    model_gs.fit(X_train, y_train)
    return (model_gs,)


@app.cell
def _(model_gs):
    model_gs.cv_results_
    return


@app.cell
def _(model_gs):
    # Лучшая модель
    model_gs.best_estimator_
    return


@app.cell
def _(model_gs):
    # Лучшее среднее значение MSE на кросс-валидации
    -model_gs.best_score_
    return


@app.cell
def _(model_gs):
    model_gs.best_params_
    return


@app.cell
def _(degree_range, model_gs, plt):
    # Лучшее значение параметров# Изменение качества на кросс-валидации
    # в зависимости от степени полинома

    plt.figure(figsize=(8, 5))

    plt.plot(
        degree_range,
        -model_gs.cv_results_["mean_test_score"],
        marker="o"
    )
    plt.yscale("log")
    plt.xlabel("Степень полинома $d$")
    plt.ylabel("MSE")
    plt.title("Зависимость MSE от степени полинома")
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [Randomized Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) (случайный поиск)

    Используется в том случае когда полный решетчатый поиск работает слишком долго.
    """)
    return


@app.cell
def _(
    RandomizedSearchCV,
    X_train,
    polynomial_regression,
    tuned_parameters,
    y_train,
):
    model_rs = RandomizedSearchCV(
        polynomial_regression(degree=1),
        tuned_parameters,
        n_iter=5,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42
    )

    model_rs.fit(X_train, y_train)
    return (model_rs,)


@app.cell
def _(model_rs):
    # В данном случае оба способа нашли одинаковое решение
    -model_rs.best_score_, model_rs.best_params_
    return


@app.cell
def _(model_gs):
    -model_gs.best_score_, model_gs.best_params_
    return


@app.cell
def _(model_rs, np, plt):
    rs_degrees = np.array(
        model_rs.cv_results_["param_polynomialfeatures__degree"],
        dtype=int
    )

    rs_mse = -model_rs.cv_results_["mean_test_score"]

    order = np.argsort(rs_degrees)

    plt.figure(figsize=(8, 5))

    plt.plot(
        rs_degrees[order],
        rs_mse[order],
        marker="o"
    )
    plt.yscale("log")
    plt.xlabel("Степень полинома $d$")
    plt.ylabel("MSE")
    plt.title("Randomized Search")
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Примечания.**
    - Для получения более устойчивых результатов подбора гиперпараметров и оптимизации модели машинного обучения  используется метод вложенной кросс-валидации ([Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py)). Детали вложенной кросс-валидации рассмотрены в [статье.](https://www.kdnuggets.com/2020/10/nested-cross-validation-python.html)
    - В разделе 3.2.3 [руководства](https://scikit-learn.org/stable/modules/grid_search.html) рассмотрены советы по подбору параметров.
    - В разделе 3.2.4 [руководства](https://scikit-learn.org/stable/modules/grid_search.html) рассмотрены варианты подбора параметров для некоторых классов моделей машинного обучения.
    - Часто RandomizedSearchCV применяют для поиска "опорного" решения, а затем в "окрестности" полученного решения используют GridSearchCV.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Другие способы

    Для оптимизации гиперпараметров также существуют отдельные от scikit-learn библиотеки и подходы:
    - https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a
    - https://towardsdatascience.com/algorithms-for-hyperparameter-optimisation-in-python-edda4bdb167
    - http://hyperopt.github.io/hyperopt/
    - https://ray.readthedocs.io/en/latest/tune.html
    - https://parameter-sherpa.readthedocs.io/en/latest/
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # [Построение кривых обучения и валидации](https://scikit-learn.org/stable/modules/learning_curve.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Построение кривых обучения - [learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve)

    Строится зависимость метрики на обучающей выборке от размера выборки.
    """)
    return


@app.cell
def _(learning_curve, np, plt):
    def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        ylim=None,
        cv=5,
        n_jobs=None,
        train_sizes=np.linspace(0.3, 1.0, 5),
        scoring="neg_mean_squared_error"
    ):
        """
        Построение кривой обучения для задачи регрессии.

        Для разных размеров обучающей выборки вычисляются:
        - ошибка на обучающей части;
        - ошибка на валидационных фолдах кросс-валидации.
        """

        plt.figure(figsize=(8, 5))
        plt.title(title)

        if ylim is not None:
            plt.ylim(*ylim)

        plt.xlabel("Размер обучающей выборки")
        plt.ylabel("MSE")

        train_sizes, train_scores, validation_scores = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            train_sizes=train_sizes
        )

        # sklearn возвращает -MSE, поэтому меняем знак
        train_scores = -train_scores
        validation_scores = -validation_scores

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        validation_scores_mean = np.mean(validation_scores, axis=1)
        validation_scores_std = np.std(validation_scores, axis=1)

        plt.grid(alpha=0.3)

        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.3,
            color="r"
        )

        plt.fill_between(
            train_sizes,
            validation_scores_mean - validation_scores_std,
            validation_scores_mean + validation_scores_std,
            alpha=0.1,
            color="g"
        )

        plt.plot(
            train_sizes,
            train_scores_mean,
            "o-",
            color="r",
            label="Train MSE"
        )

        plt.plot(
            train_sizes,
            validation_scores_mean,
            "o-",
            color="g",
            label="Cross-validation MSE"
        )

        plt.legend(loc="best")

        return plt

    return (plot_learning_curve,)


@app.cell
def _(X_train, plot_learning_curve, plt, polynomial_regression, y_train):
    plot_learning_curve(
        polynomial_regression(degree=5),
        "Кривая обучения",
        X_train,
        y_train,
        cv=5
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Построение кривой валидации - [validation_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve)

    Строится зависимость метрики на тестовой выборке от одного из гиперпараметров.
    """)
    return


@app.cell
def _(np, plt, validation_curve):
    def plot_validation_curve(
        estimator,
        title,
        X,
        y,
        param_name,
        param_range,
        cv=5,
        scoring="neg_mean_squared_error"
    ):
        train_scores, validation_scores = validation_curve(
            estimator,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )

        # sklearn возвращает -MSE
        train_scores = -train_scores
        validation_scores = -validation_scores

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        validation_scores_mean = np.mean(validation_scores, axis=1)
        validation_scores_std = np.std(validation_scores, axis=1)

        plt.figure(figsize=(8, 5))
        plt.title(title)

        plt.yscale("log")

        plt.xlabel("Степень полинома $d$")
        plt.ylabel("MSE")

        plt.plot(
            param_range,
            train_scores_mean,
            label="Train MSE",
            marker="o"
        )

        plt.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.2
        )

        plt.plot(
            param_range,
            validation_scores_mean,
            label="Cross-validation MSE",
            marker="o"
        )

        plt.fill_between(
            param_range,
            validation_scores_mean - validation_scores_std,
            validation_scores_mean + validation_scores_std,
            alpha=0.2
        )

        plt.grid(alpha=0.3)
        plt.legend(loc="best")

        return plt

    return (plot_validation_curve,)


@app.cell
def _(
    X_train,
    degree_range,
    plot_validation_curve,
    plt,
    polynomial_regression,
    y_train,
):
    plot_validation_curve(
        polynomial_regression(degree=1),
        "Валидационная кривая",
        X_train,
        y_train,
        param_name="polynomialfeatures__degree",
        param_range=degree_range,
        cv=5
    )

    plt.show()
    return


@app.cell
def _(model_gs):
    model_gs.best_params_
    return


@app.cell
def _(model_gs):
    best_model = model_gs.best_estimator_
    return (best_model,)


@app.cell
def _(X_test, X_train, best_model, mean_squared_error, y_test, y_train):
    final_train_mse = mean_squared_error(
        y_train,
        best_model.predict(X_train)
    )

    test_mse = mean_squared_error(
        y_test,
        best_model.predict(X_test)
    )

    final_train_mse, test_mse
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Дополнительные источники:
    1. http://www.machinelearning.ru/wiki/index.php?title=CV
    1. http://datascientist.one/cross-validation/
    1. http://datareview.info/article/sravnenie-razlichnyih-vidov-kross-validatsii/
    1. https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229
    """)
    return


if __name__ == "__main__":
    app.run()
