import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Градиентные методы оптимизации

    ## Метод наискорейшего спуска
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    from matplotlib import cm

    from mpl_toolkits.mplot3d import Axes3D

    return cm, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Пусть требуется минимизировать дифференцируемую функцию

    $$
    f(x) \to \min_x,
    \qquad
    x \in \mathbb{R}^d.
    $$

    Рассмотрим малое приращение $h$. В линейном приближении

    $$
    f(x+h)
    \approx
    f(x)+\langle \nabla f(x), h\rangle.
    $$

    Чтобы значение функции уменьшалось, необходимо выбрать $h$ так, чтобы

    $$
    \langle \nabla f(x), h\rangle < 0.
    $$

    Как известно, градиент $\nabla f(x)$ задаёт направление наискорейшего роста функции. Поэтому направление наискорейшего убывания задаётся антиградиентом.

    Пусть $s>0$ — длина перемещения. Выберем

    $$
    h
    =
    -s\frac{\nabla f(x)}{\|\nabla f(x)\|}.
    $$

    Тогда

    $$
    f(x+h)-f(x)
    \approx
    \left\langle
    \nabla f(x),
    -s\frac{\nabla f(x)}{\|\nabla f(x)\|}
    \right\rangle
    =
    -s\|\nabla f(x)\| < 0.
    $$

    Следовательно, в линейном приближении движение вдоль антиградиента уменьшает значение функции.

    Перепараметризуем длину перемещения:

    $$
    \eta
    =
    \frac{s}{\|\nabla f(x)\|}.
    $$

    Тогда

    $$
    h=-\eta\nabla f(x),
    $$

    а изменение функции в линейном приближении имеет вид

    $$
    f(x+h)-f(x)
    \approx
    -\eta\|\nabla f(x)\|^2.
    $$

    Получаем итерацию градиентного спуска:

    $$
    \boxed{
    x^{(k+1)}
    =
    x^{(k)}
    -
    \eta_k\nabla f\left(x^{(k)}\right)
    }
    $$

    Параметр $\eta_k>0$ называется **шагом обучения** (*learning rate*).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Квадратичный функционал в линейной регрессии

    Рассмотрим линейную модель с двумя признаками

    $$
    \alpha(x)
    =
    w_1x_{1}+w_2x_{2}.
    $$

    Будем минимизировать квадратичный функционал

    $$
    Q(w)
    =
    \frac{1}{2\ell}
    \sum_{i=1}^{\ell}
    \left(
    \langle w,x_i\rangle-y_i
    \right)^2.
    $$

    Поскольку модель содержит два параметра $w_1$ и $w_2$, значение функционала можно визуализировать как поверхность

    $$
    Q(w_1,w_2).
    $$

    Рассмотрим две выборки.

    В первой признаки имеют сопоставимые масштабы.

    Во второй масштаб одного из признаков существенно меньше, вследствие чего функционал значительно слабее изменяется по соответствующей координате параметров.
    """)
    return


@app.cell
def _(np):
    RANDOM_STATE = 42
    rng = np.random.default_rng(RANDOM_STATE)

    n_samples = 300

    w_true = np.array([
        2.0,
        -3.0,
    ])
    return RANDOM_STATE, n_samples, rng, w_true


@app.cell
def _(n_samples, rng):
    X_good = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, 2),
    )

    X_bad = X_good.copy()
    X_bad[:, 1] *= 0.1
    return X_bad, X_good


@app.cell
def _(X_bad, X_good, w_true):
    y_good = X_good @ w_true
    y_bad = X_bad @ w_true
    return y_bad, y_good


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Истинный вектор параметров одинаков в обеих задачах:

    $$
    w^*
    =
    \begin{pmatrix}
    2 \\
    -3
    \end{pmatrix}.
    $$

    Различается только масштаб признаков.

    В хорошо обусловленной задаче функционал изменяется сопоставимо по $w_1$ и $w_2$.

    В плохо обусловленной задаче изменение функционала по $w_2$ значительно слабее.
    """)
    return


@app.cell
def _(np):
    def objective(w, X, y):
        residuals = X @ w - y

        return 0.5 * np.mean(
            residuals**2
        )


    def objective_grad(w, X, y):
        residuals = X @ w - y

        return (
            X.T @ residuals
            / len(y)
        )

    return objective, objective_grad


@app.cell
def _(X_bad, X_good, objective_grad, w_true, y_bad, y_good):
    print(
        "Градиент в точке w_true"
    )

    print(
        "good:",
        objective_grad(
            w_true,
            X_good,
            y_good,
        ),
    )

    print(
        "bad: ",
        objective_grad(
            w_true,
            X_bad,
            y_bad,
        ),
    )
    return


@app.cell
def _(np):
    def make_objective_function(X, y):
        def Q(w1, w2):
            shape = np.broadcast(w1, w2).shape

            w1_flat = np.broadcast_to(
                w1,
                shape,
            ).ravel()

            w2_flat = np.broadcast_to(
                w2,
                shape,
            ).ravel()

            W = np.column_stack([
                w1_flat,
                w2_flat,
            ])

            residuals = (
                X @ W.T
                - y[:, None]
            )

            values = 0.5 * np.mean(
                residuals**2,
                axis=0,
            )

            return values.reshape(shape)

        return Q

    return (make_objective_function,)


@app.cell
def _(X_bad, X_good, make_objective_function, y_bad, y_good):
    Q_good = make_objective_function(
        X_good,
        y_good,
    )

    Q_bad = make_objective_function(
        X_bad,
        y_bad,
    )
    return Q_bad, Q_good


@app.cell
def _(cm, np, plt):
    def plot_surface_pair(
        f_left,
        f_right,
        titles,
        limits_w1=(-4, 6),
        limits_w2=(-6, 4),
        n_points=200,
    ):
        w1 = np.linspace(*limits_w1, n_points)
        w2 = np.linspace(*limits_w2, n_points)

        W1, W2 = np.meshgrid(w1, w2)

        Z_left = f_left(W1, W2)
        Z_right = f_right(W1, W2)

        fig = plt.figure(figsize=(14, 5))

        for i, (Z, title) in enumerate(
            zip(
                (Z_left, Z_right),
                titles,
            ),
            start=1,
        ):
            ax = fig.add_subplot(
                1,
                2,
                i,
                projection="3d",
            )

            ax.plot_surface(
                W1,
                W2,
                Z,
                cmap=cm.viridis,
                alpha=0.9,
            )

            ax.set_xlim(*limits_w1)
            ax.set_ylim(*limits_w2)

            ax.set_xlabel("$w_1$")
            ax.set_ylabel("$w_2$")
            ax.set_zlabel("$Q(w_1, w_2)$")

            ax.set_title(title)

        plt.tight_layout()
        plt.show()

    return (plot_surface_pair,)


@app.cell
def _(cm, np, plt):
    def plot_contour_pair(
        f_left,
        f_right,
        titles,
        limits_w1=(-4, 6),
        limits_w2=(-6, 4),
        n_points=400,
        levels=20,
    ):
        w1 = np.linspace(*limits_w1, n_points)
        w2 = np.linspace(*limits_w2, n_points)

        W1, W2 = np.meshgrid(w1, w2)

        Z_left = f_left(W1, W2)
        Z_right = f_right(W1, W2)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
        )

        for ax, Z, title in zip(
            axes,
            (Z_left, Z_right),
            titles,
        ):
            ax.contour(
                W1,
                W2,
                Z,
                levels=levels,
                cmap=cm.viridis,
            )

            ax.set_xlim(*limits_w1)
            ax.set_ylim(*limits_w2)

            ax.set_xlabel("$w_1$")
            ax.set_ylabel("$w_2$")

            ax.set_title(title)
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

    return (plot_contour_pair,)


@app.cell
def _(w_true):
    plot_radius = 5

    limits_w1 = (
        w_true[0] - plot_radius,
        w_true[0] + plot_radius,
    )

    limits_w2 = (
        w_true[1] - plot_radius,
        w_true[1] + plot_radius,
    )
    return limits_w1, limits_w2


@app.cell
def _(Q_bad, Q_good, limits_w1, limits_w2, plot_surface_pair):
    plot_surface_pair(
        Q_good,
        Q_bad,
        titles=(
            "MSE: хорошая обусловленность",
            "MSE: плохая обусловленность",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
    )
    return


@app.cell
def _(Q_bad, Q_good, limits_w1, limits_w2, plot_contour_pair):
    plot_contour_pair(
        Q_good,
        Q_bad,
        titles=(
            "Линии уровня MSE: хорошая обусловленность",
            "Линии уровня MSE: плохая обусловленность",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Градиентный спуск

    На каждой итерации будем вычислять градиент функционала по всей выборке и выполнять шаг

    $$
    w^{(k+1)}
    =
    w^{(k)}
    -
    \eta \nabla Q\left(w^{(k)}\right).
    $$

    На практике необходимо определить критерий остановки алгоритма. Наиболее распространённые варианты:

    1. Норма градиента стала достаточно малой:

    $$
    \|\nabla Q(w^{(k)})\|_2
    <
    \varepsilon.
    $$

    2. Изменение значения функционала стало достаточно малым:

    $$
    \left|
    Q(w^{(k+1)})
    -
    Q(w^{(k)})
    \right|
    <
    \varepsilon.
    $$

    3. Достигнуто максимальное число итераций:

    $$
    k \ge N.
    $$

    В наших экспериментах будем использовать третий критерий и запускать методы на фиксированное число итераций. Это позволит сравнивать их траектории на одинаковом числе шагов.


    Для квадратичного функционала линейной регрессии

    $$
    Q(w)
    =
    \frac{1}{2\ell}
    \|Xw-y\|^2
    $$

    градиент имеет вид

    $$
    \nabla Q(w)
    =
    \frac{1}{\ell}
    X^T(Xw-y).
    $$

    Градиент $Q$ является $L$-липшицевым:

    $$
    \|\nabla Q(u)-\nabla Q(v)\|
    \le
    L\|u-v\|.
    $$

    Константа $L$ ограничивает скорость изменения градиента.

    Положим постоянный шаг

    $$
    \eta
    =
    \frac{1}{L}.
    $$

    При таком выборе шага получаем теоретическую гарантию сходимости градиентного спуска.

    Для рассматриваемого квадратичного функционала константу $L$ можно вычислить как

    $$
    L
    =
    \|\nabla^2 Q(w)\|_2
    =
    \frac{\|X^TX\|_2}{\ell}.
    $$

    Поскольку столбцы матрицы признаков линейно независимы, функционал строго выпуклый и имеет единственный минимум.
    """)
    return


@app.cell
def _(X_bad, X_good, np):
    def lipschitz_constant(X):
        H = X.T @ X / len(X)
        return np.linalg.norm(H, ord=2)


    L_good = lipschitz_constant(X_good)
    L_bad = lipschitz_constant(X_bad)

    eta_good = 1.0 / L_good
    eta_bad = 1.0 / L_bad

    print(f"L_good   = {L_good:.4f}")
    print(f"L_bad    = {L_bad:.4f}")

    print(f"eta_good = {eta_good:.4f}")
    print(f"eta_bad  = {eta_bad:.4f}")
    return L_bad, L_good, eta_bad, eta_good


@app.cell
def _(np, objective, objective_grad):
    def gradient_descent(
        X,
        y,
        w_init,
        eta,
        n_steps=100,
    ):
        w = np.array(w_init, dtype=float)

        trajectory = [w.copy()]
        losses = [objective(w, X, y)]

        for _ in range(n_steps):
            grad = objective_grad(w, X, y)

            w = w - eta * grad

            trajectory.append(w.copy())
            losses.append(objective(w, X, y))

        return np.array(trajectory), np.array(losses)

    return (gradient_descent,)


@app.cell
def _(X_bad, X_good, eta_bad, eta_good, gradient_descent, np, y_bad, y_good):
    w_init = np.array([
        -2.5,
        1.5,
    ])

    n_steps = 100

    trajectory_good, loss_good = gradient_descent(
        X=X_good,
        y=y_good,
        w_init=w_init,
        eta=eta_good,
        n_steps=n_steps,
    )

    trajectory_bad, loss_bad = gradient_descent(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        eta=eta_bad,
        n_steps=n_steps,
    )
    return trajectory_bad, trajectory_good, w_init


@app.cell
def _(trajectory_bad, trajectory_good, w_true):
    print("True optimum:", w_true)
    print("GD good:", trajectory_good[-1])
    print("GD bad: ", trajectory_bad[-1])
    return


@app.cell
def _(cm, np, plt, w_true):
    def plot_contour_trajectories(
        f_left,
        f_right,
        trajectory_left,
        trajectory_right,
        titles,
        limits_w1,
        limits_w2,
        n_points=400,
        levels=20,
        method_name="GD",
    ):
        w1 = np.linspace(*limits_w1, n_points)
        w2 = np.linspace(*limits_w2, n_points)

        W1, W2 = np.meshgrid(w1, w2)

        Z_left = f_left(W1, W2)
        Z_right = f_right(W1, W2)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
        )

        for ax, Z, trajectory, title in zip(
            axes,
            (Z_left, Z_right),
            (trajectory_left, trajectory_right),
            titles,
        ):
            ax.contour(
                W1,
                W2,
                Z,
                levels=levels,
                cmap=cm.viridis,
            )

            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=method_name,
            )

            ax.scatter(
                w_true[0],
                w_true[1],
                marker="*",
                s=150,
                label="$w^*$",
            )

            ax.set_xlim(*limits_w1)
            ax.set_ylim(*limits_w2)

            ax.set_xlabel("$w_1$")
            ax.set_ylabel("$w_2$")

            ax.set_title(title)
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            ax.legend()

        plt.tight_layout()
        plt.show()

    return (plot_contour_trajectories,)


@app.cell
def _(cm, np, plt, w_true):
    def plot_surface_trajectories(
        f_left,
        f_right,
        trajectory_left,
        trajectory_right,
        titles,
        limits_w1,
        limits_w2,
        n_points=200,
        method_name="GD",
    ):
        w1 = np.linspace(*limits_w1, n_points)
        w2 = np.linspace(*limits_w2, n_points)

        W1, W2 = np.meshgrid(w1, w2)

        Z_left = f_left(W1, W2)
        Z_right = f_right(W1, W2)

        fig = plt.figure(figsize=(14, 5))

        for i, (f, Z, trajectory, title) in enumerate(
            zip(
                (f_left, f_right),
                (Z_left, Z_right),
                (trajectory_left, trajectory_right),
                titles,
            ),
            start=1,
        ):
            ax = fig.add_subplot(
                1,
                2,
                i,
                projection="3d",
            )

            # Поверхность функционала
            ax.plot_surface(
                W1,
                W2,
                Z,
                cmap=cm.viridis,
                alpha=0.75,
            )

            # Значения функционала вдоль траектории
            trajectory_z = f(
                trajectory[:, 0],
                trajectory[:, 1],
            )

            # Траектория GD
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory_z,
                marker="o",
                markersize=4,
                linewidth=2,
                label=method_name,
            )

            # Оптимум
            ax.scatter(
                w_true[0],
                w_true[1],
                f(w_true[0], w_true[1]),
                marker="*",
                s=150,
                label="$w^*$",
            )

            ax.set_xlim(*limits_w1)
            ax.set_ylim(*limits_w2)

            ax.set_xlabel("$w_1$")
            ax.set_ylabel("$w_2$")
            ax.set_zlabel("$Q(w)$")

            ax.set_title(title)
            ax.legend()

        plt.tight_layout()
        plt.show()

    return (plot_surface_trajectories,)


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad,
    trajectory_good,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good,
        trajectory_bad,
        titles=(
            "GD: хорошая обусловленность",
            "GD: плохая обусловленность",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good,
        trajectory_bad,
        titles=(
            "GD: хорошая обусловленность",
            "GD: плохая обусловленность",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Теперь увеличим шаг и выйдем за пределы условия

    $$
    \eta \le \frac{1}{L}.
    $$

    Возьмём

    $$
    \eta
    =
    \frac{1.98}{L}.
    $$
    """)
    return


@app.cell
def _(L_bad, L_good, X_bad, X_good, gradient_descent, w_init, y_bad, y_good):
    eta_large_good = 1.98 / L_good
    eta_large_bad = 1.98 / L_bad

    trajectory_good_large, loss_good_large = gradient_descent(
        X=X_good,
        y=y_good,
        w_init=w_init,
        eta=eta_large_good,
        n_steps=100,
    )

    trajectory_bad_large, loss_bad_large = gradient_descent(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        eta=eta_large_bad,
        n_steps=100,
    )
    return trajectory_bad_large, trajectory_good_large


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_large,
    trajectory_good_large,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_large,
        trajectory_bad_large,
        titles=(
            r"GD: $\eta = 1.98/L$",
            r"GD: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_large,
        trajectory_bad_large,
        titles=(
            r"GD: $\eta = 1.98/L$",
            r"GD: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Стохастический градиентный спуск

    В обычном градиентном спуске на каждой итерации вычисляется градиент по всей выборке:

    $$
    \nabla Q(w)
    =
    \frac{1}{\ell}
    \sum_{i=1}^{\ell}
    \nabla q_i(w),
    $$

    где

    $$
    q_i(w)
    =
    \frac{1}{2}
    \left(
    \langle w, x_i\rangle-y_i
    \right)^2.
    $$

    Если выборка большая, вычисление полного градиента на каждой итерации может быть дорогим.

    Вместо этого можно случайно выбрать один объект $i_k$ и использовать только его градиент:

    $$
    g_k
    =
    \nabla q_{i_k}(w_k).
    $$

    Тогда шаг принимает вид

    $$
    w_{k+1}
    =
    w_k
    -
    \eta_k g_k.
    $$

    Если объект $i_k$ выбирается равновероятно из выборки, то стохастический градиент является несмещённой оценкой полного градиента:

    $$
    \mathbb{M}
    \left[
    g_k \mid w_k
    \right]
    =
    \nabla Q(w_k).
    $$

    То есть в среднем SGD движется в том же направлении, что и обычный градиентный спуск.

    Однако отдельный стохастический градиент может существенно отличаться от полного градиента, поэтому траектория становится шумной.
    Чтобы обеспечить сходимость SGD и постепенно уменьшать влияние этого шума, шаг уменьшают с номером итерации.

    Для последовательности шагов $\eta_k$ условия Роббинса–Монро имеют вид

    $$
    \sum_{k=0}^{\infty}\eta_k=\infty,
    \qquad
    \sum_{k=0}^{\infty}\eta_k^2<\infty.
    $$

    Первое условие не позволяет шагу уменьшиться слишком быстро, а второе обеспечивает постепенное подавление стохастического шума.

    При стандартных условиях на функционал и дисперсию стохастического градиента выполнение этих условий приводит к сходимости SGD к оптимуму почти наверное.

    Будем использовать убывающий шаг вида

    $$
    \eta_k
    =
    \frac{C}{k+k_0},
    $$

    где $C>0$ задаёт масштаб шага, а $k_0>0$ определяет его начальное значение. Эти величины являются гиперпараметрами алгоритма.

    Такой шаг удовлетворяет условиям Роббинса–Монро.

    В наших экспериментах положим

    $$
    C=1,
    \qquad
    k_0=1,
    $$

    и получим

    $$
    \eta_k
    =
    \frac{1}{k+1}.
    $$
    """)
    return


@app.cell
def _(np, objective):
    def stochastic_gradient_descent(
        X,
        y,
        w_init,
        n_steps=1000,
        random_state=42,
    ):
        rng = np.random.default_rng(random_state)

        w = np.array(w_init, dtype=float)

        trajectory = [w.copy()]
        losses = [objective(w, X, y)]

        ell = len(X)

        for k in range(n_steps):
            # Случайно выбираем один объект
            i = rng.integers(ell)

            x_i = X[i]
            y_i = y[i]

            # Градиент функции потерь одного объекта
            error = x_i @ w - y_i
            grad = error * x_i

            # Гармонически убывающий шаг
            eta = 1.0 / (k + 1)

            # Шаг SGD
            w = w - eta * grad

            trajectory.append(w.copy())
            losses.append(objective(w, X, y))

        return (
            np.array(trajectory),
            np.array(losses),
        )

    return (stochastic_gradient_descent,)


@app.cell
def _(
    RANDOM_STATE,
    X_bad,
    X_good,
    stochastic_gradient_descent,
    w_init,
    y_bad,
    y_good,
):
    trajectory_good_sgd, loss_good_sgd = stochastic_gradient_descent(
        X=X_good,
        y=y_good,
        w_init=w_init,
        n_steps=100,
        random_state=RANDOM_STATE,
    )

    trajectory_bad_sgd, loss_bad_sgd = stochastic_gradient_descent(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        n_steps=10000,
        random_state=RANDOM_STATE,
    )
    return trajectory_bad_sgd, trajectory_good_sgd


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_sgd,
    trajectory_good_sgd,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_sgd,
        trajectory_bad_sgd,
        titles=(
            "SGD: хорошая обусловленность",
            "SGD: плохая обусловленность",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="SGD",
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_sgd,
        trajectory_bad_sgd,
        titles=(
            "SGD: хорошая обусловленность",
            "SGD: плохая обусловленность",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="SGD",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mini-batch градиентный спуск

    В SGD на каждой итерации градиент оценивается по одному случайному объекту. Такая оценка вычисляется дёшево, но может иметь большую дисперсию.

    Вместо одного объекта можно выбрать случайный mini-batch $B_k$ размера $b$ и усреднить градиенты объектов внутри него:

    $$
    g_k
    =
    \frac{1}{b}
    \sum_{i \in B_k}
    \nabla q_i(w_k).
    $$

    Шаг алгоритма остаётся тем же:

    $$
    w_{k+1}
    =
    w_k
    -
    \eta_k g_k.
    $$

    При равномерном случайном выборе объектов mini-batch градиент остаётся несмещённой оценкой полного градиента:

    $$
    \mathbb{M}
    \left[
    g_k \mid w_k
    \right]
    =
    \nabla Q(w_k).
    $$

    При этом усреднение нескольких градиентов уменьшает разброс оценки. Поэтому траектория обычно становится более стабильной, однако одна итерация требует больше вычислений.

    Для сравнения с SGD оставим тот же гармонически убывающий шаг

    $$
    \eta_k
    =
    \frac{1}{k+1}.
    $$
    """)
    return


@app.cell
def _(np, objective):
    def mini_batch_gradient_descent(
        X,
        y,
        w_init,
        batch_size=32,
        n_steps=10_000,
        random_state=42,
    ):
        rng = np.random.default_rng(random_state)

        w = np.array(w_init, dtype=float)

        trajectory = [w.copy()]
        losses = [objective(w, X, y)]

        ell = len(X)

        for k in range(n_steps):
            indices = rng.choice(
                ell,
                size=batch_size,
                replace=False,
            )

            X_batch = X[indices]
            y_batch = y[indices]

            errors = X_batch @ w - y_batch

            grad = (
                X_batch.T @ errors
                / batch_size
            )

            eta = 1.0 / (k + 1)

            w = w - eta * grad

            trajectory.append(w.copy())
            losses.append(
                objective(w, X, y)
            )

        return (
            np.array(trajectory),
            np.array(losses),
        )

    return (mini_batch_gradient_descent,)


@app.cell
def _(
    RANDOM_STATE,
    X_bad,
    X_good,
    mini_batch_gradient_descent,
    w_init,
    y_bad,
    y_good,
):
    batch_size = 32

    trajectory_good_batch, loss_good_batch = mini_batch_gradient_descent(
        X=X_good,
        y=y_good,
        w_init=w_init,
        batch_size=batch_size,
        n_steps=100,
        random_state=RANDOM_STATE,
    )

    trajectory_bad_batch, loss_bad_batch = mini_batch_gradient_descent(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        batch_size=batch_size,
        n_steps=10000,
        random_state=RANDOM_STATE,
    )
    return trajectory_bad_batch, trajectory_good_batch


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_batch,
    trajectory_good_batch,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_batch,
        trajectory_bad_batch,
        titles=(
            "Mini-batch: хорошая обусловленность",
            "Mini-batch: плохая обусловленность",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="Mini-batch",
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_batch,
        trajectory_bad_batch,
        titles=(
            "Mini-batch: хорошая обусловленность",
            "Mini-batch: плохая обусловленность",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="Mini-batch",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Модификации градиентного спуска

    ## Метод импульса (Momentum)

    Одна из проблем градиентного спуска проявляется на плохо обусловленных функциях: кривизна функционала может сильно различаться по разным направлениям.

    В направлении большой кривизны градиент может менять знак от итерации к итерации, из-за чего траектория начинает осциллировать. При этом вдоль направления малой кривизны движение остаётся медленным.

    Идея Momentum состоит в том, чтобы учитывать не только текущий градиент, но и направление движения на предыдущих итерациях.

    Введём экспоненциальное скользящее среднее градиентов:

    $$
    m_k
    =
    \beta m_{k-1}
    +
    (1-\beta)\nabla Q(w_k),
    \qquad
    0 \le \beta < 1.
    $$

    Тогда обновление параметров имеет вид

    $$
    w_{k+1}
    =
    w_k
    -
    \eta m_k.
    $$

    В отличие от обычного градиентного спуска, направление шага теперь определяется не одним текущим градиентом, а историей последних градиентов.

    Если компонента градиента сохраняет направление на нескольких итерациях, её вклад накапливается. Если же компонента постоянно меняет знак, её положительные и отрицательные значения частично компенсируются.

    Поэтому Momentum способен уменьшать осцилляции в направлениях большой кривизны и сохранять движение вдоль направлений, в которых градиент достаточно долго имеет одно и то же направление.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Для обычного градиентного спуска мы использовали шаг

    $$
    \eta=\frac{1.98}{L},
    $$

    для которого была получена теоретическая гарантия сходимости.

    Для градиентного спуска с Momentum допустимый шаг зависит не только от $L$, но и от параметра $\beta$ и геометрии оптимизируемого функционала, поэтому аналогичная универсальная оценка здесь отсутствует.

    Для упрощения эксперимента и сравнения с обычным градиентным спуском будем использовать тот же шаг, а параметр $\beta$ положим равным 0.9.
    """)
    return


@app.cell
def _(np, objective, objective_grad):
    def momentum_gradient_descent(
        X,
        y,
        w_init,
        eta,
        beta=0.9,
        n_steps=1000,
    ):
        w = np.array(w_init, dtype=float)
        m = np.zeros_like(w)

        trajectory = [w.copy()]
        losses = [objective(w, X, y)]

        for _ in range(n_steps):
            grad = objective_grad(w, X, y)

            m = beta * m + (1 - beta) * grad
            w = w - eta * m

            trajectory.append(w.copy())
            losses.append(objective(w, X, y))

        return (
            np.array(trajectory),
            np.array(losses),
        )

    return (momentum_gradient_descent,)


@app.cell
def _(
    L_bad,
    L_good,
    X_bad,
    X_good,
    momentum_gradient_descent,
    w_init,
    y_bad,
    y_good,
):
    beta = 0.9

    eta_momentum_good = 1.98 / L_good
    eta_momentum_bad = 1.98 / L_bad

    n_steps_momentum = 100

    trajectory_good_momentum, loss_good_momentum = momentum_gradient_descent(
        X=X_good,
        y=y_good,
        w_init=w_init,
        eta=eta_momentum_good,
        beta=beta,
        n_steps=n_steps_momentum,
    )

    trajectory_bad_momentum, loss_bad_momentum = momentum_gradient_descent(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        eta=eta_momentum_bad,
        beta=beta,
        n_steps=n_steps_momentum,
    )
    return trajectory_bad_momentum, trajectory_good_momentum


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_momentum,
    trajectory_good_momentum,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_momentum,
        trajectory_bad_momentum,
        titles=(
            r"Momentum: $\eta = 1.98/L$",
            r"Momentum: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="Momentum",
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_momentum,
        trajectory_bad_momentum,
        titles=(
            r"Momentum: $\eta = 1.98/L$",
            r"Momentum: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="Momentum",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Адаптивный шаг (AdaGrad)

    В обычном градиентном спуске один и тот же шаг $\eta$ используется для всех параметров:

    $$
    w_{k+1}
    =
    w_k
    -
    \eta \nabla Q(w_k).
    $$

    Однако разные координаты могут иметь существенно разные масштабы градиента. Это особенно заметно на плохо обусловленных функциях: по одному направлению градиент может быть большим, а по другому — значительно меньшим.

    Идея AdaGrad состоит в том, чтобы подбирать эффективный шаг отдельно для каждой координаты.

    Обозначим

    $$
    g_k
    =
    \nabla Q(w_k).
    $$

    Для каждой координаты будем накапливать квадраты предыдущих градиентов:

    $$
    G_{k,j}
    =
    G_{k-1,j}
    +
    g_{k,j}^2,
    \qquad
    G_{0,j}=0.
    $$

    Тогда обновление параметра $w_j$ имеет вид

    $$
    w_{k+1,j}
    =
    w_{k,j}
    -
    \eta
    \frac{g_{k,j}}
    {\sqrt{G_{k,j}}+\varepsilon},
    $$

    где $\varepsilon>0$ — небольшая константа, предотвращающая деление на ноль.

    Если по некоторой координате градиенты длительное время имеют большой модуль, величина $G_{k,j}$ быстро растёт и эффективный шаг по этой координате уменьшается.

    Напротив, для координат с небольшими градиентами $G_{k,j}$ растёт медленнее, поэтому относительный размер шага остаётся больше.

    Таким образом, AdaGrad автоматически задаёт различный масштаб шага для разных параметров и может частично компенсировать различие масштабов градиента на плохо обусловленных задачах.

    В отличие от обычного градиентного спуска, для AdaGrad нет простого универсального выбора, который давал бы аналогичную гарантию сходимости только через константу Липшица $L$.

    Параметр $\eta$ в AdaGrad задаёт базовый масштаб шага, тогда как фактический шаг по каждой координате дополнительно изменяется накопленной величиной $G_{k,j}$.

    Для сравнения с предыдущими экспериментами будем использовать тот же шаг

    $$
    \eta
    =
    \frac{1.98}{L}.
    $$
    """)
    return


@app.cell
def _(np, objective, objective_grad):
    def adagrad(
        X,
        y,
        w_init,
        eta,
        n_steps=1000,
        eps=1e-8,
    ):
        w = np.array(w_init, dtype=float)

        G = np.zeros_like(w)

        trajectory = [w.copy()]
        losses = [objective(w, X, y)]

        for _ in range(n_steps):
            grad = objective_grad(w, X, y)

            # Накопление квадратов градиентов
            G = G + grad**2

            # Адаптивный шаг для каждой координаты
            w = w - eta * grad / (np.sqrt(G) + eps)

            trajectory.append(w.copy())
            losses.append(objective(w, X, y))

        return (
            np.array(trajectory),
            np.array(losses),
        )

    return (adagrad,)


@app.cell
def _(L_bad, L_good, X_bad, X_good, adagrad, w_init, y_bad, y_good):
    eta_adagrad_good = 1.98 / L_good
    eta_adagrad_bad = 1.98 / L_bad

    n_steps_adagrad = 100

    trajectory_good_adagrad, loss_good_adagrad = adagrad(
        X=X_good,
        y=y_good,
        w_init=w_init,
        eta=eta_adagrad_good,
        n_steps=n_steps_adagrad,
    )

    trajectory_bad_adagrad, loss_bad_adagrad = adagrad(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        eta=eta_adagrad_bad,
        n_steps=n_steps_adagrad,
    )
    return trajectory_bad_adagrad, trajectory_good_adagrad


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_adagrad,
    trajectory_good_adagrad,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_adagrad,
        trajectory_bad_adagrad,
        titles=(
            r"AdaGrad: $\eta = 1.98/L$",
            r"AdaGrad: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="AdaGrad",
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_adagrad,
        trajectory_bad_adagrad,
        titles=(
            r"AdaGrad: $\eta = 1.98/L$",
            r"AdaGrad: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="AdaGrad",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RMSProp

    Недостаток AdaGrad состоит в том, что величина

    $$
    G_{k,j}
    =
    G_{k-1,j}
    +
    g_{k,j}^2
    $$

    накапливает квадраты градиентов за всю историю оптимизации и никогда не уменьшается. Поэтому со временем знаменатель в обновлении может становиться слишком большим, а шаг — слишком маленьким.

    RMSProp заменяет накопленную сумму экспоненциальным скользящим средним квадратов градиента:

    $$
    v_{k,j}
    =
    \beta v_{k-1,j}
    +
    (1-\beta)g_{k,j}^2,
    \qquad
    v_{0,j}=0.
    $$

    Обновление параметров имеет вид

    $$
    w_{k+1,j}
    =
    w_{k,j}
    -
    \eta
    \frac{g_{k,j}}
    {\sqrt{v_{k,j}}+\varepsilon}.
    $$

    В отличие от AdaGrad, старые значения квадратов градиента в RMSProp постепенно забываются. Поэтому величина $v_{k,j}$ характеризует масштаб недавних градиентов, а не всей их истории.

    Если по некоторой координате градиенты имеют большой модуль, знаменатель растёт и шаг по этой координате уменьшается. Для координат с меньшими градиентами масштаб шага остаётся относительно больше.

    Таким образом, RMSProp сохраняет идею адаптивного шага для каждой координаты, но не приводит к постоянному накоплению всей истории, как AdaGrad.

    Для сравнения с предыдущими экспериментами будем использовать тот же базовый шаг

    $$
    \eta
    =
    \frac{1.98}{L},
    $$

    а параметр экспоненциального усреднения положим равным

    $$
    \beta=0.95.
    $$
    """)
    return


@app.cell
def _(np, objective, objective_grad):
    def rmsprop(
        X,
        y,
        w_init,
        eta,
        beta=0.95,
        n_steps=1000,
        eps=1e-8,
    ):
        w = np.array(w_init, dtype=float)
        v = np.zeros_like(w)

        trajectory = [w.copy()]
        losses = [objective(w, X, y)]

        for _ in range(n_steps):
            grad = objective_grad(w, X, y)

            v = beta * v + (1 - beta) * grad**2

            w = w - eta * grad / (np.sqrt(v) + eps)

            trajectory.append(w.copy())
            losses.append(objective(w, X, y))

        return (
            np.array(trajectory),
            np.array(losses),
        )

    return (rmsprop,)


@app.cell
def _(L_bad, L_good, X_bad, X_good, rmsprop, w_init, y_bad, y_good):
    beta_rmsprop = 0.95

    eta_rmsprop_good = 1.98 / L_good
    eta_rmsprop_bad = 1.98 / L_bad

    n_steps_rmsprop = 100

    trajectory_good_rmsprop, loss_good_rmsprop = rmsprop(
        X=X_good,
        y=y_good,
        w_init=w_init,
        eta=eta_rmsprop_good,
        beta=beta_rmsprop,
        n_steps=n_steps_rmsprop,
    )

    trajectory_bad_rmsprop, loss_bad_rmsprop = rmsprop(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        eta=eta_rmsprop_bad,
        beta=beta_rmsprop,
        n_steps=n_steps_rmsprop,
    )
    return (
        beta_rmsprop,
        n_steps_rmsprop,
        trajectory_bad_rmsprop,
        trajectory_good_rmsprop,
    )


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_rmsprop,
    trajectory_good_rmsprop,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_rmsprop,
        trajectory_bad_rmsprop,
        titles=(
            r"RMSProp: $\eta = 1.98/L$",
            r"RMSProp: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="RMSProp",
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_rmsprop,
        trajectory_bad_rmsprop,
        titles=(
            r"RMSProp: $\eta = 1.98/L$",
            r"RMSProp: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="RMSProp",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    На первой итерации память квадратов градиента ещё пуста. Поэтому после вычисления первого градиента

    $$
    v_{0,j}
    =
    (1-\beta)g_{0,j}^2
    $$

    первый шаг по координате приближённо имеет масштаб

    $$
    \left|\Delta w_{0,j}\right|
    \approx
    \frac{\eta}{\sqrt{1-\beta}}.
    $$

    Таким образом, при $\beta$, близком к единице, начальный шаг может оказаться существенно больше базового $\eta$. Поэтому использованный ранее шаг $1.98/L$ для RMSProp оказывается слишком агрессивным.

    В дальнейшем положим

    $$
    \eta=\frac{0.5}{L}.
    $$
    """)
    return


@app.cell
def _(
    L_bad,
    L_good,
    X_bad,
    X_good,
    beta_rmsprop,
    n_steps_rmsprop,
    rmsprop,
    w_init,
    y_bad,
    y_good,
):
    eta_rmsprop_good_1 = 1 / L_good
    eta_rmsprop_bad_1 = 1 / L_bad



    trajectory_good_rmsprop_1, loss_good_rmsprop_1 = rmsprop(
        X=X_good,
        y=y_good,
        w_init=w_init,
        eta=eta_rmsprop_good_1,
        beta=beta_rmsprop,
        n_steps=n_steps_rmsprop,
    )

    trajectory_bad_rmsprop_1, loss_bad_rmsprop_1 = rmsprop(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        eta=eta_rmsprop_bad_1,
        beta=beta_rmsprop,
        n_steps=n_steps_rmsprop,
    )
    return trajectory_bad_rmsprop_1, trajectory_good_rmsprop_1


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_rmsprop_1,
    trajectory_good_rmsprop_1,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_rmsprop_1,
        trajectory_bad_rmsprop_1,
        titles=(
            r"RMSProp: $\eta = 1/L$",
            r"RMSProp: $\eta = 1/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="RMSProp",
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_rmsprop_1,
        trajectory_bad_rmsprop_1,
        titles=(
            r"RMSProp: $\eta = 1/L$",
            r"RMSProp: $\eta = 1/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="RMSProp",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adam

    Adam объединяет две рассмотренные ранее идеи:

    - Momentum сглаживает направление градиента;
    - RMSProp адаптирует масштаб шага отдельно для каждой координаты.

    Обозначим

    $$
    g_k=\nabla Q(w_k).
    $$

    Adam хранит экспоненциальные скользящие средние самого градиента и его квадрата:

    $$
    m_k
    =
    \beta_1 m_{k-1}
    +
    (1-\beta_1)g_k,
    $$

    $$
    v_k
    =
    \beta_2 v_{k-1}
    +
    (1-\beta_2)g_k^2.
    $$

    Величина $m_k$ отвечает за усреднённое направление движения, а $v_k$ — за масштаб недавних градиентов по каждой координате.

    ### Экспоненциальная память

    Раскроем первый момент:

    $$
    m_k
    =
    (1-\beta_1)
    \left(
    g_k
    +
    \beta_1 g_{k-1}
    +
    \beta_1^2 g_{k-2}
    +
    \ldots
    \right).
    $$

    Вклад градиента, полученного $N$ итераций назад, уменьшается в $\beta_1^N$ раз.

    Если хотим определить, через сколько итераций его вес уменьшится до доли $p$ от исходного, положим

    $$
    \beta^N=p.
    $$

    Отсюда

    $$
    N
    =
    \frac{\ln p}{\ln \beta}.
    $$

    Например, при $\beta=0.9$ вклад отдельного градиента уменьшится до $10\%$ примерно через

    $$
    N
    =
    \frac{\ln 0.1}{\ln 0.9}
    \approx
    22
    $$

    итерации.

    Аналогично параметр $\beta_2$ определяет скорость забывания предыдущих квадратов градиента во втором моменте.

    Чем ближе $\beta$ к единице, тем медленнее затухает влияние старой информации.

    ### Коррекция начального смещения

    Оба момента инициализируются нулями, поэтому в начале оптимизации их значения систематически занижены.

    На первой итерации

    $$
    m_0=(1-\beta_1)g_0,
    $$

    $$
    v_0=(1-\beta_2)g_0^2.
    $$

    В частности, как мы видели для RMSProp, малое начальное значение $v_0$ может искусственно уменьшить знаменатель и увеличить первый шаг.

    В Adam используется коррекция начального смещения:

    $$
    \hat m_k
    =
    \frac{m_k}
    {1-\beta_1^{k+1}},
    $$

    $$
    \hat v_k
    =
    \frac{v_k}
    {1-\beta_2^{k+1}}.
    $$

    На первой итерации получаем

    $$
    \hat m_0
    =
    \frac{(1-\beta_1)g_0}
    {1-\beta_1}
    =
    g_0,
    $$

    $$
    \hat v_0
    =
    \frac{(1-\beta_2)g_0^2}
    {1-\beta_2}
    =
    g_0^2.
    $$

    Таким образом, коррекция устраняет смещение, возникающее из-за нулевой инициализации моментов. В частности, во втором моменте исчезает дополнительный множитель

    $$
    \frac{1}{\sqrt{1-\beta_2}},
    $$

    который приводил к увеличению первого шага в RMSProp.

    Итоговое обновление Adam имеет вид

    $$
    w_{k+1}
    =
    w_k
    -
    \eta
    \frac{\hat m_k}
    {\sqrt{\hat v_k}+\varepsilon}.
    $$

    Первый момент сглаживает направление движения, второй адаптирует масштаб шага по координатам, а коррекция начального смещения компенсирует влияние нулевой инициализации моментов.

    Для экспериментального сравнения с предыдущими методами будем использовать тот же шаг

    $$
    \eta
    =
    \frac{1.98}{L}.
    $$

    Параметры экспоненциального усреднения возьмём такими же, как в рассмотренных ранее модификациях:

    $$
    \beta_1=0.9, \qquad \beta_2=0.95.
    $$
    """)
    return


@app.cell
def _(np, objective, objective_grad):
    def adam(
        X,
        y,
        w_init,
        eta,
        beta1=0.9,
        beta2=0.95,
        n_steps=1000,
        eps=1e-8,
    ):
        w = np.array(w_init, dtype=float)

        m = np.zeros_like(w)
        v = np.zeros_like(w)

        trajectory = [w.copy()]
        losses = [objective(w, X, y)]

        for k in range(n_steps):
            grad = objective_grad(w, X, y)

            # Первый момент
            m = beta1 * m + (1 - beta1) * grad

            # Второй момент
            v = beta2 * v + (1 - beta2) * grad**2

            # Коррекция начального смещения
            m_hat = m / (1 - beta1 ** (k + 1))
            v_hat = v / (1 - beta2 ** (k + 1))

            # Шаг Adam
            w = w - eta * m_hat / (np.sqrt(v_hat) + eps)

            trajectory.append(w.copy())
            losses.append(objective(w, X, y))

        return (
            np.array(trajectory),
            np.array(losses),
        )

    return (adam,)


@app.cell
def _(L_bad, L_good, X_bad, X_good, adam, w_init, y_bad, y_good):
    beta1_adam = 0.9
    beta2_adam = 0.95

    eta_adam_good = 1.98 / L_good
    eta_adam_bad = 1.98 / L_bad

    n_steps_adam = 100

    trajectory_good_adam, loss_good_adam = adam(
        X=X_good,
        y=y_good,
        w_init=w_init,
        eta=eta_adam_good,
        beta1=beta1_adam,
        beta2=beta2_adam,
        n_steps=n_steps_adam,
    )

    trajectory_bad_adam, loss_bad_adam = adam(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        eta=eta_adam_bad,
        beta1=beta1_adam,
        beta2=beta2_adam,
        n_steps=n_steps_adam,
    )
    return beta1_adam, beta2_adam, trajectory_bad_adam, trajectory_good_adam


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_adam,
    trajectory_good_adam,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_adam,
        trajectory_bad_adam,
        titles=(
            r"Adam: $\eta = 1.98/L$",
            r"Adam: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="Adam",
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_adam,
        trajectory_bad_adam,
        titles=(
            r"Adam: $\eta = 1.98/L$",
            r"Adam: $\eta = 1.98/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="Adam",
    )
    return


@app.cell
def _(
    L_bad,
    L_good,
    X_bad,
    X_good,
    adam,
    beta1_adam,
    beta2_adam,
    w_init,
    y_bad,
    y_good,
):
    eta_adam_good_1 = 0.1 / L_good
    eta_adam_bad_1 = 0.1 / L_bad

    n_steps_adam_1 = 1000

    trajectory_good_adam_1, loss_good_adam_1 = adam(
        X=X_good,
        y=y_good,
        w_init=w_init,
        eta=eta_adam_good_1,
        beta1=beta1_adam,
        beta2=beta2_adam,
        n_steps=n_steps_adam_1,
    )

    trajectory_bad_adam_1, loss_bad_adam_1 = adam(
        X=X_bad,
        y=y_bad,
        w_init=w_init,
        eta=eta_adam_bad_1,
        beta1=beta1_adam,
        beta2=beta2_adam,
        n_steps=n_steps_adam_1,
    )
    return trajectory_bad_adam_1, trajectory_good_adam_1


@app.cell
def _(
    Q_bad,
    Q_good,
    limits_w1,
    limits_w2,
    plot_contour_trajectories,
    plot_surface_trajectories,
    trajectory_bad_adam_1,
    trajectory_good_adam_1,
):
    plot_surface_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_adam_1,
        trajectory_bad_adam_1,
        titles=(
            r"Adam: $\eta = 0.1/L$",
            r"Adam: $\eta = 0.1/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="Adam",
    )

    plot_contour_trajectories(
        Q_good,
        Q_bad,
        trajectory_good_adam_1,
        trajectory_bad_adam_1,
        titles=(
            r"Adam: $\eta = 0.1/L$",
            r"Adam: $\eta = 0.1/L$",
        ),
        limits_w1=limits_w1,
        limits_w2=limits_w2,
        method_name="Adam",
    )
    return


if __name__ == "__main__":
    app.run()
