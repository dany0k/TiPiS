import math

import numpy as np
import matplotlib.pyplot as plt

V1: int = 1
V2: int = 2
V3: int = 4
V4: int = 8
FI: float = 0
AMPLITUDE: float = 1.0
FOURIER_SERIES_EXPANSIONS_AMOUNT = 2000
N = 40

SCREEN_SIZE: int = 10
GRAPH_W: float = 0.2
GRAPH_H: float = 0.2
BOTTOM_MARGIN: float = 0.78
START_MARGIN: float = 0.04
BAR_THICKNESS: float = 0.1

START_CORDS: int = 0
END_CORDS: int = 2
STEP: float = 0.001
HARMONICS_AMOUNT = 20
HARMONICS_STEP = 1


# Secondary functions #


def a1(t, n, v, amplitude):
    return amplitude * math.cos(2 * math.pi * v * t) * math.cos(n * 2 * math.pi * v * t)


def a2(n, v, amplitude):
    return ((amplitude * (period(v) / 2)) / period(v)) * np.sin(omega_n(v, n) * period(v) / 2) / (omega_n(v, n) * period(v) / 2)


def period(v: int):
    return 1 / v


def q(v):
    return period(v) / (period(v) / 2)


def omega(v: int) -> float:
    return 2 * math.pi * v


def omega_n(v: int, n: int) -> float:
    return 2 * n * math.pi / (period(v) / 2)


# Main functions #


def harm_fun(v: int, fi: float, amplitude: float, t: np.ndarray) -> np.ndarray:
    return amplitude * np.cos(omega(v) * t + fi)


def dig_fun(v: int, amplitude: float, t: np.ndarray) -> np.ndarray:
    ft_current: np.ndarray = np.sin(0)
    a: np.ndarray = 4 * amplitude / np.pi
    i: int = 1
    while i < FOURIER_SERIES_EXPANSIONS_AMOUNT:
        ft_current += 1 / i * np.sin(i * omega(v) * t)
        i += 2
    for i in range(t.size):
        if ft_current.data[i] < 0:
            ft_current.data[i] = 0
    return a * ft_current


# Spectres #

def meander_x(v) -> np.ndarray:
    x: np.ndarray = np.empty(N + 1)
    for i in range(N + 1):
        n = i + 1
        x[i] = n * v
    return x


def meander_y(v, amplitude) -> np.ndarray:
    t = 1 / v
    t_i = t / 2
    x: np.ndarray = np.empty(N + 1)
    y: np.ndarray = np.empty(N + 1)
    i = 0
    while i < N + 1:
        n = i + 1
        x[i] = n * v
        y[i] = (math.pi * amplitude * t_i / t) * np.abs(np.sin(n * omega(v) * t_i / 2) / (n * omega(v) * t_i / 2))
        i += 1
    return y

# Drawing Graphs #


def draw_graphs(v1: int, v2: int, v3: int, v4: int, fi: float, amplitude: float, t: np.ndarray, harm: np.ndarray,
                gr) -> None:
    fun_array = [[harm_fun(v1, fi, amplitude, t), harm_fun(v2, fi, amplitude, t),
                  harm_fun(v3, fi, amplitude, t), harm_fun(v4, fi, amplitude, t)],
                 [v1, v2, v3, v4],
                 [dig_fun(v1, amplitude, t), dig_fun(v2, amplitude, t),
                  dig_fun(v3, amplitude, t), dig_fun(v4, amplitude, t)],
                 [meander_y(v1, amplitude), meander_y(v2, amplitude),
                  meander_y(v3, amplitude), meander_y(v4, amplitude)]]
    harmonics_array = [meander_x(v1), meander_x(v2), meander_x(v3), meander_x(v4)]

    bottom_iteration: float = 0
    right_col_iteration: float = 0
    for i in range(len(fun_array[0])):
        if i > 0:
            bottom_iteration = 0
            right_col_iteration += 0.25
        for j in range(len(fun_array)):
            gr.add_axes([START_MARGIN + right_col_iteration, BOTTOM_MARGIN - bottom_iteration, GRAPH_W, GRAPH_H])
            bottom_iteration += 0.25
            if i == 0 or i == 2:
                plt.plot(t, fun_array[i][j])
            if i == 1:
                temp_arr = [[(fun_array[1][j]), (fun_array[1][j])], [0, amplitude]]
                plt.plot(temp_arr[0], temp_arr[1])
                plt.xlim(0, 15)
            if i == 3:
                plt.bar(harmonics_array[j], fun_array[3][j], 0.2)
                plt.xlim(0, 40)



# Main function #


def main() -> None:
    gr = plt.figure()
    plt.gcf().set_size_inches(SCREEN_SIZE, SCREEN_SIZE)
    t: np.ndarray = np.arange(START_CORDS, END_CORDS, STEP)
    harm: np.ndarray = np.arange(0, N + 1, 1)
    draw_graphs(V1, V2, V3, V4, FI, AMPLITUDE, t, harm, gr)
    plt.show()


if __name__ == '__main__':
    main()
