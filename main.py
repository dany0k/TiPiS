import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

V1: int = 1
V2: int = 2
V3: int = 4
V4: int = 8
FI: float = 0
AMPLITUDE: float = 1.0
FOURIER_SERIES_EXPANSIONS_AMOUNT = 2000

SCREEN_SIZE: int = 10
GRAPH_W: float = 0.2
GRAPH_H: float = 0.2
BOTTOM_MARGIN: float = 0.78
START_MARGIN: float = 0.04
PADDING: float = 1.2
BAR_THICKNESS: float = 0.01

START_CORDS: int = -2
END_CORDS: int = 2
STEP: float = 0.001


def a1(t, n, v, amplitude):
    return amplitude * math.cos(2 * math.pi * v * t) * math.cos(n * 2 * math.pi * v * t)


def a2(t, n, v, amplitude):
    i: int = 1
    ft_current: float = 0
    while i < FOURIER_SERIES_EXPANSIONS_AMOUNT:
        ft_current += 1 / i * np.sin(i * omega(v) * t)
        i += 2
    return amplitude * math.cos(n * 2 * math.pi * v * t) * ft_current


def period(v: int):
    return 1 / v


def omega(v: int) -> float:
    return 2 * math.pi * v


def omega_n(v: int, n: int) -> float:
    return 2 * n * math.pi / (1 / v)


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


def dig_spectre(amplitude: float, v: int, n: int) -> np.ndarray:
    points_array: np.ndarray = np.empty(n)
    i: int = 0
    while i < n:
        points_array[i] = 2 / period(v) * integrate.quad(a2, - period(v) / 2, period(v) / 2, args=(i, v, amplitude))[0]
        i += 1
    return points_array


def harm_spectre(amplitude: float, v: int, n: int) -> np.ndarray:
    points_array: np.ndarray = np.empty(n)
    i: int = 0
    while i < n:
        points_array[i] = 2 / period(v) * integrate.quad(a1, - period(v) / 2, period(v) / 2, args=(i, v, amplitude))[0]
        i += 1
    return points_array


def draw_graphs(v1: int, v2: int, v3: int, v4: int, fi: float, amplitude: float, t: np.ndarray, gr) -> None:
    fun_array: list = [harm_fun(v1, fi, amplitude, t), harm_fun(v2, fi, amplitude, t),
                       harm_fun(v3, fi, amplitude, t), harm_fun(v4, fi, amplitude, t),
                       dig_fun(v1, amplitude, t), dig_fun(v2, amplitude, t),
                       dig_fun(v3, amplitude, t), dig_fun(v4, amplitude, t)]
    left_col_iteration: float = 0
    right_col_iteration: float = 0
    arr_len: int = len(fun_array)
    for i in range(arr_len):
        if i < arr_len / 2:
            gr.add_axes([START_MARGIN, BOTTOM_MARGIN - left_col_iteration, GRAPH_W, GRAPH_H])
            left_col_iteration += 0.25
        else:
            gr.add_axes([START_MARGIN + GRAPH_H * PADDING, BOTTOM_MARGIN - right_col_iteration, GRAPH_W, GRAPH_H])
            right_col_iteration += 0.25
        plt.plot(t, fun_array[i])


def draw_specters(v1: int, v2: int, v3: int, v4: int, amplitude: float, gr):
    n: np.ndarray = np.arange(0, 20, 1)
    na: list = [harm_spectre(amplitude, v1, len(n)), harm_spectre(amplitude, v2, len(n)),
                harm_spectre(amplitude, v3, len(n)), harm_spectre(amplitude, v4, len(n))]
    left_col_iteration: float = 0
    right_col_iteration: float = 0
    na_len: int = len(na)
    for i in range(na_len):
        plt.bar(n[i], na[i], width=BAR_THICKNESS)


def draw_dig_specters(v1: int, v2: int, v3: int, v4: int, amplitude: float, gr):
    n: np.ndarray = np.arange(0, 20, 1)
    na: list = [dig_spectre(amplitude, v1, len(n)), dig_spectre(amplitude, v2, len(n)),
                dig_spectre(amplitude, v3, len(n)), dig_spectre(amplitude, v4, len(n))]
    na_len: int = len(na)
    for i in range(na_len):
        plt.bar(n[i], na[i], width=BAR_THICKNESS)
    return


def main() -> None:
    gr = plt.figure()
    plt.gcf().set_size_inches(SCREEN_SIZE, SCREEN_SIZE)
    t: np.ndarray = np.arange(START_CORDS, END_CORDS, STEP)
    # draw_graphs(V1, V2, V3, V4, FI, AMPLITUDE, t, gr)
    # draw_specters(V1, V2, V3, V4, AMPLITUDE, gr)
    draw_dig_specters(V1, V2, V3, V4, AMPLITUDE, gr)
    plt.show()


if __name__ == '__main__':
    main()
