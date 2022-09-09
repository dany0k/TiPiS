import math

import numpy as np
import matplotlib.pyplot as plt

V1: int = 1
V2: int = 2
V3: int = 4
V4: int = 8
FI: float = 0
AMPLITUDE: float = 1.0

SCREEN_SIZE: int = 10
GRAPH_W: float = 0.2
GRAPH_H: float = 0.2
BOTTOM_MARGIN: float = 0.78
START_MARGIN: float = 0.04
PADDING: float = 1.2

START_CORDS: int = -2
END_CORDS: int = 2
STEP: float = 0.01


def omega(v: int) -> float:
    return 2 * math.pi * v


def harm_fun(v: int, fi: float, amplitude: float, t: np.ndarray) -> np.ndarray:
    return amplitude * np.cos(omega(v) * t + fi)


def dig_fun(v: int, amplitude: float, t: np.ndarray) -> np.ndarray:
    i: int = 1
    ft_current: np.ndarray = np.sin(0)
    a: np.ndarray = 4 * amplitude / np.pi
    fourier_series_expansions_amount: int = 2000
    while i < fourier_series_expansions_amount:
        ft_current += 1 / i * np.sin(i * omega(v) * t)
        i += 2
    for i in range(t.size):
        if ft_current.data[i] < 0:
            ft_current.data[i] = 0
    return a * ft_current


def draw_graphs(v1: int, v2: int, v3: int, v4: int, fi: float, amplitude: float):
    gr = plt.figure()
    t: np.ndarray = np.arange(START_CORDS, END_CORDS, STEP)
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
    plt.gcf().set_size_inches(SCREEN_SIZE, SCREEN_SIZE)
    plt.show()


def main():
    draw_graphs(V1, V2, V3, V4, FI, AMPLITUDE)


if __name__ == '__main__':
    main()
