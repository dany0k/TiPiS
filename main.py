import math

import numpy
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
BAR_THICKNESS: float = 0.3

START_CORDS: int = 0
END_CORDS: int = 2
STEP: float = 0.001
HARMONICS_AMOUNT = 20
HARMONICS_STEP = 1


# Secondary functions #

def a1(t, n, v, amplitude):
    return amplitude * math.cos(2 * math.pi * v * t) * math.cos(n * 2 * math.pi * v * t)


def a2(n, v, amplitude):
    return np.abs(amplitude * ((numpy.sin(math.pi * math.pi * n * v)) / (math.pi * math.pi * n * v)))


def period(v: int):
    return 1 / v


def omega(v: int) -> float:
    return 2 * math.pi * v


def omega_n(v: int, n: int) -> float:
    return 2 * n * math.pi / period(v)


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


def harm_spectre(amplitude: float, v: int, n: int) -> np.ndarray:
    points_array: np.ndarray = np.empty(n)
    i: int = 0
    while i < n:
        points_array[i] = 2 / period(v) * integrate.quad(a1, - period(v) / 2, period(v) / 2, args=(i, v, amplitude))[0]
        i += 1
    return points_array


def dig_spectre(amplitude: float, v: int, n: int) -> np.ndarray:
    points_array: np.ndarray = np.empty(n)
    for i in range(n):
        if i == 0:
            continue
        points_array[i] = a2(i, v, amplitude)
    return points_array


# Drawing Graphs #


def draw_graphs(v1: int, v2: int, v3: int, v4: int, fi: float, amplitude: float, t: np.ndarray, harm: np.ndarray,
                gr) -> None:
    fun_array = [[harm_fun(v1, fi, amplitude, t), harm_fun(v2, fi, amplitude, t),
                  harm_fun(v3, fi, amplitude, t), harm_fun(v4, fi, amplitude, t)],
                 [harm_spectre(amplitude, v1, len(harm)), harm_spectre(amplitude, v2, len(harm)),
                  harm_spectre(amplitude, v3, len(harm)), harm_spectre(amplitude, v4, len(harm))],
                 [dig_fun(v1, amplitude, t), dig_fun(v2, amplitude, t),
                  dig_fun(v3, amplitude, t), dig_fun(v4, amplitude, t)],
                 [dig_spectre(amplitude, v1, len(harm)), dig_spectre(amplitude, v2, len(harm)),
                  dig_spectre(amplitude, v3, len(harm)), dig_spectre(amplitude, v4, len(harm))]]

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
            if i == 1 or i == 3:
                plt.bar(harm, fun_array[i][j], width=BAR_THICKNESS)


# Main function #


def main() -> None:
    gr = plt.figure()
    plt.gcf().set_size_inches(SCREEN_SIZE, SCREEN_SIZE)
    t: np.ndarray = np.arange(START_CORDS, END_CORDS, STEP)
    harm: np.ndarray = np.arange(START_CORDS, HARMONICS_AMOUNT, HARMONICS_STEP)
    draw_graphs(V1, V2, V3, V4, FI, AMPLITUDE, t, harm, gr)
    plt.show()


if __name__ == '__main__':
    main()
