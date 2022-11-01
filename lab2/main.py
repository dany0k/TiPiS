import math

import numpy as np
import matplotlib.pyplot as plt
import scipy

AMPLITUDE: int = 1
CARRIER_FREQUENCY: int = 80
INFORMATIONAL_FREQUENCY: int = 20
FI: float = 0.0

SCREEN_SIZE: int = 10
GRAPH_W: float = 0.3
GRAPH_H: float = 0.3
BOTTOM_MARGIN: float = 0.68
START_MARGIN: float = 0.03
BAR_THICKNESS: float = 0.1

START_CORDS: int = 0
END_CORDS: int = 400
POINTS_AMOUNT: int = 1000
MODULATION_COEFFICIENT: float = 1.0


######################
# Secondary functions#
######################
def omega(freq: int) -> float:
    return 2 * math.pi * freq


def carrier_signal(amplitude: int, freq: int, t: np.ndarray) -> np.ndarray:
    return amplitude * np.sin(omega(freq) * t)


def informational_signal(amplitude: int, inf_freq: int, t: np.ndarray) -> np.ndarray:
    meandr_sig: np.ndarray = amplitude * np.sign(np.sin(omega(inf_freq) * t))
    for i in range(meandr_sig.size):
        if meandr_sig.data[i] == -1:
            meandr_sig.data[i] = 0
    return meandr_sig


###############
# Modulations #
###############


def amplitude_modulate(amplitude: int, carrier_freq: int, inf_freq: int, t: np.ndarray) -> np.ndarray:
    return informational_signal(amplitude, inf_freq, t) * carrier_signal(amplitude, carrier_freq, t)


def frequency_modulation(amplitude: int, car_freq: int, inf_freq: int, t: np.ndarray) -> np.ndarray:
    t_size = t.size
    ft_inf = informational_signal(amplitude, inf_freq, t)
    ft_end = np.zeros(t_size)
    for i in range(t_size):
        if ft_inf.data[i] == 0:
            ft_end.data[i] = amplitude * np.sin(omega(car_freq) * t.data[i])
        else:
            ft_end.data[i] = amplitude * np.sin(omega(inf_freq) * t.data[i])
    return ft_end


def phase_modulation(amplitude: int, m: float, carrier_freq: int, inf_freq: int, t: np.ndarray) -> np.ndarray:
    carrier_ft = carrier_signal(amplitude, carrier_freq, t)
    inf_ft = informational_signal(amplitude, inf_freq, t)
    res = np.zeros(carrier_ft.size)
    for i in range(len(carrier_ft)):
        res.data[i] = carrier_ft.data[i] + m * inf_ft.data[i]
        if res.data[i] == 0:
            res.data[i] = 0
    return res


############
# Specters #
############


def any_modulation_spectre(fun: np.ndarray, points_amount: int) -> np.ndarray:
    t: np.ndarray = np.linspace(START_CORDS, END_CORDS, points_amount)
    spectre: np.ndarray = np.abs(scipy.fft.fft(fun)) / (points_amount / 2)
    error_point = 0.03
    for i in range(points_amount):
        if i > points_amount / 2 or spectre[i] < error_point:
            spectre[i] = 0
    return spectre


def draw_graphs(gr, t: np.ndarray,
                amp_mod_sig: np.ndarray, fr_mod_sig: np.ndarray, ph_mod_sig: np.ndarray,
                amp_mod_spectre: np.ndarray, fr_mod_spectre: np.ndarray, ph_mod_spectre: np.ndarray) -> None:
    fun_array = [
        [amp_mod_sig, fr_mod_sig, ph_mod_sig],
        [amp_mod_spectre, fr_mod_spectre, ph_mod_spectre]
    ]

    start_offset: float = 0.33
    bottom_offset: float = 0.33
    left_limit: int = START_CORDS - 5
    right_limit: float = END_CORDS / 4
    gr.add_axes([START_MARGIN, BOTTOM_MARGIN, GRAPH_W, GRAPH_H])
    plt.plot(t, fun_array[0][0])
    gr.add_axes([START_MARGIN + start_offset, BOTTOM_MARGIN, GRAPH_W, GRAPH_H])
    plt.plot(t, fun_array[0][1])
    gr.add_axes([START_MARGIN + start_offset * 2, BOTTOM_MARGIN, GRAPH_W, GRAPH_H])
    plt.plot(t, fun_array[0][2])
    gr.add_axes([START_MARGIN, BOTTOM_MARGIN - bottom_offset, GRAPH_W, GRAPH_H])
    plt.xlim(left_limit, right_limit)
    plt.bar(t, fun_array[1][0])
    gr.add_axes([START_MARGIN + start_offset, BOTTOM_MARGIN - bottom_offset, GRAPH_W, GRAPH_H])
    plt.xlim(left_limit, right_limit)
    plt.bar(t, fun_array[1][1])
    gr.add_axes([START_MARGIN + start_offset * 2, BOTTOM_MARGIN - bottom_offset, GRAPH_W, GRAPH_H])
    plt.xlim(left_limit, right_limit)
    plt.bar(t, fun_array[1][2])


def main(amplitude: int = AMPLITUDE, carrier_frequency: int = CARRIER_FREQUENCY,
         informational_frequency: int = INFORMATIONAL_FREQUENCY, modulation_coefficient: float = MODULATION_COEFFICIENT,
         start_cords: int = START_CORDS, end_cords: int = END_CORDS, points_amount: int = POINTS_AMOUNT
         ) -> None:
    t: np.ndarray = np.linspace(start_cords, end_cords, points_amount)
    amp_mod_sig: np.ndarray = amplitude_modulate(amplitude, carrier_frequency, informational_frequency, t)
    fr_mod_sig: np.ndarray = frequency_modulation(amplitude, carrier_frequency, informational_frequency, t)
    ph_mod_sig: np.ndarray = phase_modulation(amplitude, modulation_coefficient, carrier_frequency, informational_frequency, t)
    amp_mod_spectre: np.ndarray = any_modulation_spectre(amp_mod_sig, points_amount)
    fr_mod_spectre: np.ndarray = any_modulation_spectre(fr_mod_sig, points_amount)
    ph_mod_spectre: np.ndarray = any_modulation_spectre(ph_mod_sig, points_amount)

    gr = plt.figure()
    plt.gcf().set_size_inches(SCREEN_SIZE, SCREEN_SIZE)
    draw_graphs(gr, t, amp_mod_sig, fr_mod_sig, ph_mod_sig, amp_mod_spectre, fr_mod_spectre, ph_mod_spectre)
    plt.show()


if __name__ == '__main__':
    main()
