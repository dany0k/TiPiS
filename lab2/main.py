import math

import numpy as np
import matplotlib.pyplot as plt
import scipy

AMPLITUDE: int = 1
CARRIER_FREQUENCY: int = 70
INFORMATIONAL_FREQUENCY: int = 10
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
MODULATION_COEFFICIENT: float = 0.5


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


def frequency_modulation(amplitude: int, inf_freq: int, t: np.ndarray) -> np.ndarray:
    t_size = t.size
    ft_inf = informational_signal(amplitude, inf_freq, t)
    ft_end = np.zeros(t_size)
    max_freq: int = 60
    min_freq: int = 20
    for i in range(t_size):
        if ft_inf.data[i] == 0:
            ft_end.data[i] = amplitude * np.sin(omega(max_freq) * t.data[i])
        else:
            ft_end.data[i] = amplitude * np.sin(omega(min_freq) * t.data[i])
    return ft_end


def phase_modulation(amplitude: int, m: float, car_freq: int, inf_freq: int, t: np.ndarray):
    return amplitude * np.sin(omega(car_freq) * t + m * informational_signal(amplitude, inf_freq, t))


############
# Specters #
############


def any_modulation_spectre(fun: np.ndarray, points_amount: int) -> np.ndarray:
    spectre: np.ndarray = np.abs(scipy.fft.fft(fun)) / (points_amount / 2)
    return spectre


##########
# Filter #
##########


def filter_signal(demod_spec: np.ndarray, amplitude, inf_freq: np.ndarray, t: np.ndarray):
    filtered_signal: np.ndarray = np.copy(np.real(demod_spec))
    inf_sig: np.ndarray = informational_signal(amplitude, inf_freq, t)
    for i in range(filtered_signal.size):
        if filtered_signal.data[i] < inf_sig.data[i]:
            filtered_signal.data[i] = 1
        if filtered_signal.data[i] > inf_sig.data[i]:
            filtered_signal.data[i] = 0
    return filtered_signal


def cut_high_and_low_frequencies(amp_mod_spectre: np.ndarray, points_amount: int) -> np.ndarray:
    cut_sign: np.ndarray = np.copy(amp_mod_spectre)
    frequency: np.ndarray = np.fft.fftfreq(len(amp_mod_spectre), 1.0 / points_amount)
    min_freq: int = 10
    max_freq: int = 50
    for i in range(amp_mod_spectre.size):
        if frequency.data[i] < min_freq or frequency.data[i] > max_freq:
            cut_sign.data[i] = 0
    return cut_sign


def demodulate_spectre(cut_sign: np.ndarray) -> np.ndarray:
    demodulated_sig: np.ndarray = scipy.fft.ifft(cut_sign)
    return demodulated_sig


def draw_graphs(gr, t: np.ndarray,
                amp_mod_sig: np.ndarray, fr_mod_sig: np.ndarray, ph_mod_sig: np.ndarray,
                amp_mod_spectre: np.ndarray, fr_mod_spectre: np.ndarray, ph_mod_spectre: np.ndarray,
                demod_signal: np.ndarray, filtered_signal: np.ndarray, informational_signal: np.ndarray) -> None:
    fun_array = [
        [amp_mod_sig, fr_mod_sig, ph_mod_sig],
        [amp_mod_spectre, fr_mod_spectre, ph_mod_spectre],
        [demod_signal, filtered_signal, informational_signal]
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
    plt.plot(t, fun_array[1][0])

    gr.add_axes([START_MARGIN + start_offset, BOTTOM_MARGIN - bottom_offset, GRAPH_W, GRAPH_H])
    plt.xlim(left_limit, right_limit)
    plt.plot(t, fun_array[1][1])

    gr.add_axes([START_MARGIN + start_offset * 2, BOTTOM_MARGIN - bottom_offset, GRAPH_W, GRAPH_H])
    plt.xlim(left_limit, right_limit)
    plt.plot(t, fun_array[1][2])

    gr.add_axes([START_MARGIN, BOTTOM_MARGIN - bottom_offset * 2, GRAPH_W, GRAPH_H])
    plt.plot(t, fun_array[2][0])

    gr.add_axes([START_MARGIN + start_offset, BOTTOM_MARGIN - bottom_offset * 2, GRAPH_W, GRAPH_H])
    plt.plot(t, fun_array[2][1])

    gr.add_axes([START_MARGIN + start_offset * 2, BOTTOM_MARGIN - bottom_offset * 2, GRAPH_W, GRAPH_H])
    plt.plot(t, fun_array[2][2])


def main(amplitude: int = AMPLITUDE, carrier_frequency: int = CARRIER_FREQUENCY,
         informational_frequency: int = INFORMATIONAL_FREQUENCY, modulation_coefficient: float = MODULATION_COEFFICIENT,
         start_cords: int = START_CORDS, end_cords: int = END_CORDS, points_amount: int = POINTS_AMOUNT
         ) -> None:
    t: np.ndarray = np.linspace(start_cords, end_cords, points_amount)
    inf_sig = informational_signal(amplitude, informational_frequency, t)
    amp_mod_sig: np.ndarray = amplitude_modulate(amplitude, carrier_frequency, informational_frequency, t)
    fr_mod_sig: np.ndarray = frequency_modulation(amplitude, informational_frequency, t)
    ph_mod_sig: np.ndarray = phase_modulation(amplitude, modulation_coefficient, carrier_frequency,
                                              informational_frequency, t)
    amp_mod_spectre: np.ndarray = any_modulation_spectre(amp_mod_sig, points_amount)
    fr_mod_spectre: np.ndarray = any_modulation_spectre(fr_mod_sig, points_amount)
    ph_mod_spectre: np.ndarray = any_modulation_spectre(ph_mod_sig, points_amount)
    cut_signal: np.ndarray = cut_high_and_low_frequencies(amp_mod_spectre, points_amount)
    demodulated_signal: np.ndarray = demodulate_spectre(cut_signal)
    filtered_signal: np.ndarray = filter_signal(demodulated_signal, amplitude, informational_frequency, t)
    gr = plt.figure()
    plt.gcf().set_size_inches(SCREEN_SIZE, SCREEN_SIZE)
    draw_graphs(gr, t, amp_mod_sig, fr_mod_sig, ph_mod_sig, amp_mod_spectre, fr_mod_spectre, ph_mod_spectre,
                demodulated_signal, filtered_signal, inf_sig)
    plt.show()


if __name__ == '__main__':
    main()
