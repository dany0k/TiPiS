import math

import numpy as np
import matplotlib.pyplot as plt

AMPLITUDE: int = 1
CARRIER_FREQUENCY: int = 80
INFORMATIONAL_FREQUENCY: int = 20
FI: float = 0.0

SCREEN_SIZE: int = 10
GRAPH_W: float = 0.2
GRAPH_H: float = 0.2
BOTTOM_MARGIN: float = 0.78
START_MARGIN: float = 0.04
BAR_THICKNESS: float = 0.1

START_CORDS: int = -200
END_CORDS: int = 200
POINTS_AMOUNT: int = 1000
MODULATION_COEFFICIENT: float = 1.0


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


def amplitude_modulate(amplitude: int, carrier_freq: int, inf_freq: int, t: np.ndarray) -> np.ndarray:
    return informational_signal(amplitude, inf_freq, t) * carrier_signal(amplitude, carrier_freq, t)


def frequency_modulation(amplitude: int, car_freq: int, inf_freq: int, t: np.ndarray):
    t_size = t.size
    ft_inf = informational_signal(amplitude, inf_freq, t)
    ft_end = np.zeros(t_size)
    for i in range(t_size):
        if ft_inf.data[i] == 0:
            ft_end.data[i] = amplitude * np.sin(omega(car_freq) * t.data[i])
        else:
            ft_end.data[i] = amplitude * np.sin(omega(inf_freq) * t.data[i])
    return ft_end


def phase_modulation(amplitude: int, m: float, carrier_freq: int, inf_freq: int, t: np.ndarray):
    carrier_ft = carrier_signal(amplitude, carrier_freq, t)
    inf_ft = informational_signal(amplitude, inf_freq, t)
    res = np.zeros(carrier_ft.size)
    for i in range(len(carrier_ft)):
        res.data[i] = carrier_ft.data[i] + m * inf_ft.data[i]
        if res.data[i] == 0:
            res.data[i] = 0
    return res


def main(amplitude: int = AMPLITUDE, carrier_frequency: int = CARRIER_FREQUENCY,
         informational_frequency: int = INFORMATIONAL_FREQUENCY, modulation_coefficient: float = MODULATION_COEFFICIENT,
         start_cords: int = START_CORDS, end_cords: int = END_CORDS, points_amount: int = POINTS_AMOUNT
         ) -> None:
    t: np.ndarray = np.linspace(start_cords, end_cords, points_amount)
    amp_sig = amplitude_modulate(amplitude, carrier_frequency, informational_frequency, t)
    fm_sig = frequency_modulation(amplitude, carrier_frequency, informational_frequency, t)
    ph_sig = phase_modulation(amplitude, modulation_coefficient, carrier_frequency, informational_frequency, t)

    plt.title('Amplitude')
    plt.plot(t, amp_sig)
    plt.show()

    plt.title('Freq')
    plt.plot(t, fm_sig)
    plt.show()

    plt.title('Phase')
    plt.plot(t, ph_sig)
    plt.show()


if __name__ == '__main__':
    main()
