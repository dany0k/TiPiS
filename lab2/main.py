import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

AMPLITUDE: int = 1
CARRIER_FREQUENCY: int = 5
INFORMATIONAL_FREQUENCY: int = 1
FI: float = 0.0

SCREEN_SIZE: int = 10
GRAPH_W: float = 0.2
GRAPH_H: float = 0.2
BOTTOM_MARGIN: float = 0.78
START_MARGIN: float = 0.04
BAR_THICKNESS: float = 0.1

START_CORDS: int = 0
END_CORDS: int = 1
POINTS_AMOUNT: int = 700
MODULATION_COEFFICIENT: float = 1


def omega(freq: int) -> float:
    return 2 * math.pi * freq


def carrier_signal(amplitude: int, freq: int, t: np.ndarray) -> np.ndarray:
    return amplitude * np.cos(omega(freq) * t)


def informational_signal(inf_freq: int, t: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(omega(inf_freq) * t))


def amplitude_modulate(m: float, amplitude: int, carrier_freq: int, inf_freq: int, points_amount: int) -> np.ndarray:
    t = np.linspace(0, 1, points_amount)
    return amplitude * carrier_signal(amplitude, carrier_freq, t) * (m * informational_signal(inf_freq, t) + amplitude)


# def frequency_modulation(carrier_freq: int, m: int, t: np.ndarray):
#     return np.sin(omega(carrier_freq) * t + m * np.cos(omega(carrier_freq) * t))


def frequency_modulation(amp, f, points):
    max_deviation = 50
    min_deviation = 10
    x = np.linspace(0, 1, points)
    mod_fsk = (np.sign(np.sin(f * x * 2.0 * np.pi)) + amp)
    mod_frq = np.zeros(points)
    for i in range(points):
        if mod_fsk == 0:
            mod_frq[i] = min_deviation
        else:
            mod_frq[i] = max_deviation
    return amp * np.sin(mod_frq * x * 2.0 * np.pi)


def main() -> None:
    t: np.ndarray = np.linspace(0, 1, POINTS_AMOUNT)
    fm_sig = frequency_modulation(AMPLITUDE, CARRIER_FREQUENCY, POINTS_AMOUNT)
    plt.subplot(2, 1, 1)
    plt.plot(fm_sig)
    plt.title('Частотная модуляция')
    plt.xlim([0, (POINTS_AMOUNT) - 1])
    plt.show()


if __name__ == '__main__':
    main()
