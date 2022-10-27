import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy.fftpack import rfft

# частоты
message_freq = 10
carrier_freq = 70
points = 800
t = np.linspace(0, 1, points)


# демодуляция (амплитудная)
def demodulate(received_array, fc, t):
    c = np.sin(2 * np.pi * fc * t)

    demod = c * received_array
    return demod


def filter_signal(cutted_signal):
    # создание однополярного меандра
    message = np.sign(np.sin((2 * np.pi * t) * message_freq))
    for i in range(len(message)):
        if message[i] < 0:
            message[i] = 0

    demod_cut_signal = demodulate(cutted_signal, carrier_freq, len(cutted_signal))

    # получение точек в меандре, где график растет\падает
    num = points / (message_freq * 2)  # для 5гц = 80; для 10гц = 40 значение, являющееся длиной
    # одного "прямоугольника" на графике по оси Х
    j_0 = num / points  # для 5гц = 0.1; для 10гц = 0.05 #получение коэффициента, на который будет умножаться points
    j = j_0
    array_meandr = []
    for i in range(message_freq * 2):
        granica = points * j
        array_meandr.append(granica)
        j += j_0

    freqs = np.fft.fftfreq(len(demod_cut_signal), 1.0 / points)  # массив из значений по оси Х
    old_granica = 0  # начальная граница
    n = np.arange(0, points)
    freqs = n  # добавление в массив точек из points (из оси Х)

    count = 0
    # Исходя из информации, полученной из массива array_meandr, изменяем сигнал, согласно точкам в массиве
    for i in array_meandr:
        if count % 2 == 0:
            demod_cut_signal[freqs > round(old_granica)] = 1
            demod_cut_signal[freqs > round(i)] = 0
        else:
            demod_cut_signal[freqs > round(old_granica)] = 0
            demod_cut_signal[freqs > round(i)] = 1
        old_granica = i
        count += 1

    plt.title("Модулирующий сигнал (Однополярный меандр)")
    plt.plot(message)
    plt.xlim([0, points])
    plt.show()

    plt.title("Окончательный сигнал")
    plt.plot(demod_cut_signal)
    plt.xlim([0, points])
    plt.show()


# amp - амплитуда сигнала
# km - коэффициент модуляции
# fc - частота несущего сигнала
# f - частота сигнала
# points - число точек для отрисовки

def amplitude_modulation(amp, km, fc, f, points):
    x = np.linspace(0, 1, points)
    return amp * (km * (np.sign(np.sin(f * x * 2.0 * np.pi)) + amp)) * np.sin(fc * x * 2.0 * np.pi)


def frequency_modulation(amp, f, points):
    x = np.linspace(0, 1, points)
    mod_fsk = (np.sign(np.sin(f * x * 2.0 * np.pi)) + amp)
    mod_frq = np.zeros(points)
    mod_frq[mod_fsk == 0] = 10  # минимальное значение - 10
    mod_frq[mod_fsk == 2] = 50  # максимальное значение - 50

    return amp * np.sin(mod_frq * x * 2.0 * np.pi)


def phase_modulation(amp=1.0, kd=0.25, fc=10.0, f=2.0, points=100):
    x = np.linspace(0, 1, points)
    return amp * np.sin(fc * x * 2.0 * np.pi + kd * amp * (np.sign(np.sin(f * x * 2.0 * np.pi)) + amp))


if __name__ == "__main__":
    # графики амплитудной модуляции
    am_sig = amplitude_modulation(1.0, 0.45, 70, 10, points)  # несущая частота - 70 Гц
    plt.subplot(2, 1, 1)
    plt.plot(am_sig)
    plt.title('Амплитудная модуляция')
    plt.xlim([0, (points) - 1])

    plt.subplot(2, 1, 2)
    spectr_sig_am = np.abs(scipy.fft.fft(am_sig)) / (points * 0.5)

    plt.plot(spectr_sig_am)
    plt.xlim([0, 400])  # ограничение. Отсекается ненужная правая часть
    plt.tight_layout()
    plt.show()

    # графики частотной модуляции
    fm_sig = frequency_modulation(1.0, 5, points)
    plt.subplot(2, 1, 1)
    plt.plot(fm_sig)
    plt.title('Частотная модуляция')
    plt.xlim([0, (points) - 1])

    plt.subplot(2, 1, 2)

    spectr_sig_fm = np.abs(scipy.fft.fft(fm_sig)) / (points * 0.5)
    plt.plot(spectr_sig_fm)
    plt.xlim([0, 400])  # ограничение. Отсекается ненужная правая часть
    plt.tight_layout()
    plt.show()

    # графики фазовой модуляции
    pm_sig = phase_modulation(1.0, 9, 30, 15, points)  # несущая частота - 30 Гц
    plt.subplot(2, 1, 1)
    plt.plot(pm_sig)
    plt.title('Фазовая модуляция')
    plt.xlim([0, (points) - 1])

    plt.subplot(2, 1, 2)

    spectr_sig_pm = np.abs(scipy.fft.fft(pm_sig)) / (points * 0.5)
    plt.plot(spectr_sig_pm)
    plt.xlim([0, 400])  # ограничение. Отсекается ненужная правая часть
    plt.tight_layout()
    plt.show()

    # синтезированный сигнал
    plt.title('Синтез сигнала амплитудной модуляции')
    freq = np.fft.fftfreq(len(am_sig), 1.0 / points)
    cutted_spec_signal = spectr_sig_am.copy()
    cutted_spec_signal[(freq < 60)] = 0  # обрезание низких частот
    cutted_spec_signal[(freq > 80)] = 0  # обрезание высоких частот
    cutted_signal = scipy.fftpack.irfft(cutted_spec_signal)

    plt.plot(cutted_signal)
    plt.xlim([0, points])
    plt.tight_layout()
    plt.show()

    # фильтрация
    filter_signal(cutted_signal)
