import numpy as np
import matplotlib.pyplot as plt

SCREEN_SIZE: int = 10


def main():
    # Начальная скорость в км/ч
    start_speed = 60
    # Заданная скорость в км/ч
    set_speed = 90
    # Текущая скорость в км/ч
    current_speed = start_speed
    # Предыдущая на шаг измерения скорость в км/ч
    previous_speed = start_speed
    # Разница между предыдущей и текущей скоростями
    diff = 0
    # Шаг
    dt = 0.1
    # Коэффициенты ПИД-регулятора
    # K_P = 2.2135
    # K_I = 0.5845
    # K_D = 1.7255
    K_P = 1
    K_I = 0
    K_D = 0.7255
    # Ошибка
    current_error = 0
    # Предыдущее значение ошибки
    previous_error = 0

    t = np.arange(0, 200, dt)

    P = 0
    D = 0
    I = 0
    speed = np.zeros(t.size)
    # Изменение скорости
    for i in range(t.size):
        current_error = set_speed - current_speed
        P = K_P * current_error
        I = I + K_I * current_error
        D = K_D * (current_error - previous_error)
        current_speed = current_speed + diff * dt + ((P + I + D) * pow(dt, 2) / 2)
        speed.data[i] = current_speed
        diff = current_speed - previous_speed
        previous_speed = current_speed
        previous_error = current_error

    plt.gcf().set_size_inches(SCREEN_SIZE, SCREEN_SIZE)
    plt.plot(t, speed)
    plt.show()


if __name__ == '__main__':
    main()
