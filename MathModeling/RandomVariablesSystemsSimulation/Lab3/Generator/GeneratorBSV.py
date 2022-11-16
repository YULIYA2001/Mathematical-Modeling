from random import randrange
import numpy as np
import math
import matplotlib.pyplot as plt


# Мультипликативный конгруэнтный датчик базовой случайной величины(БСВ)
def multiplicative_congruential_method(Ao=0, m=2**35, k=5**16, count=10**6):
    ''' Ao - начальное значение, случайное число [0..10^6), прописано ниже из-за псевдослучайнсти
        m - модуль, k - множитель (начальные значения из языка моделирования СИМУЛА)
        count - количество чисел в последовательности (генератора)
        результат - генератор с соunt псевдослучайных чисел '''
    Ao = randrange(10**6)
    Ai = Ao
    for i in range(count):
        Ai = (k * Ai) % m
        yield Ai/m


# Тестирование равномерности датчика БСВ
def uniformity_testing(k=10, n=100):
    ''' k - количество отрезков для разбивания интервала (0, 1)
        n - количество генерируемых случайных величин
        результат - гистограмма относительных частот попадания в каждый интервал '''
    rand_nums_generator = multiplicative_congruential_method(count=n)

    intervals = np.arange(1/k, 1 + 1/k, 1/k)
    count_for_interval = np.zeros(len(intervals))
    sorted_nums = sorted(rand_nums_generator)
        
    # частоты попадания случайных чисел в каждый интервал
    i = 0
    for number in sorted_nums:
        if number <= intervals[i]:
            count_for_interval[i] += 1
        else:
            count_for_interval[i+1] += 1
            i += 1

    # относительные частоты попаданий
    for i in range(len(count_for_interval)):
        count_for_interval[i] /= n
    print('Oтносительные частоты попаданий', count_for_interval)

    # математическое ожидание 
    M = 1/n * sum(sorted_nums)
    print('M = {:.6f} (\u0394 = {:.6f})'.format( M, abs(1.0/2 - M) ))

    # дисперсия
    D = 1/(n-1) * sum( [(num - M)** 2 for num in sorted_nums])
    print('D = {:.6f} (\u0394 = {:.6f})'.format( D, abs(1.0/12 - D) ))

    # гистограмма частот
    plt.figure(figsize=(5, 3))
    plt.title('k = {:}   n = {:}'.format(k, n))
    plt.bar([el - 1/k/2 for el in intervals], count_for_interval, width=1/k, 
            color='green', ec='black')
    plt.show()



def test_generator():
    # Тестирование равномерности датчика БСВ
    uniformity_testing(k=10, n=100)
    uniformity_testing(k=10, n=10000)
    uniformity_testing(k=20, n=10000)
    uniformity_testing(k=20, n=1000000)
