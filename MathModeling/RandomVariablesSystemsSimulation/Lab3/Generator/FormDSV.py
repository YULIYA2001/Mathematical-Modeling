import random
import numpy as np
import matplotlib.pyplot as plt

import Generator.DSVMethod as NM


def form_values(P, count=10000):
    ''' 
        Формирование дискретных СВ Х и У

        P - матрицa вероятностей двумерной ДСВ
        count - кол-во точек генерации - |X| и |У| 

        return X, Y - массивы псевдослучайных дискретных величин 
    '''
    X, Y = NM.calculate_SV(P, count)

    # график для сгенерированных точек
    plot_SV_dots_3D(X, Y)

    return X, Y


def plot_SV_dots_3D(x, y):
    '''
        График сгенерированных точек

        х, у - массивы псевдослучайных ДСВ Х и У
    '''
    
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter(x, y, marker=".", c="r", s=1)
    ax.set_title("Сгенерированные точки. |X| = |Y| = {:} точек".format(len(x)), fontsize=8)

    fig = plt.gcf()
    fig.canvas.set_window_title('График 1. Генерация точек Х и У двумерной ДСВ')
    plt.show()











def generate_independent_DSV(n):
    ''' Генрация массива P вероятностей СВ из n шт. случайных вероятностей '''
    P = np.zeros(n)

    max = 1
    for i in range(n-1):
        P[i] = random.uniform(0, max)
        max -= P[i]
    P[-1] = 1 - sum(P)

    if sum(P) != 1:
        raise ValueError('Сумма вероятностей должна быть равна 1')

    return P


def generate_independent_2DSV_matrix(pX, pY):
    ''' 
        Генрация матрицы вероятностей двумерной ДСВ из независимых СВ Х и У

        pX, pY - вероятности независимых СВ Х и У

        return P - матрицa вероятностей двумерной ДСВ
    '''
    n, m = len(pX), len(pY)
    P = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            P[i][j] = pX[i] * pY[j]

    if sum(sum(P)) > 1 + 10**(-6) or sum(sum(P)) < 1 - 10**(-6):
        raise ValueError('Сумма вероятностей должна быть равна 1')

    return P





def generate_2DSV_matrix(n, m):
    ''' 
        Генрация матрицы вероятностей двумерной ДСВ из 2 СВ Х и У

        n - кол-во СВ Х 
        m - кол-во СВ У

        return P - матрицa вероятностей двумерной ДСВ
    '''
    P = np.zeros((n,m))

    max = 1
    for i in range(n):
        for j in range(m):
            P[i][j] = random.uniform(0, max)
            max -= P[i][j]

    P[-1][-1] = 1 - sum(sum(P)) + P[-1][-1]

    if sum(sum(P)) != 1:
        raise ValueError('Сумма вероятностей должна быть равна 1')

    return P


def find_one_dimensional_2DSV(P):
    ''' 
        Поиск одномерных рядов вероятностей СВ Х и У

        P - матрицa двумерной СВ

        return pX, pY - одномерные ряды вероятностей СВ Х и У
    '''
    n, m = len(P), len(P[0])

    pX, pY = np.zeros(n), np.zeros(m)
    for i in range(n):
        pX[i] = sum(P[i])

    for j in range(m):
        pY[j] = sum([P[i][j] for i in range(n)])

    if (sum(pX) > 1 + 10**(-6) or sum(pX) < 1 - 10**(-6)) and (sum(pY) > 1 + 10**(-6) or sum(pY) < 1 - 10**(-6)):
        raise ValueError('Сумма вероятностей должна быть равна 1')
    
    return pX, pY
