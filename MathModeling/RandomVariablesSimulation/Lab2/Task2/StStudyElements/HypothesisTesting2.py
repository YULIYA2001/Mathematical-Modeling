import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.ticker as ticker

from sympy import diff, symbols, solve, simplify, lambdify


import Task2.FormDiscreteSV as DSV
import Task2.StStudyElements.Histograms2 as Histograms
import Task2.StatisticalStudy2 as StatisticalStudy



def test(G, p, f, n=0, kind='A'):
    '''
        G - функция распределения
        p - вероятность успеха испытания
        n - размер выборки (>40 !!!)
    '''
    params = []

    if n == 0:
        if kind == 'P':
            params = Pirson_criterion(p, f, G)
        elif kind == 'K':
            params = Kolmogorov_criterion(p, f, G)
        else:
            Pirson_criterion(p, f, G)
            Kolmogorov_criterion(p, f, G)
    else:
        if kind == 'P':
            params = Pirson_criterion(p, f, G, n)
        elif kind == 'K':
            params = Kolmogorov_criterion(p, f, G, n)
        else:
            Pirson_criterion(p, f, G, n)
            Kolmogorov_criterion(p, f, G, n)

    return params


#--------------------------------------------- Критерий Пирсона ------------------------------------------------
def plot_hist(A, B, F, theor_f):
    """Построение гистограммы и аналитической функции для функции плотности"""

    # НАСТРОЙКИ ГРАФИКА
    fig, ax = plt.subplots()
    # цена деления шкалы
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # сетка
    ax.grid()   
    plt.grid(which='minor', linestyle=':')

    # теоритическая функция плотности распределения
    Ay, By = A[0], B[-1]
    x = [i for i in range(Ay, By+1, 1)]
    y = [theor_f(i) for i in x]
    ax.step(x, y, "k-", where='post', label = "Аналитическая функция", linewidth=1)

    # гистограмма
    x_pl, y_pl = A.copy(), F.copy()
    x_pl.append(B[-1])
    y_pl.append(y_pl[-1])
    ax.step(x_pl, y_pl, "r--", where='post', label = "Гистограмма", linewidth=1)

    #Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('Критерий Хи2 - Пирсона. Функция плотности')
    ax.legend()
    ax.grid()
    plt.show()



def X_square(A, B, M, n, v, G):
    """Расчет значения критерия хи-квадрат"""

    Ax = A.copy()
    Ax.append(B[-1])
    Ax.insert(0, 0)
    X_sq, sum_pi = 0, 0
    for i in range(M):
        pi = G(Ax[i+1]) - G(Ax[i])   #f(A[i], p)
        pi_star = v[i]/n
        X_sq += n * (pi - pi_star)**2 / pi
        sum_pi += pi

    good = False
    if abs(1-sum_pi) <= 0.01:
        print("\nКонтрольное соотношение для суммы Pi выполняется")
        print(abs(1-sum_pi), "<= 0.01\n")
        good = True
        
    return X_sq, good, sum_pi


def Pirson_criterion(p, f, G, n=200):
    """
            Проверка гипотезы по критерию согласия Пирсона
        p - вероятность успеха испытания
        f - Функция вероятности - Геометрическое распределение
        G - исходная функция распределения
        n - размер выборки (>40 !!!)
    """

    # менять выборку, если не выполняется контрольное соотношенеие E(Pi)~1
    while True:
        Y = DSV.calculate_Y(p, n, f)[0]
        M = Y[-1] - 1
        A, B, F, v, h = Histograms.coef_ABFv_eq_interval(Y, M)
        A.pop()
        B.pop()
        X_sq, good, sum_pi = X_square(A, B, M, n, v, lambda n: G(n, p))
        if good:
            break

    plot_hist(A, B, F, lambda n: f(n, p))

    s = 1       # s - колво параметров в функции распределения
    k = M - s - 1
    # k значения Хи для уровня значимости-а из таблицы Хи-квадрат распределения
    X_tb = [scipy.stats.chi2.ppf(0.90, k), scipy.stats.chi2.ppf(0.95, k), scipy.stats.chi2.ppf(0.99, k)]
    a = [0.1, 0.05, 0.01]
    for i in range(3):
        print('a = ', a[i], '\n',
              'X^2 = {:.4f}'.format(X_sq), 
              '>' if X_sq > X_tb[i] else '<',
              X_tb[i],
              ' => гипотеза отклоняется' if X_sq > X_tb[i] else ' => гипотеза принимается')

    return 'P', sum_pi, a, X_sq, X_tb







#------------------------------------------- Критерий Колмогорова ----------------------------------------------

def plot_empirical_and_analytical_function(y, Fy, G):

    n_max = y[-1]

    # НАСТРОЙКИ ГРАФИКА
    fig, ax = plt.subplots()
    # цена деления шкалы
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # диапазоны шкалы
    plt.xlim([0, n_max+0.4])
    plt.ylim([-0.01, 1.01])
    # сетка
    ax.grid()   
    plt.grid(which='minor', linestyle=':')

    # ГРАФИК АНАЛИТИЧЕСКИЙ
    x_pl_t = [i for i in range(n_max+1)]
    y_pl_t = [G(i) for i in range(n_max+1)]
    x_pl_t.append(x_pl_t[-1] + 1)
    y_pl_t.append(y_pl_t[-1])
    ax.step(x_pl_t, y_pl_t, "b-", where='post', linewidth = 0.8, label = "Аналитическая функция")
    # не выколотые точки графика
    x_pl_t.pop(0)
    y_pl_t.pop()
    ax.scatter(x_pl_t, y_pl_t, s=8, c="b", marker="o", edgecolors="b", linewidth = 0.5)
    # выколотые точки графика
    y_pl_t.pop(0)
    x_pl_t.pop()
    ax.scatter(x_pl_t, y_pl_t, s=12, c="w", marker="o", edgecolors="b", linewidth = 0.5)

    #ГРАФИК ПРАКТИЧЕСКИЙ
    x_pl, y_pl = y.copy(), Fy.copy() 
    x_pl.insert(0, 0)
    y_pl.insert(0, 0)
    x_pl.append(x_pl[-1]+1)
    y_pl.append(y_pl[-1])
    ax.step(x_pl, y_pl, "r--", where='post', linewidth = 0.8, label = "Эмпирическая функция")
    # не выколотые точки графика
    x_pl.pop(0)
    y_pl.pop()
    ax.scatter(x_pl, y_pl, s=8, c="g", marker="o", edgecolors="r", linewidth = 0.5)
    # выколотые точки графика
    y_pl.pop(0)
    x_pl.pop()
    ax.scatter(x_pl, y_pl, s=12, c="w", marker="o", edgecolors="r", linewidth = 0.5)

    #Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('Критерий Колмогорова. Функция распределения')
    ax.legend()
    plt.show()


def max_deviation(y, Fy, G):
    '''
        y - точки оси Ох
        Fy - эмпирическая, G(у) - теоретическая
    '''

    max_d_pl = -1000
    for i in range(len(y)):
        d_pl = abs(Fy[i] - G(y[i]))
        if d_pl > max_d_pl:
            max_d_pl = d_pl
            point_pl = (y[i], Fy[i])
       
    Fy.insert(0,0)
    y.insert(0,0)

    max_d_mi = -1000
    for i in range(len(y)):
        d_mi = abs(Fy[i] - G(y[i]))
        if d_mi > max_d_mi:
            max_d_mi = d_mi
            point_mi = (y[i], Fy[i])

    Fy.pop(0)
    y.pop(0)

    if max_d_pl > max_d_mi:
        return max_d_pl, point_pl

    return max_d_mi, point_mi


def empirical_function(Y):
    N = len(Y)
    res = {}
    for i in range(N):
        if not Y[i] in res:
            res[Y[i]] = i + 1
        else:
            res[Y[i]] += 1

    x, n = [], []
    for value, count in res.items():
        x.append(value)
        n.append(count / N)

    return x, n  


def Kolmogorov_criterion(p, f, G, n=30):
    """
            Проверка гипотезы по критерию согласия Колмогорова
        p - вероятность успеха испытания
        f - Функция вероятности - Геометрическое распределение
        n - размер выборки (>40 !!!)
    """

    print("\n\t Критерий Колмогорова")

    # построение вариационного ряда
    Y = DSV.calculate_Y(p, n, f)[0]

    # построение графика эмпирической функции распределения
    y, Fy = empirical_function(Y)

    # определение мах отклонения эмпирической функции от теоретической
    max_d, point = max_deviation(y, Fy, lambda n: G(n, p))
    print("Максимальное отклонение {:.4f} в точке ({:.4f}, {:.4f})".format(max_d, point[0], point[1]))
    plot_empirical_and_analytical_function(y, Fy, lambda n: G(n, p))

    # Т.к. Y - дискретна используется статистика с поправкой Большева:
    # вычисление значения критерия (Лямбда)
    criterion = (6*n*max_d + 1) / (6*n**(1/2))

    L_tb = [1.22, 1.36, 1.63]
    a = [0.1, 0.05, 0.01]
    for i in range(3):
        print('a = ', a[i], '\n',
              'Lambda = {:.4f}'.format(criterion),
              '>' if criterion > L_tb[i] else '<',
              L_tb[i],
              ' => гипотеза отклоняется' if criterion > L_tb[i] else ' => гипотеза принимается')

    return 'K', max_d, point, a, criterion, L_tb
