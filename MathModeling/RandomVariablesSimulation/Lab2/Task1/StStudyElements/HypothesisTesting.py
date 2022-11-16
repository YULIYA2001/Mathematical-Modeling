import math
import numpy as np
import matplotlib.pyplot as plt
import scipy

from sympy import diff, symbols, solve, simplify, lambdify


import Task1.InverseFunctionMethod as IFM
import Task1.StStudyElements.Histograms as Histograms



def test(G, G_1, kind='A', n=0):
    '''
        G - lambda-выражение; исходная функция распределения G(y)
        G_1 - lambda-выражение; функция, обратная G(y)
        n - размер выборки (>40 !!!)
    '''
    params = []

    if n == 0:
        if kind == 'P':
            params = Pirson_criterion(G, G_1)
        elif kind == 'K':
            params = Kolmogorov_criterion(G, G_1)
        elif kind == 'M':
            params = Mises_criterion(G, G_1)
        else:
            Pirson_criterion(G, G_1)
            Kolmogorov_criterion(G, G_1)
            Mises_criterion(G, G_1)
    else:
        if kind == 'P':
            params = Pirson_criterion(G, G_1, n)
        elif kind == 'K':
            params = Kolmogorov_criterion(G, G_1, n)
        elif kind == 'M':
            params = Mises_criterion(G, G_1, n)
        else:
            Pirson_criterion(G, G_1, n)
            Kolmogorov_criterion(G, G_1, n)
            Mises_criterion(G, G_1, n)

    return params


#------------------------------------------- Критерий Колмогорова ----------------------------------------------

def plot_empirical_and_analytical_function(y, Fy, G):

    Y = y

    # обратная функция поиск
    x, y = symbols('x, y') 
    expr = x - G(y) 
    y = y
    str_G_1 = np.abs(solve(expr, y)[0])
    G_1 = lambdify(x, str_G_1)

    y = Y

    Ay = min(G_1(0), G_1(1))
    By = max(G_1(0), G_1(1))
    if By == float('inf'):
        By = y[-1]

    x_pl, y_pl = y.copy(), Fy.copy() 
    x_pl.insert(0, Ay-0.1)
    y_pl.insert(0, 0)
    x_pl.append(By+0.1)
    y_pl.append(1)
    fig, ax = plt.subplots()
    ax.step(x_pl, y_pl, "r-", where='post', label = "Эмпирическая функция")
    # выколотые точки графика
    y_pl.pop(0)
    x_pl.pop(0)
    y_pl.pop()
    x_pl.pop()
    ax.scatter(x_pl, y_pl, s=25, c="w", marker="o", edgecolors="r")    

    x_pl_t = np.linspace(Ay, By, 20)

    y_pl_t = [G(x) for x in x_pl_t]
    ax.plot(x_pl_t, y_pl_t, 'b', label = "Аналитическая функция")
    ax.scatter(x_pl_t, y_pl_t, s=10, c="b", marker="o", edgecolors="b")
    ax.grid()
    ax.legend()
    #Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('Критерий Колмогорова. Функция распределения')
    plt.show()


def max_deviation(y, Fy, G):
    '''
        Fy - эмпирическая, G - теоретическая
    '''

    max_d_pl = -1000
    for i in range(len(y)):
        d_pl = abs(Fy[i] - G(y[i]))
        if d_pl > max_d_pl:
            max_d_pl = d_pl
            point_pl = (y[i], Fy[i])
       
    Fy2 = Fy.copy()
    Fy2.pop()
    Fy2.insert(0,0)
    max_d_mi = -1000
    for i in range(len(y)):
        d_mi = abs(Fy2[i] - G(y[i]))
        if d_mi > max_d_mi:
            max_d_mi = d_mi
            point_mi = (y[i], Fy2[i])

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


def Kolmogorov_criterion(G, G_1, n=30):
    """
            Проверка гипотезы по критерию согласия Колмогорова
        G - lambda-выражение; исходная функция распределения G(y)
        G_1 - lambda-выражение; функция, обратная G(y)
        n - размер выборки (>40 !!!)
    """

    print("\n\t Критерий Колмогорова")

    # построение вариационного ряда
    Y = IFM.calculate_Y(G, G_1, Y_count=n)[0]

    # построение графика эмпирической функции распределения
    y, Fy = empirical_function(Y)

    # определение мах отклонения эмпирической функции от теоретической
    max_d, point = max_deviation(y, Fy, G)
    print("Максимальное отклонение {:.4f} в точке ({:.4f}, {:.4f})".format(max_d, point[0], point[1]))
    plot_empirical_and_analytical_function(y, Fy, G)

    # вычисление значения критерия (Лямбда)
    criterion = n**(1/2) * max_d

    L_tb = [1.22, 1.36, 1.63]
    a = [0.1, 0.05, 0.01]
    for i in range(3):
        print('a = ', a[i], '\n',
              'Lambda = {:.4f}'.format(criterion),
              '>' if criterion > L_tb[i] else '<',
              L_tb[i],
              ' => гипотеза отклоняется' if criterion > L_tb[i] else ' => гипотеза принимается')

    return 'K', max_d, point, a, criterion, L_tb





#--------------------------------------------- Критерий Пирсона ------------------------------------------------

def plot_hist(A, B, F, Y, G):
    """Построение гистограммы (равновер.) и аналитической функции для функции плотности"""

    x_pl, y_pl = A.copy(), F.copy()
    x_pl.insert(0, A[0])
    y_pl.insert(0, 0)
    x_pl.append(B[-1])
    y_pl.append(0)
    fig, ax = plt.subplots()
    ax.step(x_pl, y_pl, "r-", where='post', label = "Гистограмма равновер. методом")
    ax.scatter(Y, [0]*len(Y), s=1, c="k", marker="o", edgecolors="k")

    # поиск функции плотности g(y) по функции распределения G(y)
    y = symbols('y')
    dencity_func = diff( G(y) )
    y = y
    g = lambdify(y, dencity_func)

    Ay, By = 0, Y[-1]
    x = np.linspace(Ay, By, 1000)
    y = [g(i) for i in x]
    ax.plot(x, y, color='k', label="Аналитическая функция")
    ax.legend()
    ax.grid()
    #Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('Критерий Хи2 - Пирсона. Функция плотности')
    plt.show()


def X_square(A, B, M, n, v, G):
    """Расчет значения критерия хи-квадрат"""

    Ax = A.copy()
    Ax.append(B[-1])
    X_sq, sum_pi = 0, 0
    for i in range(M):
        p = G(Ax[i+1]) - G(Ax[i])
        p_ = v[i]/n   #1/M
        X_sq += n * (p - p_)**2 / p
        sum_pi += p

    good = False
    if abs(1-sum_pi) <= 0.01:
        print("\nКонтрольное соотношение для суммы Pi выполняется")
        print(abs(1-sum_pi), "<= 0.01\n")
        good = True
        
    return X_sq, good, sum_pi


def Pirson_criterion(G, G_1, n=200):
    """
            Проверка гипотезы по критерию согласия Пирсона
        G - lambda-выражение; исходная функция распределения G(y)
        G_1 - lambda-выражение; функция, обратная G(y)
        n - размер выборки (>40 !!!)
    """

    if n < 40:
        print('Размер выборки должен быть больше 40')
        return

    print("\n\n Критерий X^2 Пирсона")
    #M = (int)(math.log10(n)*3)
    M = int(round(n ** (1/3) + 1/2))
    if (M > n): M = n
    print("M =", M, "разрядов")

    # менять выборку, если не выполняется контрольное соотношенеие E(Pi)~1
    while True:
        Y = IFM.calculate_Y(G, G_1, Y_count=n)[0]
        A, B, F, v, h = Histograms.coef_ABFv_equiprobable(Y, M)
        X_sq, good, sum_pi = X_square(A, B, M, n, v, G)
        if good:
            break

    plot_hist(A, B, F, Y, G)

    k = M - 1
    #print(k)
    # k = 5 значения Х для а из таблицы: a = 0.1 - X = 9.236, a = 0.05 - X = 11.070, a = 0.01 - X = 15.086
    #X_tb = [9.236, 11.070, 15.086]
    X_tb = [scipy.stats.chi2.ppf(0.90, k), scipy.stats.chi2.ppf(0.95, k), scipy.stats.chi2.ppf(0.99, k)]
    a = [0.1, 0.05, 0.01]
    for i in range(3):
        print('a = ', a[i], '\n',
              'X^2 = {:.4f}'.format(X_sq), 
              '>' if X_sq > X_tb[i] else '<',
              X_tb[i],
              ' => гипотеза отклоняется' if X_sq > X_tb[i] else ' => гипотеза принимается')

    return 'P', sum_pi, a, X_sq, X_tb






#------------------------------------------- Критерий Мизеса ----------------------------------------------

def Mises_criterion(G, G_1, n=50):
    """
            Проверка гипотезы по критерию согласия Мизеса
        G - lambda-выражение; исходная функция распределения G(y)
        G_1 - lambda-выражение; функция, обратная G(y)
        n - размер выборки (>40 !!!)
    """
    print("\n\n\t\t\t Критерий Мизеса")

    Y = IFM.calculate_Y(G, G_1, Y_count=n)[0]

    y, Fy = empirical_function(Y)
    #F_teor = [G(x) for x in y]

    sum = 0
    for i in range(n):
        sum += (G(y[i]) - (i - 0.5)/n)**2
    criterion = 1/(12 * n) + sum

    L_tb = [0.347, 0.461, 0.744]
    a = [0.1, 0.05, 0.01]
    for i in range(3):
        print('a = ', a[i], '\n',
              'Lambda = {:.4f}'.format(criterion),
              '>' if criterion > L_tb[i] else '<',
              L_tb[i],
              ' => гипотеза отклоняется' if criterion > L_tb[i] else ' => гипотеза принимается')

    return 'M', a, criterion, L_tb

