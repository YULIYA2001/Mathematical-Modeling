import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sympy import diff, symbols, lambdify




# главная функция подсчетов и построения гистограммы, многоугольника вероятностей и функции вероятности
def evaluate_and_draw(Y, theor_f, A0):
    ''' 
        Y - массив дискретных СВ Y с заданным распределением - точки на Ох
        A0 - левая граница интервала по оси Ох (мин значение дискретной СВ Y) 
        theor_f - Функция вероятности
    '''

    # n - объем выборки
    n = len(Y) #n = 100

    # M - количество интервалов
    M = Y[-1]   

    '''
    #################
    M = int(round(n ** (1/3) + 1/2))
    if (M > n): M = n
    ################
    '''
   
    print("\nРавноинтервальный метод")
    A, B, F, v, h = coef_ABFv_eq_interval(Y, M, A0)
    #print_table(A, B, v, [h]*M, F)

    # Проверка суммы площади столбиков гистограммы
    sum = 0
    A.append(B[-1])
    for i in range(len(A)-1):
        sum += F[i]*(A[i+1]-A[i])
    A.pop()
    print("Суммарная площадь прямоугольников S = {:}".format(sum))

    # гистограмма, многоугольник вероятностей и теоретическая функция вероятности
    title = "Равноинтервальный. n={:} M={:} S={:.3f}".format(n, M, sum)
    plot_all_charts(A, B, F, v, n, theor_f, Y, title)

    return ['Равноинтервальный метод', A, B, v, [h]*M, F]
    


# Поиск коэффициентов A B F v для построения гистограммы равноинтервальным методом
def coef_ABFv_eq_interval(Y, M, A0=1):
    '''
        Y - массив значений СВ Y с заданной функцией распределения
        M - количество интервалов разбиения
        A0 - левая граница интервала по оси Ох (мин значение дискретной СВ Y) 
        return A (массив левых границ интервалов разбиения)
               B (массив правых границ)
               F (массив средних плотностей вероятности для каждого интервала)
               v (массив количества СВ в каждом интервале)
               h (высота столбцов)
    '''

    n = len(Y)
    h = 1

    A = [ i for i in range(A0, Y[-1]+1) ]
    B = [ i for i in range(A0+1, Y[-1]+2) ]

    '''
    #####################
    h = (Y[-1] - Y[0]) / M

    A, B = [Y[0]], []
    for i in range(1,M):
        A.append(Y[0] + i*h)
        B.append(A[-1])
    B.append(Y[-1])
    ######################
    '''

    setY = set(Y)
    v = np.zeros(M)
    for i in range(M):
        if i+1 in setY:
            v[i] = Y.count(i+1)


    '''
    ##############
    v = np.zeros(M)
    
    for yi in Y:
        i = 0
        while yi > B[i]:
            i+=1
        v[i] += 1
    ################
    '''

    F = []
    for i in range(M):
        F.append(v[i] / (n*h))

    return A, B, F, v, h



# Построение гистограммы равноинтервальным методом
def plot_all_charts(A, B, F, v, n, theor_f, Y, title):
    '''
        A - массив левых границ интервалов разбиения
        B - массив правых границ
        F - массив средних плотностей вероятности для каждого интервала (высота столбца)
        v - массив количества СВ в каждом интервале
        n - объем выборки (кол-во эл-тов Y)
        theor_f - теоретическая функция вероятности СВ Y
        Y - массив СВ Y с заданным распределением
        title - строка видом метода, размером выборки, кол-вом интервалов разбиения, S под гистограммой
    '''

    n_max = Y[-1]
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

    # полигон распределения
    dots = B.copy()
    dots.insert(0, A[0])
    x_pl_p = [dots[0]]
    for i in range(len(dots)-1):
        x_pl_p.append((dots[i+1] + dots[i]) / 2)
    x_pl_p.pop(0)
    freq = []
    for i in range(len(v)):
        freq.append(v[i]/n)
    y_pl_p = freq
    x_pl_p.insert(0, dots[0])
    x_pl_p.append(dots[-1])
    y_pl_p.insert(0, 0)
    y_pl_p.append(0)
    ax.plot(x_pl_p, y_pl_p, 'g--', marker='.', markersize=3, label="Многоугольник вероятностей", linewidth=0.7)
    
    # Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('График 2. Графическое отображение плотности распределения')

    ax.legend()
    ax.grid()
    plt.title(title)
    plt.show()



# вывод в виде таблицы коэфф. распределения  fi* - высота столбца, y - координаты по оси Oх многоугольника вер.
def print_table(A, B, v, h, F):

    dots = B.copy()
    dots.insert(0, A[0])
    y = []
    for i in range(len(dots)-1):
        y.append((dots[i+1] + dots[i]) / 2)

    n = len(A)
    print("   i   |    Ai    |    Bi    |    vi   |   hi    |   fi*   |    y")
    print("---------------------------------------------------------------------")
    for i in range(n):
        print("%5d" % (i+1), 
              " | %.4f" % A[i] if A[i] < 0 else " |  %.4f" % A[i],
              " | %.4f" % B[i] if B[i] < 0 else " |  %.4f" % B[i],
              " | %6d" % v[i],
              " | %.4f" % h[i],
              " | %.4f" % F[i],
              " | %.4f" % y[i] if y[i] < 0 else " |  %.4f" % y[i]
              )
