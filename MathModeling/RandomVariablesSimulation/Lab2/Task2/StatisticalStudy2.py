import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sympy import diff, integrate, symbols, lambdify

import Task2.FormDiscreteSV as DSV
import Task2.StStudyElements.Histograms2 as Histograms
import IntervalEstimates


# основная функция статистического исследования
def execute(f, G, M_t, D_t, p, Y_count, A0):
    ''' 
        f - Функция вероятности - Геометрическое распределение
        G - cтупенчатая функция распределения дискретной СВ 
        M_t - функция для расчета теоретического значения матожидания
        D_t - функция для расчета теоретического значения дисперсии
        p - вероятность успеха испытания
        Y_count - количество СЧ в Y для генерации
        A0 - левая граница интервала по оси Ох (мин значение дискретной СВ Y) 
    '''
    # генерация СВ Х и нахождение дискретных СВ Y
    Y, X = DSV.calculate_Y(p, Y_count, f)

    # гистограмма частот, полигон распределения, теоретическая и эмпирическая функции плотности
    hist_params = Histograms.evaluate_and_draw(Y, lambda n: f(n, p), A0)

    # теоритический и практический графики функции распределения
    draw_graph(G, Y, X, p)

    # точечные оценки
    pe_params = point_estimates(Y, M_t(p), D_t(p))

    # интервальные оценки
    m, D = pe_params[0], pe_params[1]
    ie_params = IntervalEstimates.evaluate(Y, m, D, M_t(p), D_t(p))

    print('\n\n', '-'*106, '\n\n')

    return hist_params, pe_params, ie_params




# Построение теоретического и практического графиков
def draw_graph(G, Y, X, p):
    ''' 
        G - cтупенчатая функция распределения дискретной СВ 
        Y - дискретные СЧ Y 
        X - сгенерированные генератором БСВ СЧ X
        p - вероятность успеха испытания
    '''

    Y_count = len(Y)
    n_max = Y[-1]


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
    x_pl = [i for i in range(n_max+1)]
    y_pl = [G(i, p) for i in range(n_max+1)]
    x_pl.append(x_pl[-1] + 1)
    y_pl.append(y_pl[-1])
    ax.step(x_pl, y_pl, "r-", where='post', linewidth = 0.8, label = "Аналитическая функция")
    # не выколотые точки графика
    x_pl.pop(0)
    y_pl.pop()
    ax.scatter(x_pl, y_pl, s=6, c="r", marker="o", edgecolors="r", linewidth = 0.5)
    # выколотые точки графика
    y_pl.pop(0)
    x_pl.pop()
    ax.scatter(x_pl, y_pl, s=6, c="w", marker="o", edgecolors="r", linewidth = 0.5)

    #ГРАФИК ПРАКТИЧЕСКИЙ
    Y.insert(0, 0)
    X.insert(0, 0)
    Y.append(Y[-1] + 1)
    X.append(X[-1])
    ax.step(Y, X, "g-", where='post', linewidth = 0.8, label = "Эмпирическая функция")
    # не выколотые точки графика
    Y.pop(0)
    X.pop()
    ax.scatter(Y, X, s=6, c="g", marker="o", edgecolors="g", linewidth = 0.5)
    # выколотые точки графика
    X.pop(0)
    Y.pop()
    ax.scatter(Y, X, s=6, c="w", marker="o", edgecolors="g", linewidth = 0.5)

    plt.title('Геометрическое распределение, |Y| = {:}'.format(Y_count))
    # Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('График 1. Функция распределения')
    ax.legend()
    plt.show()




# Точечные оценки
def point_estimates(Y, M_t=-1, D_t=-1, is_print=True):
    ''' 
        Y - СЧ Y по формуле у = G_-1(x)
        M_t - число - теоретическое значение матожидания  (-1 - не задано (не важно))
        D_t - число - теоретическое значение дисперсии
        return m_dots - точечная оценка МО СВ
               D_dots - точечная оценка дисперсии СВ 
    '''
    n = len(Y)

    # математическое ожидание
    m_dots = sum(Y) / n

    # дисперсия
    D_dots = 0
    for x in Y:
        D_dots += (x - m_dots) ** 2 
    D_dots /= (n - 1)

    if is_print:
        print("\n\nТочечная оценка МО СВ: ", m_dots)
        print("Точечная оценка дисперсии СВ: ", D_dots)

        print('\nТеоретическое значение МО СВ: ', M_t)
        print('Теоретическое значение дисперсии СВ: ', D_t)

        delta_m = np.abs(M_t-m_dots)
        delta_D = np.abs(D_t-D_dots)
        print('\nDelta МО СВ: ', delta_m)
        print('Delta дисперсии СВ: ', delta_D, '\n\n\n')

    return [m_dots, D_dots, M_t, D_t, delta_m, delta_D]



