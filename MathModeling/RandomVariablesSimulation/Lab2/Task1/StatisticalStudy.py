import numpy as np
import matplotlib.pyplot as plt

from sympy import diff, integrate, symbols, lambdify
import scipy.integrate

import Task1.InverseFunctionMethod as IFM
import Task1.StStudyElements.Histograms as Histograms
import IntervalEstimates


# основная функция статистического исследования
def execute(G, G_1, Y_count, G_str='не указано'):
    ''' 
        G - lambda-выражение; исходная функция распределения G(y)
        G_1 - lambda-выражение; функция, обратная G(y)
        Y_count - количество точек генерации СВ Y
        G_str - строкавая запись функции G
        return:
            hist_params - массивы с параметрами для построения гистограмм для 2-х методов
            pe_params - массив с параметрами точечных оценок 
            ie_params - массив с параметрами интервальных оценок 
    '''
    # генерация СВ Х и нахождение СВ Y методом обратных функций
    y_nums, x_nums = IFM.calculate_Y(G, G_1, Y_count)

    # гистограмма частот, полигон распределения, теоретическая и эмпирическая функции плотности
    hist_params = Histograms.evaluate_and_draw(y_nums, G)

    # теоритический и практический графики функции распределения
    draw_graph(G, G_str, y_nums, x_nums)

    # точечные оценки
    pe_params = point_estimates(y_nums, G, G_1)

    # интервальные оценки
    m, D, m_t, D_t = pe_params[0], pe_params[1], pe_params[2], pe_params[3]
    ie_params = IntervalEstimates.evaluate(y_nums, m, D, m_t, D_t)

    print('\n\n', '-'*106, '\n\n')


    return hist_params, pe_params, ie_params





# Построение теоретического и практического графиков
def draw_graph(G, G_str, y_nums, x_nums):
    ''' 
        G - lambda-выражение; исходная функция распределения G(y)
        G_1 - lambda-выражение; функция, обратная G(y)
        y_nums - СЧ Y по формуле у = G_-1(x)
        x_nums - сгенерированные генератором БСВ  СЧ X
    '''

    Y_count = len(y_nums)

    fig, ax = plt.subplots()

    # теоретический график исходной функции распределения G(y)
    y = np.arange(0, y_nums[-1]+0.1, 0.001)
    plt.plot(y, G(y), 'k', label='Теоретический', linewidth=2)

    # практический график по точкамм Y, Х
    if Y_count > 100:
        plt.plot(y_nums, x_nums, 'r:', label='Практический', linewidth=2)
        #plt.plot(y_nums, G(y_nums), 'k:', label='Практический')
    else:
        plt.plot(y_nums, x_nums, 'r:', marker=".", label='Практический', linewidth=2)
        #plt.plot(y_nums, G(y_nums), 'k:', marker=".", label='Практический')

    plt.title('n = {:} точек,  {:}'.format(Y_count, G_str))
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    # Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('График 1. Функция распределения')
    plt.show()



# Точечные оценки
def point_estimates(Y, G, G_1):
    ''' 
        Y - СЧ Y по формуле у = G_-1(x)
        G - lambda-выражение; исходная функция распределения G(y)
        G_1 - lambda-выражение; функция, обратная G(y)
        return m_dots - точечная оценка МО СВ
               D_dots - точечная оценка дисперсии СВ 
               m_theor - теоретическое значение МО СВ
               D_theor - теоретическое значение дисперсии СВ
    '''
    n = len(Y)

    # математическое ожидание
    m_dots = sum(Y) / n
    print("\n\nТочечная оценка МО СВ: ", m_dots)

    # дисперсия
    D_dots = 0
    for x in Y:
        D_dots += (x - m_dots) ** 2 
    D_dots /= (n - 1)
    print("Точечная оценка дисперсии СВ: ", D_dots)

    # теоретические значения
   
    # поиск функции плотности g(y) по функции распределения G(y)
    y = symbols('y')
    dencity_func = diff( G(y) )
    y = y
    g = lambdify(y, dencity_func)

    # границы интегрирования
    a, b = G_1(0), G_1(1)

    y = symbols('y')
    #m_theor = integrate(y * g(y), (y, a, b))
    m_theor = scipy.integrate.quad(lambda y: y * g(y), a, b)[0]
    #D_theor = integrate( (y-m_theor)**2 * g(y), (y, a, b) )
    D_theor = scipy.integrate.quad(lambda y: (y-m_theor)**2 * g(y), a, b)[0]
    print('\nТеоретическое значение МО СВ: ', m_theor)
    print('Теоретическое значение дисперсии СВ: ', D_theor)

    delta_m = np.abs(m_theor-m_dots)
    delta_D = np.abs(D_theor-D_dots)
    print('\nDelta МО СВ: ', delta_m)
    print('Delta дисперсии СВ: ', delta_D, '\n\n\n')

    return [m_dots, D_dots, m_theor, D_theor, delta_m, delta_D]


