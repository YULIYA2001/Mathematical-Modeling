import numpy as np
import math
import matplotlib.pyplot as plt

from sympy import diff, symbols, lambdify




# главная функция подсчетов и построения гистограммы, полигона распределения и эмпирической функции
def evaluate_and_draw(Y, theoretical_func):
    ''' 
        Y - массив СВ Y с заданным распределением - точки на Ох
        X - массив значений СВ X (X = G(Y)) - точки на Оу
        theoretical_func - lambda, теоретическая функция распределения СВ Y
        return - массив с параметрами для построения гистограмм для 2-х методов
    '''

    # n - объем выборки
    n = len(Y) #n = 100

    # M - количество интервалов (нестрогие формулы из 4сем-ТВиМС-Модуль4.2)
    if n <= 100:
        M = (int)(n**0.5) 
    else:
        M = (int)(math.log10(n)*2)
        for m in [(int)(math.log10(n)*3), (int)(math.log10(n)*4)]:
            if n % m == 0:
                M = m

    # -----------     Построение гистограммы равноинтервальным методом     -----------
    A, B, F, v, h = coef_ABFv_eq_interval(Y, M)

    print("\nРавноинтервальный метод")
    print_table(A, B, v, [h]*M, F)

    # Проверка суммы площади столбиков гистограммы
    sum = 0
    A.append(B[-1])
    for i in range(len(A)-1):
        sum += F[i]*(A[i+1]-A[i])
    A.pop()
    print("Суммарная площадь прямоугольников S = {:}".format(sum))

    # гистограмма, полигон распределения и теоретическая функция плотности
    title = "Равноинтервальный. n={:} M={:} S={:.3f}".format(n, M, sum)
    window_title = 'График 3. Графическое отображение плотности распределения'
    plot_all_charts(A, B, F, v, n, theoretical_func, Y, title, window_title)

   
    #---------        Построение гистограммы равновероятностным методом    -----------
    A2, B2, F2, v2, h2 = coef_ABFv_equiprobable(Y, M)
    print("\nРавновероятостный метод")
    print_table(A2, B2, v2, h2, F2)

    # Проверка суммы площади столбиков гистограммы
    sum2 = 0
    A2.append(B2[-1])
    for i in range(len(A2)-1):
        sum2 += F2[i]*(A2[i+1]-A2[i])
    A2.pop()
    print("Суммарная площадь прямоугольников S = {:}".format(sum2))

    # гистограмма, полигон распределения и теоретическая функция плотности
    title = "Равновероятностный. n={:} M={:} S={:.3f}".format(n, M, sum)
    window_title = 'График 2. Графическое отображение плотности распределения'
    plot_all_charts(A, B, F, v, n, theoretical_func, Y, title, window_title)


    return [ ['Равноинтервальный метод', A, B, v, [h]*M, F], ['Равновероятостный метод', A2, B2, v2, h2, F2] ]
    



# Поиск коэффициентов A B F v для построения гистограммы равноинтервальным методом    
def coef_ABFv_eq_interval(Y, M):
    '''
        Y - массив значений СВ Y с заданной функцией распределения
        M - количество интервалов разбиения
        return A (массив левых границ интервалов разбиения)
               B (массив правых границ)
               F (массив средних плотностей вероятности для каждого интервала)
               v (массив количества СВ в каждом интервале)
               h (высота столбцов)
    '''

    n = len(Y)
    h = (Y[-1] - Y[0]) / M
    
    A, B = [Y[0]], []
    for i in range(1,M):
        A.append(Y[0] + i*h)
        B.append(A[-1])
    B.append(Y[-1])

    v = np.zeros(M)
    i, j = 0, 0
    while i < n:
        while Y[i] < B[j]:
            v[j] += 1
            i += 1
        if Y[i] == B[j]:
            if j == len(B)-1:
                v[j] += 1
                i +=1
            else:
                v[j] += 0.5
                v[j+1] += 0.5
                i += 1
        j += 1

    F = []
    for i in range(M):
        F.append(v[i] / (n*h))

    return A, B, F, v, h



# Поиск коэффициентов A B F v для построения гистограммы равновероятностным методом
def coef_ABFv_equiprobable(Y, M):
    '''
        Y - массив значений СВ Y с заданной функцией распределения
        M - количество интервалов разбиения
        return A (массив левых границ интервалов разбиения)
               B (массив правых границ)
               F (массив средних плотностей вероятности для каждого интервала - высота столбцов)
               V (массив количества СВ в каждом интервале)
               h (шаг)
    '''

    n = len(Y)
    v = int(n / M)

    A, B = [Y[0]], []
    for i in range(1, M):
        A.append((Y[i*v-1] + Y[i*v]) / 2)
        B.append(A[i])
    B.append(Y[-1])

    h = []
    a = A.copy()
    a.append(B[-1])
    for i in range(M):
        h.append(a[i+1] - a[i])

    F = []
    for i in range(M-1):
        F.append(v / (n*h[i]))
    # "лишние" точки в последний интервал
    F.append((n - v*(M - 1)) /  (n*h[-1]))

    V = [v]*M
    V[M-1] = n - v*(M - 1)

    return A, B, F, V, h





# Построение гистограммы равноинтервальным/равновероятностным методом
def plot_all_charts(A, B, F, v, n, theoretical_func, Y, title, title_w):
    '''
        A - массив левых границ интервалов разбиения
        B - массив правых границ
        F - массив средних плотностей вероятности для каждого интервала (высота столбца)
        v - массив количества СВ в каждом интервале
        n - объем выборки (кол-во эл-тов Y)
        theoretical_func - lambda, теоретическая функция распределения СВ Y
        Y - массив СВ Y с заданным распределением
        title - строка c видом метода, размером выборки, кол-вом интервалов разбиения, S под гистограммой
        title_w - заголовок окна графика
    '''
    # гистограмма
    x_pl, y_pl = A.copy(), F.copy()
    x_pl.insert(0, A[0])
    y_pl.insert(0, 0)
    x_pl.append(B[-1])
    y_pl.append(0)
    fig, ax = plt.subplots()
    ax.step(x_pl, y_pl, "r-", where='post', label = "Гистограмма")

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
    ax.plot(x_pl_p, y_pl_p, 'g-o', label="Полигон распределения")

    # поиск функции плотности g(y) по функции распределения G(y)
    y = symbols('y')
    dencity_func = diff( theoretical_func(y) )
    y = y
    g = lambdify(y, dencity_func)

    # теоритическая функция плотности распределения
    Ay, By = A[0], B[-1]
    x = np.linspace(Ay, By, 1000)
    y = [g(i) for i in x]

    #y = [diff(theoretical_func(i)) for i in x]

    ax.plot(x, y, color='k', label="Аналитическая функция")

    # Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title(title_w)

    ax.legend()
    ax.grid()
    plt.title(title)
    plt.show()



# вывод в виде таблицы коэффициентов распределения  fi* - высота столбца, y - координаты по оси Oх плолигона
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


        
