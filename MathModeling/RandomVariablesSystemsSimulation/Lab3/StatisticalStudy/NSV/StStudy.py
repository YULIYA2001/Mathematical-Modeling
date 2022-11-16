import inspect
import math
import matplotlib.patches as patches
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt 
from matplotlib.ticker import LinearLocator

import StatisticalStudy.IntervalEstimates as IntervalEstimates

def plot_M2yx_t(x, M2yx_t):
    '''  
        График функции МО СВ У, зависящей от СВ Х - My(x)

        x - массив сгенерированных псевдослучайных чисел СВ Х
        M2yx_t - функция МО У от аргумента х
    '''
    x_ = x.copy()
    x_.sort()

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    plt.plot(x_, M2yx_t(x_), 'black', linewidth=1, label='M(y|x) теор.')
    plt.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("M(y|x)")
    ax.legend(loc='best', fontsize=8)
    fig = plt.gcf()
    fig.canvas.set_window_title('График 6. Функция МО СВ У - My(x)')
    plt.show()






def is_independent(f, f1x, f1y, ax, bx, ay, by):
    ''' 
        Проверка на независимость СВ Х и У

        f - функция плотности двух СВ f(x,y)
        f1x, f1y - безусловные функции плотности СВ Х и У - f(x), f(y)
        (ax, bx), (ay, by) - интервалы распределения СВ Х и У соотв.
        
        return True - СВ Х и У независимы, False - зависимы
    '''
    x = np.linspace(ax, bx, 5)
    y = np.linspace(ay, by, 5)
    for i in range(5):
        if f1x(x[i]) * f1y(y[i]) != f(x[i], y[i]):
            return False
    return True







def conditional_dencity_functions(f2xy, f2yx, f1x, f1y, Ax, Bx, Ay, By):
    '''
        Представление функций условных плотностей в виде строк и построение их графиков

        f2xy, f2yx - условные плотности СВ Х и У f(x|y) и f(y|x)
        f1x, f1y - безусловные плотности СВ Х и У f(x) и f(y)
        (Ax,Bx), (Ay,By) - интервалы распределения СВ Х и Y соотв.

        return f2xy_str, f2yx_str - строковое представление условных плотностей СВ Х и У
    '''
    f2xy_str = inspect.getsource(f2xy).split('return')[1].replace('np.','').replace('\n', '')
    f2yx_str = inspect.getsource(f2yx).split('return')[1].replace('np.','').replace('\n', '')

    fig = plt.figure(figsize=(9, 4))
    step = np.pi/100
    x = np.arange(Ax, Bx+step, step)
    y = np.arange(Ay, By+step, step)

    # ГРАФИК f(x|y)
    ax = fig.add_subplot(121)
    plt.plot(x, f2xy(x, y[0]), 'yellow', label='y=0', linewidth=1)
    for i in range(1,len(y)):
        if i % 10 == 0:
            plt.plot(x, f2xy(x, y[i]), label='y=π/{:.1f}'.format(np.pi/y[i]), linewidth=1)
        else:
            plt.plot(x, f2xy(x, y[i]), 'b', linewidth=1, alpha=0.3)
    plt.plot(x, f1x(x), 'black', linestyle='--', label='f(x)', linewidth=1)
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x|y)")

    # ГРАФИК f(x|y) 3D
    ax = fig.add_subplot(122, projection='3d')
    x, y = np.meshgrid(x, y)
    surf1 = ax.plot_surface(x, y, f2xy(x, y), cmap="Blues", linewidth=0, 
                            antialiased=False, alpha=0.6, label='f(x|y)')
    surf2 = ax.plot_surface(x, y, f1x(x), color='black', linewidth=0, antialiased=False, alpha=0.7, label='f(x)')
    # for legend
    surf1._edgecolors2d, surf2._edgecolors2d = surf1._edgecolor3d, surf2._edgecolor3d
    surf1._facecolors2d, surf2._facecolors2d = surf1._facecolor3d, surf2._facecolor3d
    # Add a color bar which maps values to colors.
    fig.colorbar(surf1, shrink=0.5, ax=[ax], location='left')
    # название графика и надписи на осях
    ax.set_title('f(x|y) = {:}\nf(y|x) = {:}'.format(f2xy_str, f2yx_str))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x), f(x|y)")
    ax.legend(loc='best', fontsize=8)

    # Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('График 2. Функции условных плотностей распределения f(x|y) и f(y|x)')
    fig.suptitle("Функции f(x|y) и f(y|x) симметричны (для СВ Y графики аналогичны)"
                 .format(len(x)), fontsize=10)
    plt.show()

    return f2xy_str, f2yx_str






"""
def conditional_MOy_Dy_t(M2yx_t, D2yx_t):
    '''  
        M2yx_t, D2yx_t - теоретические значения МО и дисперсии СВ Y, зависящей от СВ Х
    '''
    M2yx_t_str = inspect.getsource(M2yx_t).split('return')[1].replace('np.','').replace('\n', '')
    D2yx_t_str = inspect.getsource(D2yx_t).split('return')[1].replace('np.','').replace('\n', '')
    return M2yx_t_str, D2yx_t_str
"""
def point_estimates(X, Y):
    '''
        Точечные оценки

        X, Y - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.

        return M_x, D_x, M_y, D_y, R_xy - МО и дисперсии СВ Х и У и их ккорреляция
    '''
    n = len(X)
    M_x = sum(X) / n

    D_x = 0
    for xi in X:
        D_x += (xi - M_x) ** 2 
    D_x /= (n - 1)

    M_y = sum(Y) / n

    D_y = 0
    for yi in Y:
        D_y += (yi - M_y) ** 2 
    D_y /= (n - 1)

    K_xy = sum(X[i]*Y[i] for i in range(n)) / n - (M_x * M_y)
    R_xy = K_xy / (np.sqrt(D_x*D_y))
    #R_xy = st.pearsonr(X, Y)[0]

    return M_x, D_x, M_y, D_y, R_xy

def interval_estimates(x, y, Mx, Mx_t, Dx, Dx_t, My, My_t, Dy, Dy_t, Rxy, Rxy_t):
    '''
        Интервальные оценки МО, дисперсии, коррелляции
        x, y - массивы СВ
        Mx, Mx_t, Dx, Dx_t - точечные и теоретические МО и дисперсия СВ Х
        My, My_t, Dy, Dy_t - точечные и теоретические МО и дисперсия СВ У
        Rxy, Rxy_t - точечная и теоретическая корреляция СВ Х и У
    '''
    
    ie_params_x = IntervalEstimates.evaluate(x, Mx, Dx, Mx_t, Dx_t)
    ie_params_y = IntervalEstimates.evaluate(y, My, Dy, My_t, Dy_t)
    ie_params_r = IntervalEstimates.evaluate_R(len(x), Rxy, Rxy_t)

    return ie_params_x, ie_params_y, ie_params_r







def plot_histograms_and_graphs(x, y, f1x, f2yx, ax, bx, ay, by):
    '''
        Гистограммы для СВ Х и У и графики их плотностей распределения в одной сист. координат

        х, у - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.
        f1x, f2yx - функция плотности СВ Х и условная функция плотности СВ У
        (ax, bx), (ay, by) - интервалы распределения СВ Х и У соотв.
    '''

    # для практических гистограмм
    M = count_M(len(x))
    hist_params_x = coef_eq_interval(x.copy(), M, ax, bx)
    hist_params_y = coef_eq_interval(y.copy(), M, ay, by)
    sum_x_hist = check_hist_sum(hist_params_x[0], hist_params_x[1], hist_params_x[2])
    sum_y_hist = check_hist_sum(hist_params_y[0], hist_params_y[1], hist_params_y[2])

    # для теоретических функций
    step = np.pi/100
    X = np.arange(ax, bx+step, step)
    Y = np.arange(ay, by+step, step)

    # 3D и 2D графики
    plot_fx_fyx_3D(X, Y, f1x, f2yx, x, y)
    plot_fx_fyx_2D(X, Y, x, y, f1x, f2yx, hist_params_x, hist_params_y)
    

def check_hist_sum(A, B, F):
    '''
        Проверка суммы площади столбиков гистограммы на равенство 1

        A - массив левых границ интервалов разбиения
        B - массив правых границ
        F - массив средних плотностей вероятности для каждого интервала (высоты столбцов)

        return sum - сумма площади столбцов гистограммы
    '''
    sum = 0
    A.append(B[-1])
    for i in range(len(A)-1):
        sum += F[i]*(A[i+1]-A[i])
    A.pop()
    print("Суммарная площадь прямоугольников S = {:}".format(sum))
    return sum


def count_M(n):
    ''' 
        Поиск числа интервалов для построения гистограммы

        n - кол-во СВ в выборке

        return M - количество столбцов для гистограммы 
    '''
    if n <= 100:
        M = (int)(np.sqrt(n)) 
    else:
        M = (int)(math.log10(n)*2)
        for m in [(int)(math.log10(n)*3), (int)(math.log10(n)*4)]:
            if n % m == 0:
                M = m
    return M
    

def coef_eq_interval(Y, M, a, b):
    '''
        Поиск коэффициентов A B F v для построения гистограммы равноинтервальным методом

        Y - массив значений СВ Y с заданной функцией распределения
        M - количество интервалов разбиения
        (a, b) - интервал распределения СВ Y

        return A (массив левых границ интервалов разбиения)
               B (массив правых границ)
               F (массив средних плотностей вероятности для каждого интервала)
               v (массив количества СВ в каждом интервале)
               h (высота столбцов)
    '''
    Y.sort()
    n = len(Y)
    h = (b - a) / M
    
    A, B = [Y[0]], []
    for i in range(1,M):
        A.append(Y[0] + i*h)
        B.append(A[-1])
    B.append(Y[-1])

    v = np.zeros(M)
    i, j = 0, 0
    while i < n:
        if j >= len(B):
            j -= 1
        while Y[i] < B[j]:
            v[j] += 1
            i += 1
            if i >= n:
                break
        if i >= n:
            continue

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

    return A, B, F, v, h, n


def plot_fx_fyx_3D(X, Y, f1x, f2yx, x, y):
    '''
        График 3D плотностей распределения CВ Х и У в одной сист. координат

        X, Y - равномерные массивы для теоретических функций
        f1x, f2yx - функция плотности СВ Х и условная функция плотности СВ У
        x, y - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.
    '''

    # ГРАФИК f(x) и f(y|x) 3D
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111, projection='3d')
    X_, Y_ = np.meshgrid(X, Y)

    # поверхности
    surf1 = ax.plot_surface(X_, Y_, f2yx(Y_, X_), cmap="Blues", linewidth=0, 
                            antialiased=False, alpha=0.5, label='f(y|x)')
    surf2 = ax.plot_surface(X_, Y_, f1x(X_), color='grey', linewidth=0,
                           antialiased=False, alpha=0.5, label='f(x)')
    # проекции поверхностей
    ax.plot(X, f1x(X), 'k', zs=np.pi/2, zdir='y', label='f(x) Ox', linewidth=1)
    ax.plot(Y, f2yx(Y, X[0]), 'b', zs=0, zdir='x', linewidth=1, alpha=0.5, label='f(y|x) Oy')
    for i in range(1,len(X)):
        ax.plot(Y, f2yx(Y, X[i]), 'b', zs=0, zdir='x', linewidth=1, alpha=0.5)

    # for legend
    surf1._edgecolors2d, surf2._edgecolors2d = surf1._edgecolor3d, surf2._edgecolor3d
    surf1._facecolors2d, surf2._facecolors2d = surf1._facecolor3d, surf2._facecolor3d
    # Add a color bar which maps values to colors.
    fig.colorbar(surf1, shrink=0.5, ax=[ax], location='left')
    # название графика и надписи на осях
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='best', fontsize=8)
    # Заголовок окна и графиков
    fig = plt.gcf()
    fig.suptitle("Функции f(x) и f(y|x) и их проекции на оси Ох и Оу соотв.", fontsize=10)
    fig.canvas.set_window_title('График 3. Функции состовляющих двумерной НСВ f(x) и f(y|x)')
    plt.show()


def plot_hist(axis, hist_params, SV='X'):
    '''
        Часть построения: гистограмма и полигон
        axis - из plot
        hist_params - параметры для построения гистограммы 
    '''
    A, B, F, v, h, n = hist_params
    x_pl, y_pl = A.copy(), F.copy()
    x_pl.insert(0, A[0])
    y_pl.insert(0, 0)
    x_pl.append(B[-1])
    y_pl.append(0)
    # гистограмма
    axis.step(x_pl, y_pl, "r-", where='post', label = "Гистограмма (равноинт.) "+SV)
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
    axis.plot(x_pl_p, y_pl_p, 'g-o', label="Полигон распределения "+SV)
    plt.legend(loc='best', fontsize=8)
    plt.ylim(-0.2, 1.1)


def plot_fx_fyx_2D(X, Y, x, y, f1x, f2yx, hist_params_x, hist_params_y):
    '''
        График 2D плотностей распределения и гистограмм CВ Х и У в одной сист. координат

        X, Y - равномерные массивы для теоретических функций
        f1x, f2yx - функция плотности СВ Х и условная функция плотности СВ У
        x, y - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.
        hist_params_x, hist_params_y - параметры для построения гистограмм
    '''
    # ГРАФИКИ f(x) и f(y|x) 2D
    fig = plt.figure(figsize=(9.5, 4))

    # ГРАФИК f(x)
    ax = fig.add_subplot(121)
    plt.plot(X, f1x(X), 'black', linewidth=1)
    plt.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    f1x_str = inspect.getsource(f1x).split('return')[1].replace('np.','').replace('\n', '')
    ax.set_title('f(x) = {:}'.format(f1x_str), fontsize=10)

    # ГИСТОГРАММА И ПОЛИГОН РАСПРЕДЕЛЕНИЯ СВ Х
    plot_hist(ax, hist_params_x, 'X')

    # ГРАФИК f(y|x)
    ax = fig.add_subplot(122)
    for i in range(len(X)):
        plt.plot(Y, f2yx(Y, X[i]), 'b', linewidth=1, alpha=0.7)
    # гистограмма для СВ У будет соответствовать f(y|x_const) x_const - среднему для СВ Х
    x_mid = sum(x) / len(x)
    plt.plot(Y, f2yx(Y, x_mid), 'k', linewidth=1)
    plt.grid(True)
    ax.set_xlabel("y")
    ax.set_ylabel("f(y|x)")
    f2yx_str = inspect.getsource(f2yx).split('return')[1].replace('np.','').replace('\n', '')
    ax.set_title('f(y|x) = {:}'.format(f2yx_str), fontsize=10)

    # ГИСТОГРАММА И ПОЛИГОН РАСПРЕДЕЛЕНИЯ СВ Y
    plot_hist(ax, hist_params_y, 'Y')

    fig = plt.gcf()
    fig.canvas.set_window_title('График 4. Функции состовляющих двумерной НСВ f(x) и f(y|x)')
    fig.suptitle("|X| = |Y| = {:} точек".format(len(x)), fontsize=10)
    plt.show()






def plot_histograms_and_graphs_3D(x, y, Ax, Bx, Ay, By, f_min, f_max):
    '''
        Гистограммa распределения двумерной НСВ и 3D-график плотности распределения в одной сист. координат

        x, y - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.
        (Ax, Bx), (Ay, By) - интервалы распределения СВ Х и У соотв.
        f_min, f_max - мин и мах значения функции плотности двумерной СВ f(x,y)
    '''
    M = count_M(len(x))
    hist_params = coef_eq_interval_2SV(x, y, M, Ax, Bx, Ay, By)
    check_hist_sum_2SV(hist_params[6], hist_params[7], hist_params[4])

    # построение 3D графика
    plot_3D_graphs(x, y, Ax, Bx, Ay, By, f_min, f_max, hist_params)


def check_hist_sum_2SV(hx, hy, F):
    '''
        Проверка суммы площади столбиков гистограммы на равенство 1

        hx, hy - шаги по Ох и Оу соотв. (ширина и глубина столбцов)
        F - массив средних плотностей вероятности для каждого интервала (высоты столбцов)

        return sum - сумма площади столбцов гистограммы
    '''
    sum = 0
    for i in range(len(F)):
        sum += F[i] * hx * hy
    print("Суммарная площадь параллелепипедов S = {:}".format(sum))
    return sum
   

def coef_eq_interval_2SV(X, Y, M, ax, bx, ay, by):
    '''
        Поиск коэффициентов A B F v для построения гистограммы равноинтервальным методом

        X, Y - массивы значений СВ Х и Y
        M - количество интервалов разбиения
        (a, b) - интервалы распределения СВ Х и Y
        return Ax, Ay (массив левых границ интервалов разбиения)
               Bx, By (массив правых границ)
               F (массив средних плотностей вероятности для каждого квадрата - высота столбца)
               v (массив количества СВ в каждом квадрате (ax,bx,ay,by))
               hx, hy (шаг по х, у)
    '''

    n = len(X)
    hx = (bx - ax) / M
    hy = (by - ay) / M
    
    Ax, Bx = [min(X)], []
    Ay, By = [min(Y)], []
    for i in range(1,M):
        Ax.append(min(X) + i*hx)
        Bx.append(Ax[-1])
        Ay.append(min(Y) + i*hy)
        By.append(Ay[-1])
    Bx.append(X[-1])
    By.append(Y[-1])

    v = np.zeros((M,M))
    for k in range(n):
        i, j = 0, 0
        while X[k] > Bx[i]:
            i += 1
            if i == len(Bx):
                i -= 1
                break;
        i_res = i
        
        while Y[k] > By[j]:
            j += 1
            if j == len(By):
                j -= 1
                break;
        j_res = j

        v[i_res][j_res] += 1
    
    #F = np.zeros((M,M))
    F = np.zeros((M*M))
    for i in range(M):
        for j in range(M):
            F[M*i+j] = v[i][j] / (n*hx*hy)

    return Ax, Bx, Ay, By, F, v, hx, hy, n


def plot_3D_graphs(x, y, Ax, Bx, Ay, By, f_min, f_max, hist_params):
    '''
        График 3D плотностей распределения и гистограмм CВ Х и У в одной сист. координат

        x, y - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.
        (Ax, Bx), (Ay, By) - интервалы распределения СВ Х и У соотв.
        f_min, f_max - мин и мах значения функции плотности двумерной СВ f(x,y)
        hist_params_x, hist_params_y - параметры для построения гистограмм
    '''
    # 3D ГРАФИКИ ПЛОТНОСТИ ДВУМЕРНОЙ НСВ
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')

    # ТЕОРЕТИЧЕСКИЙ
    X = np.linspace(Ax, Bx, 50)
    Y = np.linspace(Ay, By, 50)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, np.sin(X+Y)/2, cmap="plasma", linewidth=0, antialiased=False)
    # ПРАКТИЧЕСКИЙ (точки)
    ax.scatter(x, y, np.zeros_like(x), marker=".", c="red", s=2, label='Сгенерированные точки')
    # ГИСТОГРАММА
    Ax_, Bx_, Ay_, By_, F_, v_, hx_, hy_, n_ = hist_params
    X, Y = np.meshgrid(Ax_, Ay_)
    X, Y = X.ravel(), Y.ravel()
    ax.bar3d(X, Y, np.zeros_like(F_), hx_, hy_, F_, 'b', shade=True, alpha=0.05)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, ax=[ax], location='left')
    # название графика и надписи на осях
    ax.set_title("Гистограмма \nточки X, Y\nf(x,y)=1/2*sin(x+y)", fontsize=10)
    ax.set_xlabel("CВ Х")
    ax.set_ylabel("СВ Y")
    ax.set_zlabel("f(x,y)")
    # Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('График 5. Функция плотности f(x,y) и гистограмма')

    ax = fig.add_subplot(122, projection='3d')
    # ГИСТОГРАММА (отдельно)
    Ax_, Bx_, Ay_, By_, F_, v_, hx_, hy_, n_ = hist_params
    X, Y = np.meshgrid(Ax_, Ay_)
    X, Y = X.ravel(), Y.ravel()
    ax.bar3d(X, Y, np.zeros_like(F_), hx_, hy_, F_, 'b', shade=True)
    ax.set_title("Гистограмма", fontsize=10)
    ax.set_xlabel("CВ Х")
    ax.set_ylabel("СВ Y")
    ax.set_zlabel("f(x,y)")

    fig.suptitle("n = {:} точек (X, Y)".format(n_), fontsize=12)
    plt.show()





'''
    # границы осей
    ax.set_xlim(Ax, Bx)
    ax.set_ylim(Ay, By)
    ax.set_zlim(f_min, f_max)
    # цифры на осях
    plt.gca().xaxis.set_major_locator(LinearLocator(5))
    plt.gca().yaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter('{x:.02f}')
    ax.yaxis.set_major_formatter('{x:.02f}')
    ax.zaxis.set_major_formatter('{x:.01f}')
'''