import copy
import numpy as np
import math
import matplotlib.pyplot as plt 

import StatisticalStudy.IntervalEstimates as IntervalEstimates


def round_half_up(n, decimals=0):
    ''' 
        Стандартное математическое округление 
        n, decimals - что округлять, до каких ед. округлять
    '''
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


def is_independent(P, pX, pY):
    '''
        Проверка независимости ДСВ Х и У

        P - матрицa двумерной ДСВ
        pX, pY - одномерные ряды вероятностей СВ Х и У

        return independent - True/False - зависимы ли СВ Х и Y
    '''
    n, m = len(P), len(P[0])

    independent = True

    for i in range(n):
        for j in range(m):
            if round_half_up(P[i][j], 6) != round_half_up(pX[i] * pY[j], 6):
                independent = False
                break   
            
    return independent





def conditional_dencity(P, pX, pY):
    '''
        Представление функций условных плотностей в виде строк и построение их графиков

        P - матрицa вероятностей двумерной ДСВ
        pX, pY - одномерные ряды вероятностей СВ Х и У

        return Pxy, Pyx - матрицы условных законов распределения
    '''
    n, m = len(P), len(P[0])
    Pxy, Pyx = np.zeros((n,m)), np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            Pyx[i][j] = P[i][j] / pX[i]

    for j in range(m):
        for i in range(n):
            Pxy[i][j] = P[i][j] / pY[j]

    return Pxy, Pyx





def plot_histograms(x, y, Px, Pyx):
    '''
        Гистограммы составляющих Х и У двумерной ДСВ

        pX, pYx - одномерный ряд вероятностей СВ Х и матрица усл. вероятностей У
        x, y - массивы псевдослучайных дискретных величин Х и У
    '''

    # для практических гистограмм
    n, m = len(Px), len(Pyx[0])
    hist_params_x = coef_eq_interval(x, n, A0=0)
    hist_params_y = coef_eq_interval(y, m, A0=0)
    sum_x_hist = check_hist_sum(hist_params_x[0], hist_params_x[1], hist_params_x[2])
    sum_y_hist = check_hist_sum(hist_params_y[0], hist_params_y[1], hist_params_y[2])    

    # для теоретических функций
    X = [0] + [i for i in range(n+1)] 
    X += [X[-1]]
    Y = [0] + [i for i in range(m+1)] 
    Y += [Y[-1]]
    Xy, Yy = Px.copy().tolist(), copy.deepcopy(Pyx)
    Xy = [0] + Xy + [Px[-1], 0]
    Yy = np.hstack((np.zeros((n,1)), Yy))
    Yy = np.hstack((Yy, [[i] for i in Pyx[:,-1]]))
    Yy = np.hstack((Yy, np.zeros((n,1))))

    fig = plt.figure(figsize=(10, 4))

    # ГРАФИК ДЛЯ СВ Х
    ax = fig.add_subplot(131)
    plt.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("Px")
    ax.set_title('ДСВ Х \nS = {:.3f}'.format(sum_x_hist), fontsize=10)
    # практическая гистограмма и полигон распределения
    plot_hist(ax, hist_params_x, 'X')
    # теоретическая гистограмма
    ax.step(X, Xy, "k--", where='post', label = "Аналитическая функция", linewidth=1)
    plt.legend(loc='best', fontsize=8)
    
    # ГРАФИК ДЛЯ СВ У P(y|xi)
    ax = fig.add_subplot(132)
    plt.grid(True)
    ax.set_xlabel("y")
    ax.set_title('ДСВ Y      P(Y|Xi)теор.'.format(sum_y_hist), fontsize=10)
    # теоретическая гистограмма
    for i in range(n):
        ax.step(Y, Yy[i], '--', where='post', linewidth=2, label = "x={:}".format(i))
    plt.legend(loc='best', fontsize=8)

    # ГРАФИК ДЛЯ СВ У P(y|x_middle)
    mid_x = int(round_half_up( sum(x) / len(x) ))
    ax = fig.add_subplot(133)
    plt.grid(True)
    ax.set_xlabel("y")
    ax.set_title('ДСВ Y \nP(Y|{:}) S = {:.3f}'.format(mid_x, sum_y_hist), fontsize=10)
    # практическая гистограмма и полигон распределения
    plot_hist(ax, hist_params_y, 'Y')
    # теоретическая гистограмма
    ax.step(Y, Yy[mid_x], "k--", where='post', label = "Аналитическая функция", linewidth=1)
    plt.legend(loc='best', fontsize=8)
   
    fig = plt.gcf()
    fig.canvas.set_window_title('График 2. Гистограммы составляющих двумерной ДСВ')
    fig.suptitle("|X| = |Y| = {:} точек".format(len(x)), fontsize=10)
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
    axis.plot(x_pl_p, y_pl_p, 'g-o', label="Многоугольник вероятностей "+SV)



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


def coef_eq_interval(Y_, M, A0=1):
    '''
        Поиск коэффициентов A B F v для построения гистограммы равноинтервальным методом

        Y_ - массив значений СВ Y с заданной функцией распределения
        M - количество интервалов разбиения
        A0 - левая граница интервала по оси Ох (мин значение дискретной СВ Y) 

        return A (массив левых границ интервалов разбиения)
               B (массив правых границ)
               F (массив средних плотностей вероятности для каждого интервала)
               v (массив количества СВ в каждом интервале)
               h (высота столбцов)
    '''

    Y = Y_.copy()
    #Y.sort()

    n = len(Y)
    h = 1

    A = [i for i in range(M)]
    B = [i for i in range(1, M+1)]
    #A = [ i for i in range(A0, Y[-1]+1) ]
    #B = [ i for i in range(A0+1, Y[-1]+2) ]

    setY = set(Y)
    v = np.zeros(M)
    for i in range(M):
        if i in setY:
            v[i] = Y.count(i)

    F = []
    for i in range(M):
        F.append(v[i] / (n*h))

    return A, B, F, v, h, n










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
   

def coef_eq_interval_2SV(X, Y, N, M):
    '''
        Поиск коэффициентов A B F v для построения гистограммы равноинтервальным методом

        X, Y - массивы значений СВ Х и Y
        N, M - количество интервалов разбиения для Х и У соотв.
       
        return Ax, Ay (массив левых границ интервалов разбиения)
               Bx, By (массив правых границ)
               F (массив средних плотностей вероятности для каждого квадрата - высота столбца)
               v (массив количества СВ в каждом квадрате (ax,bx,ay,by))
               hx, hy (шаг по х, у)
    '''
    n = len(X)

    hx = hy = 1
    Ax = [i for i in range(N)]
    Bx = [i for i in range(1, N+1)]
    Ay = [i for i in range(M)]
    By = [i for i in range(1, M+1)]

    v = np.zeros((N,M))
    for k in range(n):
        v[ X[k] ][ Y[k] ] += 1 
        
    F = np.zeros((N*M))
    for i in range(N):
        for j in range(M):
            F[M*i+j] = v[i][j] / (n*hx*hy)

    return Ax, Bx, Ay, By, F, v, hx, hy, n


def plot_histogram_3D(x, y, P):
    '''
        График 3D гистограммы распределения двумерной ДСВ (3D-график)

        P - матрицa двумерной ДСВ
        x, y - массивы псевдослучайных дискретных величин Х и У
    '''
    n, m = len(P), len(P[0])
    hist_params = coef_eq_interval_2SV(x, y, n, m)
    hist_sum = check_hist_sum_2SV(hist_params[6], hist_params[7], hist_params[4])

    # 3D ГРАФИКИ ПЛОТНОСТИ ДВУМЕРНОЙ НСВ
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(131, projection='3d')

    # ТЕОРЕТИЧЕСКИЙ
    X_ = [x for x in range(n)]
    Y_ = [y for y in range(m)]
    X_, Y_ = np.meshgrid(X_, Y_)
    X_, Y_ = X_.ravel(), Y_.ravel()
    P = np.array(P)
    P_ = P.flatten()
    ax.bar3d(X_, Y_, np.zeros_like(P_), 1, 1, P_, 'b', shade=True, alpha=0.7)
 
    # ПРАКТИЧЕСКИЙ
    Ax_, Bx_, Ay_, By_, F_, v_, hx_, hy_, n_ = hist_params
    X, Y = np.meshgrid(Ax_, Ay_)
    X, Y = X.ravel(), Y.ravel()
    ax.bar3d(X, Y, np.zeros_like(F_), hx_, hy_, F_, 'r', shade=True, alpha=0.7)
    ax.set_title('Теоретическая + практическая', fontsize=8)
    ax.set_xlabel("Х")
    ax.set_ylabel("Y")
    ax.set_zlabel("P")

    
    ax = fig.add_subplot(132, projection='3d')
    # ГИСТОГРАММА ПРАКТИЧЕСКАЯ (отдельно)
    ax.bar3d(X, Y, np.zeros_like(F_), hx_, hy_, F_, 'r', shade=True, alpha=1)
    ax.set_title('Практическая', fontsize=8)
    ax.set_xlabel("Х")
    ax.set_ylabel("Y")
    ax.set_zlabel("P")
    
    ax = fig.add_subplot(133, projection='3d')
    # ГИСТОГРАММА ТЕОРЕТИЧЕСКАЯ (отдельно)
    ax.bar3d(X_, Y_, np.zeros_like(P_), 1, 1, P_, 'b', shade=True, alpha=1)
    ax.set_title('Теоретическая', fontsize=8)
    ax.set_xlabel("Х")
    ax.set_ylabel("Y")
    ax.set_zlabel("P")
    
    # Заголовок окна
    fig = plt.gcf()
    fig.canvas.set_window_title('График 3. Функция плотности f(x,y) и гистограмма')
    fig.suptitle("Гистограммы  распределения двумерной ДСВ\nn = {:} точек (X, Y). S = {:.4f}"
                 .format(n_, hist_sum), fontsize=12)
    plt.show()








def point_estimates(X, Y, P, Pyx):
    '''
        Точечные оценки и теоретические

        P - матрицa двумерной ДСВ
        Pyx - матрица условных вероятностей СВ У
        X, Y - массивы псевдослучайных дискретных величин Х и У

        return Mx, Dx, My, Dy, Rxy, Mx_t, Dx_t, My_t, Dy_t, Rxy_t - МО и дисперсии СВ Х и У 
                                                                    и их корреляция (точечн. и теор.)
    '''
    n = len(X)
    N, M = len(P), len(P[0])
    x_t = [i for i in range(N)]
    y_t = [j for j in range(M)]

    # точечные MO
    Mx = sum(X) / n
    My = sum(Y) / n

    # точечные дисперсии
    Dx, Dy = 0, 0
    for i in range(n):
        Dx += (X[i] - Mx) ** 2 
        Dy += (Y[i] - My) ** 2 
    Dx /= (n - 1)
    Dy /= (n - 1)
    # точечные ковариация и корреляция
    Kxy = sum(X[i]*Y[i] for i in range(n)) / n - (Mx * My)
    Rxy = Kxy / (np.sqrt(Dx*Dy))

   
    #mid_X = int(round_half_up(Mx))

    # теоретические MO
    Mx_t = 0
    for i in range(N):
        for j in range(M):
            Mx_t += x_t[i] * P[i][j]
    My_t = 0
    for j in range(M):
        for i in range(N):
            My_t += y_t[j] * P[i][j]
    #Myx_t = 0
    #for j in range(M):
    #    Myx_t += y_t[j] * Pyx[mid_X][j]
    # теоретические дисперсии
    Dx_t = 0
    for i in range(N):
        for j in range(M):
            Dx_t += (x_t[i] - Mx_t)**2 * P[i][j]
    Dy_t = 0
    for j in range(M):
        for i in range(N):
            Dy_t += (y_t[j] - My_t)**2 * P[i][j]
    #Dyx_t = 0
    #for j in range(M):
    #    Dyx_t += (y_t[j] - Myx_t)**2 * Pyx[mid_X][j]
    # теоретические ковариация и корреляция
    Kxy_t = 0
    for i in range(N):
        for j in range(M):
            Kxy_t += x_t[i] * y_t[j] * P[i][j]
    Kxy_t -= (Mx_t * My_t)
    Rxy_t = Kxy_t / (np.sqrt(Dx_t*Dy_t))

    return Mx, Dx, My, Dy, Rxy, Mx_t, Dx_t, My_t, Dy_t, Rxy_t







def interval_estimates(x, y, Mx, Mx_t, Dx, Dx_t, Myx, Myx_t, Dyx, Dyx_t, Rxy, Rxy_t):
    '''
        Интервальные оценки МО, дисперсии, коррелляции
        x, y - массивы ДСВ
        Mx, Mx_t, Dx, Dx_t - точечные и теоретические МО и дисперсия СВ Х
        Myx, Myx_t, Dyx, Dyx_t - точечные и теоретические МО и дисперсия СВ У
        Rxy, Rxy_t - точечная и теоретическая корреляция СВ Х и У
    '''
    
    ie_params_x = IntervalEstimates.evaluate(x, Mx, Dx, Mx_t, Dx_t)
    ie_params_y = IntervalEstimates.evaluate(y, Myx, Dyx, Myx_t, Dyx_t)
    ie_params_r = IntervalEstimates.evaluate_R(len(x), Rxy, Rxy_t)

    return ie_params_x, ie_params_y, ie_params_r