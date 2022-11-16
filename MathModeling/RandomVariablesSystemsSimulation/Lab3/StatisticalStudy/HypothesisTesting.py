import numpy as np
from scipy import stats


def test(Mx, Mx_t, My, My_t, Dx, Dx_t, Dy, Dy_t, Rxy, Rxy_t, n):
    '''
        Проверить  статистические  гипотезы  о  соответствии  полученных  оценок характеристик СВ теоретическим

        Mx, Mx_t, My, My_t - математическое ожидание СВ Х и У практическое и теоретическое
        Dx, Dx_t, Dy, Dy_t - дисперсия СВ Х и У практическое и теоретическое
        Rxy, Rxy_t - корреляция СВ Х и У практическое и теоретическое
    '''
    print('СВ Х')
    x_params_M = test_M_know_D(Mx, Mx_t, Dx_t, n)
    print('СВ Y')
    y_params_M = test_M_know_D(My, My_t, Dy_t, n)

    print('\nСВ Х')
    x_params_D = test_D(Dx, Dx_t, n)
    print('СВ Y')
    y_params_D = test_D(Dy, Dy_t, n)

    return [x_params_M, y_params_M, x_params_D, y_params_D]



def test_D(D, D2, n):
    ''' Проверка соответствия дисперсии '''
    n2=n
    if D == 0 or D2 == 0:
        return

    if D > D2:
        F = np.abs(D/D2)
    else:
        F = np.abs(D2/D)

    level = [0.1, 0.05, 0.01]
    # квантили для распределения Фишера для level = [0.1, 0.05, 0.01] 
    f_tb = [stats.f.ppf(0.95, n-1, n2-1), stats.f.ppf(0.975, n-1, n2-1), stats.f.ppf(0.995, n-1, n2-1)]

    print('Тестирование дисперсии D. Уровень значимости a')
    for i in range(len(level)):
        print('a = ', level[i], '\n',
              '|F| = {:.4f}'.format(F),
              '>' if F > f_tb[i] else '<',
              '{:2f}'.format(f_tb[i]),
              ' => гипотеза отклоняется' if F > f_tb[i] else ' => гипотеза принимается')

    return level, F, f_tb


def test_M_know_D(M, Mt, Dt, n):
    ''' Проверка соответствия МО '''
    Z = np.abs( (M - Mt) / (np.sqrt(Dt/n+Dt/n)))
    level = [0.1, 0.05, 0.01]
    # квантили для стандартного нормального распределения для level = [0.1, 0.05, 0.01] 
    z_tb = [1.64, 1.96, 2.58]

    print('Тестирование МО при известых дисперсиях. Уровень значимости a')
    for i in range(len(level)):
        print('a = ', level[i], '\n',
              '|Z| = {:.4f}'.format(Z),
              '>' if Z > z_tb[i] else '<',
              z_tb[i],
              ' => гипотеза отклоняется' if Z > z_tb[i] else ' => гипотеза принимается')

    return level, Z, z_tb


def test_M_unknow_D(M, Mt, D, n):
    ''' Проверка соответствия МО '''
    T = np.abs( (M - Mt) / (np.sqrt(D/n)) )
    level = [0.1, 0.05, 0.01]
    # квантили для распределения Стьюдента для level = [0.1, 0.05, 0.01] 
    t_tb = [stats.t.ppf((0.95), n+n2-2), stats.t.ppf((0.975), n+n2-2), stats.t.ppf((0.995), n+n2-2)]

    print('Тестирование МО при неизвестых дисперсиях. Уровень значимости a')
    for i in range(len(level)):
        print('a = ', level[i], '\n',
              '|T| = {:.4f}'.format(T),
              '>' if T > t_tb[i] else '<',
              '{:2f}'.format(t_tb[i]),
              ' => гипотеза отклоняется' if T > t_tb[i] else ' => гипотеза принимается')

    return level, T, t_tb
    
    

