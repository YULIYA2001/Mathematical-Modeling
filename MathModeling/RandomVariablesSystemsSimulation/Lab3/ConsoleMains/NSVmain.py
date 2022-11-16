#################################################################################            
#                           Двумерная непрерывная СВ                            #
#                                   Вариант 3                                   #
#################################################################################

import math
import numpy as np
import Generator.FormNSV as Form2NSV
import StatisticalStudy.NSV.StStudy as StStudyNSV




def f(x, y):
    ''' функция плотности распределения СНСВ '''
    return np.sin(x+y) / 2 

# (a, b) - интервал распределения СВ (для X и Y)
ax, ay = 0, 0
bx, by = np.pi/2, np.pi/2

# максимальное значение функции f1x(x)
f_min = 0
f_max = 1/2


def f1x(x):
    ''' частная функция плотности для х f1(x) '''
    return (np.cos(x) + np.sin(x)) / 2
f1x_max = f1x(math.pi/4)

def f1y(y):
    ''' частная функция плотности для y f1(y) '''
    return (np.cos(y) + np.sin(y)) / 2

def f2xy(x, y):
    ''' частная функция плотности для y f2(x|y) '''
    return np.sin(x+y) / (np.cos(y) + np.sin(y))

def f2yx(y, x):
    ''' частная функция плотности для y f2(y|x) '''
    return np.sin(x+y) / (np.cos(x) + np.sin(x))
# max(f2yx) достигается при y_max = pi/2 - x
def f2yx_max(x):
    return 1 / (np.sin(x)+np.cos(x))

def F1x(x):
    ''' частная функция распределения для х F1(x) '''
    return 1/2 + (1/2) * np.sin(x) - (1/2) * np.cos(x)

def F2yx(y, x):
    ''' частная функция распределения для y f2(y|x) '''
    return -( -np.cos(x) + np.cos(x+y) ) / (np.cos(x) + np.sin(x))

def Mx_t():
    ''' теоритическое МО величины х '''
    return np.pi / 4

def My_t():
    ''' теоритическое МО величины y '''
    return np.pi / 4

def M2yx_t(x):
    ''' теоритическое условное МО величины y '''
    return (1/2) * (np.sin(x)*np.pi + 2*np.cos(x) - 2*np.sin(x)) / (np.cos(x) + np.sin(x))

def Dx_t():
    ''' теоритическая дисперсия величины х '''
    return (1/16)*np.pi**2 + (1/2)*np.pi - 2

def Dy_t():
    ''' теоритическая дисперсия величины y '''
    return (1/16)*np.pi**2 + (1/2)*np.pi - 2

def D2yx_t(x):
    ''' теоритическая условная дисперсия величины y '''
    return (np.pi**2 * np.cos(x)*np.sin(x) - 8*np.cos(x)*np.sin(x) + 4*np.pi - 12) / (8*np.cos(x) * np.sin(x) + 4)

def Rxy_t():
    ''' теоритическая корелляция величин х и y '''
    return -(np.pi**2 - 8*np.pi + 16) / (np.pi**2 + 8*np.pi - 32)




if __name__ == '__main__':
    #  формированиe двумерной НСВ
    x, y = Form2NSV.form_values(f1x, f2yx, f1x_max, f2yx_max, ax, bx, ay, by, count=10)
    
    # проверка на независимость
    if StStudyNSV.is_independent(f, f1x, f1y, ax, bx, ay, by):
        print('1. Независимые СВ X и Y. f(x,y) = f(x)*f(y)')
    else:
        print('1. Зависимые СВ X и Y. f(x,y) ≠ f(x)*f(y)')

    # условные плотности распределения
    print('\n2. Условные плотности распределения:')
    str_res = StStudyNSV.conditional_dencity_functions(f2xy, f2yx, f1x, f1y, ax, bx, ay, by)
    print('f(x|y) = {:} \nf(y|x) = {:}'.format(str_res[0], str_res[1]))

    # гистограммы составляющих двумерной НСВ и графики их плотностей распределения в одной сист. координат
    StStudyNSV.plot_histograms_and_graphs(x, y, f1x, f2yx, ax, bx, ay, by)

    # гистограммa распределения двумерной НСВ и 3D-график плотности распределения в одной сист. координат 
    StStudyNSV.plot_histograms_and_graphs_3D(x, y, ax, bx, ay, by, f_min, f_max)
    
    # теоретические, точечные и интервальные значения характеристик ДНСВ (МО, дисперсия, корреляция)
    Mx, Dx, My, Dy, Rxy = StStudyNSV.point_estimates(x, y)


    print('\n5. Теоретические значения характеристик:')
    print('MO[x] = {:.6f}\nD[x] = {:.6f}\nM[y] = {:.6f}\nD[y] = {:.6f}\nr[xy] = {:.6f}'
          .format(Mx_t(), Dx_t(), My_t(), Dy_t(), Rxy_t()))

    print('Точечные значения характеристик:')
    print('MO[x] = {:.6f}\nD[x] = {:.6f}\nM[y] = {:.6f}\nD[y] = {:.6f}\nr[xy] = {:.6f}'
          .format(Mx, Dx, My, Dy, Rxy))
    #StStudyNSV.plot_M2yx_t(x, M2yx_t)

    print('Интервальные значения характеристик:')
    StStudyNSV.interval_estimates(x, y, Mx, Mx_t(), Dx, Dx_t(), My, My_t(), Dy, Dy_t(), Rxy, Rxy_t())

    # проверка гипотез
    # Проверка гипотезы о равенстве статистических средних значений и дисперсий
    import StatisticalStudy.HypothesisTesting as HT
    HT.test(Mx, Mx_t(), My, My_t(), Dx, Dx_t(), Dy, Dy_t(), Rxy, Rxy_t(), len(x))