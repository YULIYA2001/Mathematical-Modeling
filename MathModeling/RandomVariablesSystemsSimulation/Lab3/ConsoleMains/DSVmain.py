import random
import numpy as np

import Generator.FormDSV as FormDSV
import StatisticalStudy.DSV.StStudyDSV as StStudyDSV



n = 4
m = 5

# независимые СВ
pX_n = FormDSV.generate_independent_DSV(n)
pY_n = FormDSV.generate_independent_DSV(m)
P = FormDSV.generate_independent_2DSV_matrix(pX_n, pY_n)

#P = FormDSV.generate_2DSV_matrix(n, m)

P = [[0.1, 0.05, 0.1],
     [0.1, 0.1, 0.05],
     [0.05, 0.05, 0.1],
     [0.05, 0.2, 0.05]]
'''
P = [[0.01, 0.04, 0.02, 0.03],
     [0.04, 0.16, 0.08, 0.12],
     [0.02, 0.08, 0.04, 0.06],
     [0.03, 0.12, 0.06, 0.09]]
'''
 


if __name__ == '__main__':
    #  формированиe двумерной ДСВ
    x, y = FormDSV.form_values(P, count=1000)

    print('\n0. Одномерные законы распределения:')
    Px, Py = FormDSV.find_one_dimensional_2DSV(P)
    print('p(xi|y) = {:}     sum = {:}\np(yi|x) = {:}     sum = {:}'.format(Px, sum(Px), Py, sum(Py)))

    # проверка на независимость
    independent = StStudyDSV.is_independent(P, Px, Py)
    print('\n1. СВ Х и У независимы?', independent)

    # условные плотности распределения
    print('\n2. Условные законы распределения:')
    Pxy, Pyx = StStudyDSV.conditional_dencity(P, Px, Py)
    print('p(xi|y) = \n{:}\nsum = {:}\n\np(yi|x) = \n{:}\nsum = {:}\n'.format(Pxy, sum(Px), Pyx, sum(Py)))
    
    # гистограммы составляющих двумерной ДСВ
    StStudyDSV.plot_histograms(x, y, Px, Pyx)

    # гистограммa распределения двумерной ДСВ (3D-график)
    StStudyDSV.plot_histogram_3D(x, y, P)
    
    # теоретические, точечные и интервальные значения характеристик ДНСВ (МО, дисперсия, корреляция)
    Mx, Dx, My, Dy, Rxy, Mx_t, Dx_t, My_t, Dy_t, Rxy_t = StStudyDSV.point_estimates(x, y, P, Pyx)

    print('\n5. Теоретические значения характеристик:')
    print('MO[x] = {:.6f}\nD[x] = {:.6f}\nM[y] = {:.6f}\nD[y] = {:.6f}\nr[xy] = {:.6f}'
          .format(Mx_t, Dx_t, My_t, Dy_t, Rxy_t))

    print('Точечные значения характеристик:')
    print('MO[x] = {:.6f}\nD[x] = {:.6f}\nM[y] = {:.6f}\nD[y] = {:.6f}\nr[xy] = {:.6f}'
          .format(Mx, Dx, My, Dy, Rxy))

    print('Интервальные значения характеристик:')
    StStudyDSV.interval_estimates(x, y, Mx, Mx_t, Dx, Dx_t, My, My_t, Dy, Dy_t, Rxy, Rxy_t)

    # проверка гипотез
    # Проверка гипотезы о равенстве статистических средних значений и дисперсий
    import StatisticalStudy.HypothesisTesting as HT
    HT.test(Mx, Mx_t, My, My_t, Dx, Dx_t, Dy, Dy_t, Rxy, Rxy_t, len(x))




