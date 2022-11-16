import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sympy import diff, integrate, symbols, lambdify


def T(alpha, k):
    ''' таблица Стьюдента? '''
    return stats.t.ppf(alpha, k)

def table_for_confidence_interval_M(level, m_dots, eps, M_t):
    '''Таблица зависимости доверительного интервала MO от уровня значимости'''
    print("Уровень значимости   Доверительный интервал МО   Накрывает MO?")
    for i in range(3):
        print("%10.2f" % level[i], 
              "           [{:.4f}, {:.4f}]".format(m_dots-eps[i], m_dots+eps[i]),
              "          ", ('Да' if m_dots-eps[i] <= M_t <= m_dots+eps[i] else 'Нет'))

def MO_intervals(n, m_dots, D_dots, D_t):
    '''
            Поиск осн. параметров для интервальной оценки МО
        n - кол-во СЧ Y
        m_dots, D_dots, D_t  - см. ф. evaluate()
        return level, m_dots, eps, eps_D  - см. ф. table_for_confidence_interval_M()
    '''
    # значения альфа - уровень значимости
    level = [0.1, 0.05, 0.01]
    # квантили по таблице Стьюдента для степеней свободы n - 1 = 19 для alpha [0.1, 0.05, 0.01]
    # Student => t(1 - alpha/2, k)
    level_value_Student = [T((0.95), n-1), T((0.975), n-1), T((0.995), n-1)]
    eps = [(t * (D_dots / n)**0.5) for t in level_value_Student]

    # квантили для стандартного нормального распределения для level = [0.1, 0.05, 0.01] 
    level_value_normal = [1.64, 1.96, 2.58]
    eps_D = [(u * (D_t / n)**0.5) for u in level_value_normal]

    return level, m_dots, eps, eps_D




def X2(alpha, k):
    '''  таблица Хи-квадрат? '''
    return stats.chi2.ppf(alpha, k)

def table_for_confidence_interval_D(level, D_dots, l_border, r_border, D_t):
    '''Таблица зависимости доверительного интервала для дисперсии от уровня значимости'''
    print("Уровень значимости   Доверительный интервал D   Накрывает D?")
    for i in range(3): 
        print("%10.2f" % level[i], 
              "           [{:.4f}, {:.4f}]".format(l_border[i], r_border[i]),
              "          ", ('Да' if l_border[i] <= D_t <= r_border[i] else 'Нет'))

def D_intervals(Y, n, m_dots, D_dots, m_t):
    ''' 
            Поиск осн. параметров для интервальной оценки дисперсии
        Y - СЧ Y 
        n - кол-во СЧ Y
        m_dots, D_dots, m_t  - см. ф. evaluate()
        return level, m_dots, eps, eps_D  - см. ф. table_for_confidence_interval_D()
    '''
    level = [0.1, 0.05, 0.01]
    # квантили по таблице хи-квадрат для степеней свободы n-1 для alpha/2 и 1-alpha/2
    # где alpha => [0.1, 0.05, 0.01]
    X2_for_alpha_div_2 = [X2(0.95, n-1), X2(0.975, n-1), X2(0.995, n-1)] 
    X2_for_1_minus_alpha_div_2 = [X2(0.05, n-1), X2(0.025, n-1), X2(0.005, n-1)] 
    l_border = [( (n-1) * D_dots / X2_for_alpha_div_2[i] ) for i in range(3)]
    r_border = [( (n-1) * D_dots / X2_for_1_minus_alpha_div_2[i] ) for i in range(3)]

    S1_sq = 0
    for x in Y:
        S1_sq += (x - m_t) ** 2 
    S1_sq /= n
    X2_for_alpha_div_2_M = [X2(0.95, n), X2(0.975, n), X2(0.995, n)] 
    X2_for_1_minus_alpha_div_2_M = [X2(0.05, n), X2(0.025, n), X2(0.005, n)]
    l_border_M = [( n * S1_sq / X2_for_alpha_div_2_M[i] ) for i in range(3)]
    r_border_M = [( n * S1_sq / X2_for_1_minus_alpha_div_2_M[i] ) for i in range(3)]

    return level, D_dots, l_border, r_border, l_border_M, r_border_M


# Интервальные оценки - основной метод
def evaluate(Y, m_dots, D_dots, m_t, D_t):
    ''' 
        Y - СЧ Y 
        m_dots - точечная оценка МО СВ Y
        D_dots - точечная оценка дисперсии СВ Y 
        m_t - теоретическое значение МО СВ Y
        D_t - теоретическое значение дисперсии СВ Y
        return - массивы с параметрами для построения интервальных оценок
    '''
    n = len(Y)

    # ----------------------------------------------- MO -----------------------------------------
    level_m, m_dots, eps, eps_D = MO_intervals(n, m_dots, D_dots, D_t)

    print('Доверительный интервал для оценки МО={:.4f} СВ при неизвестной D={:.4f} для различных '\
       'уровней значимости:'.format(m_dots, D_dots))
    table_for_confidence_interval_M(level_m, m_dots, eps, m_t)

    print('\nДоверительный интервал для оценки МО={:.4f} СВ при известной D={:} для различных уровней '\
         'значимости:'.format(m_dots, D_t))
    table_for_confidence_interval_M(level_m, m_dots, eps_D, m_t)

    # -------------------------------------------- Дисперсия ---------------------------------------
    level, D_dots, l_border, r_border, l_border_M, r_border_M = D_intervals(Y, n, m_dots, D_dots, m_t)
    
    print('\n\nДоверительный интервал для оценки D={:.4f} СВ при неизвестном M={:.4f} для различных '\
       'уровней значимости:'.format(D_dots, m_dots))
    table_for_confidence_interval_D(level, D_dots, l_border, r_border, D_t)

    print('\nДоверительный интервал для оценки D={:.4f} СВ при известном M={:} для различных уровней '\
         'значимости:'.format(D_dots, m_t))
    table_for_confidence_interval_D(level, D_dots, l_border_M, r_border_M, D_t)

    return [ [level_m, eps, eps_D], [level, l_border, r_border, l_border_M, r_border_M] ]