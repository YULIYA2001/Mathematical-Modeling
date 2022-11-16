#################################################################################            
#                                 Дискретная СВ                                 #
#################################################################################

import numpy as np

import Task2.StatisticalStudy2 as StatisticalStudy
import Task2.StStudyElements.HypothesisTesting2 as HypothesisTesting


#n >= 1
A0 = 1

def M(p):
    ''' Математическое ожидание '''
    return 1 / p

def D(p):
    ''' Дисперсия '''
    q = 1-p
    return q / p**2

def G(n, p):
    ''' 
            Ступенчатая функция распределения дискретной СВ - Геометрическое распределение 
        n - 1, 2, ...  (целое) число испытаний Бернулли с вероятностью успеха р вплоть до появления 
            первого успеха (включая также и первый успех)
        p - вероятность успеха испытаний
    '''
    q = 1 - p
    return 1 - q**(n)

def f(n, p):
    '''
            Функция вероятности - Геометрическое распределение 
        n - 1, 2, ...  (целое) число испытаний Бернулли с вероятностью успеха р
        p - вероятность успеха испытаний
    '''
    q = 1 - p
    return p * q**(n-1)

def start_st_study(p, Y_count=50):
    ''' статистическое исследование: построение гистограммы, точечной и интервальной оценок '''
    print_params = StatisticalStudy.execute(f, G, M, D, p, Y_count, A0=1)
    return print_params


def check_hypothesis(p, n=0, kind='A'):
    ''' проверка гипотезы o распределении дискретной СВ Y критерии Пирсона, Колмогорова 
        kind: A - все 2 критерия, P - Пирсона, K - Колмогорова
    '''
    print_params =  HypothesisTesting.test(G, p, f, n, kind)
    return print_params


def execute_task2():
    check_hypothesis()
    #statistical_study()