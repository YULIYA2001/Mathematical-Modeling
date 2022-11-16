#################################################################################            
#                                Непрерывная СВ                                 #
#################################################################################

import sympy, math, scipy
import numpy as np
import Task1.StatisticalStudy as StatisticalStudy

from sympy import symbols, solve, simplify, lambdify, diff
import Task1.StStudyElements.HypothesisTesting as HypothesisTesting

from pynverse import inversefunc



def G1(y, L):
    ''' функция распределения '''
    return 1 - np.e ** (-L*y)       

def G1_str(L):
   ''' строковое представление '''
   return 'G(y) = 1 - e^(-{:}*y)'.format(L)

 

def G2(y):
    ''' функция распределения '''
    return y**2
  
def G2_str():
   ''' строковое представление '''
   y = symbols('y')
   return 'y = ' + str( G2(y) )



def start_st_study(G, G_str, Y_count=50):
    ''' '''
    # обратная функция поиск
    x, y = symbols('x, y') 
    expr = x - G(y) 
    x, y = x, y
    str_G_1 = np.abs(solve(expr, y)[0])
    G_1 = lambdify(x, str_G_1)
    
    #G_1 = inversefunc(G)

    print_params = StatisticalStudy.execute(G, G_1, Y_count, G_str)
    return print_params


def check_hypothesis(G, G_str, kind='A', n=0):
    ''' проверка гипотезы o распределении СВ Y критерии Пирсона, Колмогорова, Мизеса 
        kind: A - все 3 критерия, P - Пирсона, K - Колмогорова, M - Мизеса
    '''

    # обратная функция поиск
    x, y = symbols('x, y') 
    expr = x - G(y) 
    x, y = x, y
    str_G_1 = np.abs(solve(expr, y)[0])
    G_1 = lambdify(x, str_G_1)
    
    #G_1 = inversefunc(G)
    # print(G(1))
    # print(G_1(G(1)))

    print_params = HypothesisTesting.test(G, G_1, kind, n)
    return print_params


def create_func(G_str):
    ''' convert str to lambda'''
    G_res = G_str
    G_res = G_res.replace('^', '**')
    G_res = G_res.replace('e', 'np.e')
    G_res = G_res.replace('pi', 'np.pi')

    '''
    G_res = G_res.replace('arccos', 'np.arccos')
    G_res = G_res.replace('arcsin', 'np.arcsin')
    G_res = G_res.replace('arctan', 'np.arctan')
    G_res = G_res.replace('cos', 'np.cos')
    G_res = G_res.replace('sin', 'np.sin')
    G_res = G_res.replace('tan', 'np.tan')
    G_res = G_res.replace('sqrt', 'np.sqrt')
    G_res = G_res.replace('sqrt', 'math.sqrt')
    G_res = G_res.replace('log10', 'np.log10')
    G_res = G_res.replace('log2', 'np.log2')
    G_res = G_res.replace('log', 'np.log')
    G_res = G_res.replace('ln', 'np.log')
    G_res = G_res.replace('abs', 'np.abs')
    G_res = G_res.replace('e', 'np.e')
    G_res = G_res.replace('pi', 'np.pi')
    '''

    try:
        print('G(y) = ' + G_res)

        # не функция от y
        if G_res.find('y') == -1:
            return 'error'

        G = lambda y: eval(G_res)
        # test = G(4)
    except:
        print('Wrong G(y) input')
        return 'error'
   
    return G



def execute_task1():
    
    G_str = 'y**(1/2)'
    G = create_func(G_str)
    if G == 'error':
        return

    G = lambda y: G2(y)
    G_str = G2_str()

    #check_hypothesis(G, G_str)
    start_st_study(G, G_str)

    '''
    # G(y) = y^2
    G = lambda y: G2(y)
    G_str = G2_str()
    check_hypothesis(G, G_str)
    start_st_study(G, G_str)

    # G(y) - по экспоненциальному закону
    G = lambda y: G1(y, 4)
    G_str = G1_str(4)
    check_hypothesis(G, G_str)
    start_st_study(G, G_str)
    '''
