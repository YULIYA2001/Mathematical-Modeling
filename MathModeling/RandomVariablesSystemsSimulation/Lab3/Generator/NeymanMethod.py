import random
import numpy as np
import Generator.GeneratorBSV as Generator


def calculate_SV(f1x, f2yx, f1x_max, f2yx_max, ax, bx, ay, by, count):
    '''
        Формирование зависимых СВ Х и У методом Неймана, расширенным на 2 зависимые СВ

        f1x, f2yx - функция плотности СВ Х и условная функция плотности СВ У
        f1x_max - число - мах значение f1x(x)
        f2yx_max - функция от х - мах значение f2yx(y|x) при фиксированном х
        (ax, bx), (ay, by) - интервалы распределения СВ Х и У соотв.
        count - кол-во точек генерации - |X| и |У| 

        return 
          x_nums, y_nums - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.
          fx_nums, fy_nums - массивы точек, соответствовавшие x_nums, y_nums при генерации
    '''
    x_nums, y_nums = [], []
    fx_nums, fy_nums = [], []
    
    while len(x_nums) != count:
        x, fx = [x for x in Generator.multiplicative_congruential_method(count=2)]
        #x, fx = random.random(), random.random()
        x = ax + x*(bx-ax)
        fx = fx * f1x_max
        if f1x(x) < fx:
            continue

        while True:
            y, fy = [x for x in Generator.multiplicative_congruential_method(count=2)]
            y = ay + y*(by-ay)
            fy = fy * f2yx_max(x)
            if f2yx(y, x) >= fy:
                break

        x_nums.append(x)
        y_nums.append(y)
        fx_nums.append(fx)
        fy_nums.append(fy)
        
    return x_nums, y_nums, fx_nums, fy_nums