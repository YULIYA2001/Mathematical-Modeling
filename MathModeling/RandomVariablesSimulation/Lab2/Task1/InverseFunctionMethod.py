import numpy as np
import Generator.GeneratorBSV as GeneratorBSV


def calculate_Y(G, G_1, Y_count=50):
    ''' 
        G - lambda-выражение; исходная функция распределения G(y)
        G_1 - lambda-выражение; функция, обратная G(y)
        G_str - строкавая запись функции G
        Y_count - количество точек генерации СВ Y
        return y_nums - массив псевдоСВ Y с функцией распределения G(y)
               x_nums - массив псевдоСВ X с функцией распределения G-1(x), созданный генератором БСВ
    '''

    # генерация СЧ Х
    x_generator = GeneratorBSV.multiplicative_congruential_method(count=Y_count)

    # поиск СЧ Y по формуле у = G_-1(x)
    y_nums = []
    x_nums = []
    for x in x_generator:
        y_nums.append(G_1(x))
        x_nums.append(x)

    # сортировка X, Y
    x_nums.sort()
    y_nums.sort()

    y_nums = np.array(y_nums)
    if Y_count <= 100:
        print('\nСгенерированная последовательность из {:} СВ \nY = \n'.format(Y_count), y_nums)
    else:
        print('\nСгенерирована последовательность из {:} СВ Y.'.format(Y_count))
        
    return y_nums, x_nums