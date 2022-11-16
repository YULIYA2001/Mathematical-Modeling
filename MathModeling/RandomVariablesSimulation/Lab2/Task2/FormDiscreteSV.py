import Generator.GeneratorBSV as GeneratorBSV


def calculate_Y(p, Y_count, f):
    ''' 
            Формирование дискретной СВ
        p - вероятность успеха испытания
        Y_count - количество СЧ в Y для генерации
        f - Функция вероятности - Геометрическое распределение
        return Y, X - отсортированные масивы СВ Y и X
    '''

    # генерация СЧ Х
    x_generator = GeneratorBSV.multiplicative_congruential_method(count=Y_count)

    # поиск СЧ Y по формуле у = G_-1(x)
    Y = []
    X = []
    for x in x_generator:
        i, delta = 1, p
        while x >= delta:
            #print('delta={:} yi={:}'.format(delta, i))
            delta += p*(1-p)**i
            i += 1
        #print('delta={:} yi={:}'.format(delta, i))
        #print('x={:} y={:}\n'.format(x, i))
        X.append(x)
        Y.append(i)
        
    #print('X = {:} \nY = {:} \n'.format(X, Y))

    X.sort()
    Y.sort()

    #print('X = {:} \nY = {:} \n'.format(X, Y))

    if Y_count <= 100:
        print('\nСгенерированная последовательность из {:} дискретных СВ\nY = ['.format(Y_count), end='')
        setY = set(Y)
        for y in setY:
            print('{:}({:})'.format(y, Y.count(y)), end=' ')
        print(']')

    else:
        print('\nСгенерирована последовательность из {:} дискретных СВ Y.'.format(Y_count))

    return Y, X
