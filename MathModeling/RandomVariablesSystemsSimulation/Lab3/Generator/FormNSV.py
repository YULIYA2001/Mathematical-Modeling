import numpy as np
import matplotlib.pyplot as plt

import Generator.NeymanMethod as NM


def form_values(f1x, f2yx, f1x_max, f2yx_max, ax, bx, ay, by, count=10000):
    ''' 
        Формирование зависимых СВ Х и У + "график правильности" их генерации

        f1x, f2yx - функция плотности СВ Х и условная функция плотности СВ У
        f1x_max - число - мах значение f1x(x)
        f2yx_max - функция от х - мах значение f2yx(y|x) при фиксированном х
        (ax, bx), (ay, by) - интервалы распределения СВ Х и У соотв.
        count - кол-во точек генерации - |X| и |У| 

        return X, Y - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.
    '''
    X, Y, fX, fY = NM.calculate_SV(f1x, f2yx, f1x_max, f2yx_max, ax, bx, ay, by, count) 

    # график для сгенерированных точек
    plot_SV_dots_3D(f1x, f2yx, X.copy(), Y, fX, fY, ax, bx, ay, by)

    return X, Y


def plot_SV_dots_3D(f1x, f2yx, x, y, fx, fyx, ax, bx, ay, by):
    '''
        3D график проекций функций плотностей и сгенерированных точек

        f1x, f2yx - функция плотности СВ Х f(x) и условная функция плотности СВ У f(y|x)
        х, у - массивы псевдослучайных зависимых величин с заданными ф. плотности и интервалами распр.
        fx, fyx - массивы точек, соответствовавшие х, у при их генерации
        (ax, bx), (ay, by) - интервалы распределения СВ Х и У соотв.
    '''

    # генерация точек для теоретических графиков
    step = np.pi/100
    X = np.arange(ax, bx+step, step)
    Y = np.arange(ay, by+step, step)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # практический - точки генерации, проекция на ось X
    ax.scatter(x, fx, marker=".", c="r", zs=np.pi/2, zdir='y', s=0.5, label='Проекция сгенер. точек на Ох')
    # теоретический, проекция на ось X
    ax.plot(X, f1x(X), 'black', zs=np.pi/2, zdir='y', label='f(x)', linewidth=2)
    
    # теоретический, проекция на ось Y
    ax.plot(Y, f2yx(Y, X[0]), 'grey', zs=0, zdir='x', linewidth=1, label='f(y|x)')
    for i in range(1, len(X)):
        ax.plot(Y, f2yx(Y, X[i]), 'grey', zs=0, zdir='x', linewidth=1)
    # практический - точки генерации, проекция на ось Y
    ax.scatter(y, fyx, marker=".", c="b", zs=0, zdir='x', s=0.5, label='Проекция сгенер. точек на Оу')

    
    fig = plt.gcf()
    fig.suptitle("Проекции функций f(x) и f(y|x) и точки генерации Х и Y\n |X| = |Y| = {:} точек"
                 .format(len(x)), fontsize=10)
    fig.canvas.set_window_title('График 1. Проверка правильности генерации (метод аналитических преобразований и метод Неймона) точек')
    ax.legend(loc='best', fontsize=8)
    plt.show()
