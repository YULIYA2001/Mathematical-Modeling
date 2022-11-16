import random
import numpy as np
import Generator.GeneratorBSV as Generator


def calculate_SV(P, count):
    '''
        Формирование дискретных СВ Х и У

        P - матрицa вероятностей двумерной ДСВ
        count - кол-во точек генерации - |X| и |У| 

        return 
          x_nums, y_nums - массивы псевдослучайных величин 
          fx_nums, fy_nums - массивы точек, соответствовавшие x_nums, y_nums при генерации
    '''
    x_nums, y_nums = [], []

    n = len(P)
    m = len(P[0])

    q = np.zeros(n)
    for i in range(n):
        q[i] = sum(P[i])

    l = np.zeros(n)
    l[0] = q[0]
    for i in range(1,n):
        l[i] = l[i-1] + q[i] 

    r = np.zeros((n,m))
    for k in range(n):
        r[k][0] = P[k][0]
        for i in range(1,m):
            r[k][i] = r[k][i-1] + P[k][i]

    
    while len(x_nums) != count:
        num_x = [x for x in Generator.multiplicative_congruential_method(count=1)]
        #num = random.random()
        
        k = 0
        while num_x > l[k]:
            k += 1
            continue
        x_nums.append(k) 


        rk = r[k].copy()

        while True:
            num_y = [x for x in Generator.multiplicative_congruential_method(count=1)]

            s = 0
            while num_y > rk[s]:
                s += 1
                if s == len(rk):
                    break
                continue

            if s != len(rk):
                break

        y_nums.append(s)
        
    return x_nums, y_nums
