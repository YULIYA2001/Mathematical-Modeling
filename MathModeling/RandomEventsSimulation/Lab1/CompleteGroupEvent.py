import numpy as np
import GeneratorBSV as Generator


def simulation_complete_group_events(p=[0.1, 0.3, 0.6], count=10**6):
    if not 1 - 0.1**10 <= sum(p) <= 1 + 0.1**10:
        print('sum(p[i]) != 1   ', sum(p))
        exit(0)

    #print('p ', p)

    k = len(p)
    l = np.zeros(k+1)
    for i in range(k):
        l[i+1] = l[i] + p[i]
    l = l[1:]

    print('\nl = ', l)

    rand_nums_generator = Generator.multiplicative_congruential_method(count)
    rand_num = next(rand_nums_generator)
    print('СЧ: ', rand_num)

    interval_count = np.zeros(k)
    for i in range(k):
        if rand_num < l[i]:
            result = i
            print(i)
            interval_count[i] += 1
            break

    if count == 10**6:      # проверка работы

        for num in rand_nums_generator:
            for i in range(k):
                if num < l[i]:
                    interval_count[i] += 1
                    break

        print('на 10^6 СЧ: p = ', interval_count / 10**6)

        np.set_printoptions(suppress = True, precision = 6, floatmode = "fixed")
        print('\u0394 = ', np.array( 
            [ abs( p[i] - interval_count[i] / 10**6 ) for i in range(k)]     ))

    return result, np.array(interval_count / 10**6)
