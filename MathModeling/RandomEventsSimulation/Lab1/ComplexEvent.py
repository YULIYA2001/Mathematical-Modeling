import numpy as np
import GeneratorBSV as Generator


# Задание 2
def simulation_complex_event(p=[0.1, 0.1, 0.8]):
    k = len(p)

    rand_nums_generator = Generator.multiplicative_congruential_method()
    rand_nums = [next(rand_nums_generator) for i in range(k)]
    print('\nСЧ: ', rand_nums)
    #print ('   p     ', p)

    result = np.zeros(k)
    true_count = np.zeros(k)

    for i in range(k):
        if rand_nums[i] <= p[i]:
            result[i] = True
            true_count[i] += 1
        else:
            result[i] = False

    print(result.astype(np.bool_))

    j = 0
    for i in rand_nums_generator:
        if i <= p[j]:
            true_count[j] += 1
        j = (j + 1) % k

    true_count = [true_count[i] / 10**6 * k for i in range(k)]
    print('на 10^6 CЧ: р = ', true_count)

    np.set_printoptions(suppress = True, precision = 6, floatmode = "fixed")
    print('\u0394 = ', np.array( 
        [ abs( p[i] - true_count[i] ) for i in range(k)]   ))

    return result.astype(np.bool_), np.array(true_count)
