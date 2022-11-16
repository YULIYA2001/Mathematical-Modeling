import numpy as np
import GeneratorBSV as Generator


def simulation_complex_dependent_event(Pa, Pba):
    #Pa = 0.973
    #Pba = 0.302
    Pb_a = 1 - Pba

    P_AB = Pa * Pba
    P_nAB = (1 - Pa) * (1 - Pba)

    Pb = P_AB + P_nAB

    P_AnB = Pa + P_nAB - Pb
    P_nAnB = 1 - Pa + P_AB - Pb

    teor = [P_AB, P_nAB, P_AnB, P_nAnB]
    print('\nТеоретические вероятности: ', teor)
    if not 1 - 0.1**10 <=P_AB+P_AnB+P_nAB+P_nAnB <= 1 + 0.1**10:
        print('Smth wrong in ComplexDependentEvent')
        exit(0)

    rand_nums_generator = Generator.multiplicative_congruential_method()#count=10)
    x1 = next(rand_nums_generator)
    x2 = next(rand_nums_generator)
    print('СЧ:  ', x1, '  ', x2)


    variants_count = np.zeros(4)

    if x1 <= Pa:
        if x2 <= Pba:
            result = 0
            print('AB  ', 0)
            variants_count[0] += 1
        else:
            result = 2
            print('~AB  ', 2)
            variants_count[2] += 1
    else:
        if x2 <= Pb_a:
            result = 1
            print('A~B  ', 1)
            variants_count[1] += 1
        else: 
            result = 3
            print('~A~B  ', 3)
            variants_count[3] += 1

    
    for x1 in rand_nums_generator:
        x2 = next(rand_nums_generator)
        if x1 <= Pa:
            if x2 <= Pba:
                variants_count[0] += 1
            else:
                variants_count[2] += 1
        else:
            if x2 <= Pb_a:
                variants_count[1] += 1
            else: 
                variants_count[3] += 1

    variants_count = variants_count / 10**6 *2
    print('на 10^6 СЧ:  p = ', variants_count)

    np.set_printoptions(suppress = True, precision = 6, floatmode = "fixed")
    print('\u0394 = ', np.array( 
        [ abs( teor[i] - variants_count[i] ) for i in range(4)]     ))


    return result, np.array(variants_count), np.array(teor)
