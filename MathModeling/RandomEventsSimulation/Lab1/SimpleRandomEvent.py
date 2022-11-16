import GeneratorBSV as Generator

# Задание 1 Имитация простого случайного события
def simulation_simple_random_event(p):
    rand_nums_generator = Generator.multiplicative_congruential_method()
    rand_num = next(rand_nums_generator)

    #p = float(input('Введите вероятность 0..1:   '))
    print('\nСЧ  ', rand_num)

    true_count = 0

    if rand_num <= p:
        result = True
        print(True)
        true_count += 1
    else:
        result = False
        print(False)

    for i in rand_nums_generator:
        if i <= p:
            true_count += 1

    p_10_6 = true_count / 10**6
    print('на 10^6 СЧ:  p = ', p_10_6)

    print('\u0394 = {:.6f}'.format( abs(p - p_10_6) ))

    return result, p_10_6

    
