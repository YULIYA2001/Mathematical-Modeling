import CompleteGroupEvent as CGE
import GeneratorBSV as Generator
import math

    
def fortune_wheel_for_streamer(games_list={'a': 510, 'b': 45, 'c': 40}):
    """
    while(True):
        name = input('print name or 000 to stop: ')
        if name == '000':
            break
        sum = float(input('print sum: '))

        if name in games_list.keys():
            games_list[name] += sum
        else:
            games_list[name] = sum
    """

    #print('games_list: ', games_list)

    p = list(games_list.values())
    total = math.fsum(p)
    p = [pi / total for pi in p]
    print('\np = ', p)

    result = CGE.simulation_complete_group_events(p, count=1)[0]
    print('Win game: ', list(games_list.keys())[result])

    return list(games_list.keys())[result]



if __name__ == '__main__':
    
    # тестирование равномерности датчика БСВ
    Generator.test_generator()
    # Task 1
    #SRE.simulation_simple_random_event()
    # Task 2
    #CE.simulation_complex_event()
    # Task 3
    #CDE.simulation_complex_dependent_event()
    # Task 4
    #CGE.simulation_complete_group_events()
    # Dop task
    #fortune_wheel_for_streamer()
