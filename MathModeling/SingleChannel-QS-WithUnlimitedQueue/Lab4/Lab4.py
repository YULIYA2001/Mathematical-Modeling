import math
import random
import matplotlib.pyplot as plt
import numpy as np

import Generator.GeneratorBSV as Generator


def final_p_t(i, alpha):
    ''' теоретические финальные вероятности '''
    return alpha**i * (1-alpha)

def L_CMO_t(alpha):
    ''' Среднее число заявок в СМО (теоретическoе)'''
    return alpha / (1-alpha)

def L_Q_t(alpha):
    ''' Среднее число заявок в очереди (теоретическoе)'''
    return alpha**2 / (1-alpha)

def T_CMO_t(alpha, L):
    ''' Среднее время пребывания заявки в СМО (теоретическoе)'''
    return alpha / (L * (1-alpha))

def T_Q_t(alpha, L):
    ''' Среднее время пребывания заявки в очереди (теоретическoе)'''
    return alpha**2 / (L * (1-alpha))

def init_data():
    L = 2
    Mu = 3
    alpha = L / Mu
    return L, Mu, alpha


def final_p(CMO_state):
    ''' Практические значения финальных вероятностей '''
    n = max(CMO_state)+1
    final_p = np.zeros(n)
    for i in range(n):
        final_p[i] = ( CMO_state.count(i) / len(CMO_state) )
    return final_p


def plot_final_p(final_p_teor, final_p_empirical, Tn, dt):
    ''' График практический и теоретический для финальных вероятностей '''
    fig, ax = plt.subplots(figsize=(5, 4))
    x = [i for i in range(len(final_p_teor))]
    ax.scatter(x, final_p_teor, s=10, c="r", marker="o", edgecolors="r", label=' Теоретические')
    ax.scatter(x, final_p_empirical, s=10, c="b", marker="o", edgecolors="b", label=' Эмпирические')
    ax.set_xlabel("m", fontsize=8)
    ax.set_ylabel("P_final", fontsize=8)
    plt.title('Финальные вероятности \n(время наблюдения = {:}ч, шаг = {:}ч)'.format(Tn, dt), fontsize=10)
    plt.grid(True)
    ax.legend(loc='best', fontsize=8)
    fig = plt.gcf()
    fig.canvas.set_window_title('График 1. СМО с неограниченной очередью')
    plt.show()



def task3_about_trains(dt=0.01, Tn=10000, is_print=False):
    ''' Задача про поезда - СМО с неограниченной очередью '''
    L, Mu, alpha = init_data()

    time_arrive = 0
    t_train = []

    while time_arrive <= Tn:
        # генерация прибытия жд состава
        r = next(Generator.multiplicative_congruential_method())
        tau_train = -1/L * math.log(r) #random.random())
        time_arrive += tau_train
        t_train.append(time_arrive)


    time_now = 0
    time_arrive = 0
    full_time_process = 0
    state = []
    Q = []
    wait_time, stay_time = [], []

    i = 0
    processing = False
    
    # обслуживание жд состава
    while time_now <= Tn:
        if full_time_process <= time_now:
            processing = False

        if i < len(t_train) and time_now < t_train[i]:
            if processing:
                state.append(len(Q)+1)
            else:
                if len(Q):
                    # генерация времени обслуживания
                    tau_process = -1/Mu * math.log(random.random())
                    q_t_train = Q.pop(0)
                    if full_time_process > q_t_train:
                        wait_time.append(full_time_process - q_t_train)
                    else:
                        wait_time.append(0)
                    stay_time.append(wait_time[-1] + tau_process)
                    full_time_process = max(full_time_process, q_t_train) + tau_process
                    processing = True
                    state.append(len(Q)+1)
                else:
                    state.append(len(Q))

            time_now += dt
            continue

        if processing:
            if i < len(t_train):
                Q.append(t_train[i])
                i += 1
            if i < len(t_train) and t_train[i] > time_now:
                time_now += dt
                state.append(len(Q)+1)
            continue

        if len(Q):
            if i < len(t_train):
                Q.append(t_train[i])
                i += 1
            # генерация времени обслуживания
            tau_process = -1/Mu * math.log(random.random())
            q_t_train = Q.pop(0)
            if full_time_process > q_t_train:
                wait_time.append(full_time_process - q_t_train)
            else:
                wait_time.append(0)
            stay_time.append(wait_time[-1] + tau_process)
            full_time_process = max(full_time_process, q_t_train) + tau_process
            processing = True
            state.append(len(Q)+1)
            continue

        if i < len(t_train):
            # генерация времени обслуживания
            tau_process = -1/Mu * math.log(random.random())
            if full_time_process > t_train[i]:
                wait_time.append(full_time_process - t_train[i])
            else:
                wait_time.append(0)
            stay_time.append(wait_time[-1] + tau_process)
            full_time_process = max(full_time_process, t_train[i]) + tau_process
            i += 1
            processing = True
            if i < len(t_train) and t_train[i] > time_now:
                time_now += dt
                state.append(len(Q)+1)


    params = study(stay_time, wait_time, state, alpha, L, Mu, Tn, dt, is_print)
    return params



def study(stay_time, wait_time, state, alpha, L, Mu, Tn, dt, is_print):
        T_CMO = sum(stay_time) / len(stay_time)
        T_Q = sum(wait_time) / len(wait_time)
        L_CMO = sum(state) / len(state)
        wait_state = [el-1 for el in state if el!=0]
        L_Q = sum(wait_state) / len(state)
        f_p = final_p(state)

        f_p_t = []
        for i in range(max(state)+1):
            f_p_t.append( final_p_t(i, alpha) )


        if is_print:
            plot_final_p(f_p_t, f_p, Tn, dt)

            print('Обозначение: теор.знач. - практич.знач. (дельта)')
            print('Финальные вероятности состояний СМО:')
            for i in range(max(state)+1):
                print('p[{:}] = {:.6f} - {:.6f}  ({:.6f})'.format( i, f_p_t[i], f_p[i], abs(f_p_t[i]-f_p[i]) ))
            print('Среднее число составов, связанных с горкой (в СМО):\n L_смо = {:.4f} - {:.4f}  ({:.4f})'
                  .format( L_CMO_t(alpha), L_CMO, abs(L_CMO_t(alpha)-L_CMO) ))
            print('Среднее число составов в очереди:\n L_q = {:.4f} - {:.4f}  ({:.4f})'
                  .format( L_Q_t(alpha), L_Q, abs(L_Q_t(alpha)-L_Q) ))
            print('Среднее время пребывания состава в СМО:\n T_смо = {:.4f} - {:.4f}  ({:.4f})'
                  .format( T_CMO_t(alpha, L), T_CMO, abs(T_CMO_t(alpha,L)-T_CMO) ))
            print('Среднее время пребывания состава в очереди:\n T_q = {:.4f} - {:.4f}  ({:.4f})'
                  .format( T_Q_t(alpha, L), T_Q, abs(T_Q_t(alpha, L)-T_Q) ))

        return f_p, T_CMO, T_Q, L_CMO, L_Q, f_p_t, T_CMO_t(alpha, L), T_Q_t(alpha, L), L_CMO_t(alpha), L_Q_t(alpha)



def plot_diff_dt(final_p, diff_params, const_param, labels):
    ''' График практический (несколько, разный шаг) и теоретический для финальных вероятностей '''
    final_p_t, final_p_e = zip(*final_p)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i in range(len(final_p_e)):
        x = [i for i in range(len(final_p_e[i]))]
        plt.plot(x, final_p_e[i], '-', label=labels[0].format(diff_params[i]))#'dt = {:}ч'.format(diff_params[i]))

    max_len, max_i = len(final_p_t[0]), 0 
    for i in range(1, len(final_p_t)):
        if len(final_p_t[i]) > max_len:
            max_len = len(final_p_t[i])
            max_i = i
    x = [i for i in range(max_len)]
    plt.plot(x, final_p_t[max_i], '--k', label=labels[1].format(const_param))#'Теор. Тн = {:}ч'.format(const_param))

    ax.set_xlabel("m", fontsize=8)
    ax.set_ylabel("P_final", fontsize=8)
    plt.title('Финальные вероятности', fontsize=10)
    plt.grid(True)
    ax.legend(loc='best', fontsize=8)
    fig = plt.gcf()
    fig.canvas.set_window_title('График. СМО с неограниченной очередью (зависимость)')
    plt.show()

    
if __name__ == "__main__":
    dt_ = [10, 1, 0.1, 0.01]
    Tn = 1000
    final_p_params = []
    for i in range(len(dt_)):
        params = task3_about_trains(dt=dt_[i], Tn=Tn)
        final_p_params.append( (params[5], params[0]) )
    plot_diff_dt(final_p_params, dt_, Tn, ['dt = {:}ч', 'Теор. Тн = {:}ч'])


    Tn_ = [10, 100, 1000, 10000]
    dt = 0.1
    final_p_params = []
    for i in range(len(Tn_)):
        params = task3_about_trains(dt=dt, Tn=Tn_[i])
        final_p_params.append( (params[5], params[0]) )
    plot_diff_dt(final_p_params, Tn_, dt, ['Tn = {:}ч', 'Теор. dt = {:}ч'])

    task3_about_trains(dt=0.01, Tn=1000, is_print=True)
