import numpy as np
import matplotlib.pyplot as plt
import Queue
import math
import random

np.random.seed(123)

nobs = 1
loops = 100
loop_count = 1000

r = 1.05
lambda_arrival = [0.031, 0.15, 0.031, 0.15, 0.031, 0.15]
lambda_channel = [0.3, 0.4, 0.3, 0.4, 0.3, 0.4]
q = [0, 0, 0, 0, 0, 0]
for m in range(0, 6):
    q[m] = Queue.Queue(maxsize=loop_count)

q_tran = [0, 0, 0, 0, 0, 0]
q_del = [0, 0, 0, 0, 0, 0]
q_arr = [0, 0, 0, 0, 0, 0]
q_tran_record = [[0], [0], [0], [0], [0], [0]]
q_arr_record = [[0], [0], [0], [0], [0], [0]]
q_size_record = [[0], [0], [0], [0], [0], [0]]
q_avg = [0.0, 0.0]
d_avg = [0.0, 0.0]
K = 1
k1 = [0.2, 0.1]


def clear_queue():
    for i in range(0, 6):
        while q[i].empty() is False:
            q[i].get()

next_arrival = [0, 0, 0, 0, 0, 0]
channel_active = [0, 0, 0, 0, 0, 0]


def Arrival(m, k):
    next_arrival[
        m] = k + int(math.ceil(-math.log(1.0 - random.random()) / lambda_arrival[m]))
    # print(next_arrival[m])
    return next_arrival[m]


def Channel(m, k):
    channel_active[
        m] = k + int(math.ceil(-math.log(1.0 - random.random()) / lambda_channel[m]))
    return channel_active[m]


def Queues_clear():
    for m in range(0, 6):
        while q[m].empty() is False:
            q[m].get()
# end of queues destroy


def Variable_clear():
    for m in range(0, 6):
        q_tran[m] = 0
        q_arr[m] = 0
for i in range(0, loops):
    print(i)
    q_size_history = [0, 0]
    d_histroy = [0.0, 0.0]
    for k in range(1, loop_count):
        if k == 1:
            for m in range(0, 6):
                next_arrival[m] = 2
                channel_active[m] = 2
        ch = [0, 0, 0, 0, 0, 0]
        for m in range(0, 6):
            if channel_active[m] == k:
                ch[m] = 1
                channel_active[m] = Channel(m, k)

        q_choose = [0, 0, 0]
        tmp = 0
        tmp = q[0].qsize() * ch[0] - q[1].qsize() * ch[1]
        if tmp == 0:
            rand = np.random.binomial(1, 0.5, 1)
            if rand[0] == 1:
                q_choose[0] = 1
            else:
                q_choose[0] = 0
        elif tmp > 0:
            q_choose[0] = 0
        else:
            q_choose[0] = 1
        tmp = 0
        if q[2].empty() is False and q[3].empty() is False:
            tmp = q[2].queue[0] * ch[2] - q[3].queue[0] * ch[3]
        elif q[2].empty() is True and q[3].empty() is False:
            tmp = -1
        elif q[2].empty() is False and q[3].empty() is True:
            tmp = 1
        else:
            tmp = 0
        if tmp == 0:
            rand = np.random.binomial(1, 0.5, 1)
            if rand[0] == 0:
                q_choose[1] = 2
            else:
                q_choose[1] = 3
        elif tmp > 0:
            q_choose[1] = 2
        else:
            q_choose[1] = 3
            #
        q_size_history[0] += q[4].qsize()
        q_size_history[1] += q[5].qsize()
        for m in range(0, 2):
            q_avg[m] = float(q_size_history[m]) / float(k)
            d_avg[m] = d_histroy[m] / float(k)
        tmp1 = 0.0
        b = [0.0, 0.0]
        a = k1[0] * lambda_channel[4] * q_avg[0] * d_avg[0] + k1[1] * lambda_channel[5] * q_avg[1] * d_avg[1]
        if a > 0:
            for m in range(0, 2):
             b[m] = float(q[m + 4].qsize() * ch[m + 4]) + K * (k1[m] * lambda_channel[m + 4] * q_avg[m] * d_avg[m]) / a
        else:
            b[0] = b[1] = 0
        if b[0] == b[1]:
            rand = np.random.binomial(1, 0.5, 1)
            if rand[0] == 0:
                q_choose[2] = 4
            else:
                q_choose[2] = 5
        elif b[0] > b[1]:
            q_choose[2] = 4
        else:
            q_choose[2] = 5
        # schedule
        for m in range(0, 3):
            ind = q_choose[m]
            if q[ind].empty() is False and ch[ind] == 1:
                tmp2 = q[ind].get()
                if ind > 3:
                    d_histroy[ind - 4] += tmp2
                q_tran[ind] += 1
        # data arrival
        for m in range(0, 6):
            if next_arrival[m] == k:
                if q[m].full() is False:
                    q[m].put(k)
                else:
                    print('queue', m, 'is full')
                Arrival(m, k)
                q_arr[m] += 1
    for m in range(0, 6):
        q_tran_record[m].append(q_tran[m])
    for m in range(0, 6):
        q[m].queue.clear()
        q_tran[m] = 0
        q_arr[m] = 0
#    for m in range(0, 6):
#        lambda_arrival[m] *= r
    for m in range(0, 6):
        ch[m] = 0
    K += 1
plt.figure(1)

plt.xlim(0, loops)
plt.ylim(0, loop_count)
for m in range(0, 6):
    plt.plot(q_tran_record[m], label = m)
plt.legend()
plt.show()
