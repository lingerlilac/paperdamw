import numpy as np
import matplotlib.pyplot as plt
import Queue
import math
import random

np.random.seed(123)

nobs = 1
loops = 100
count = 10000
I0 = '0101'
I1 = '1010'
I2 = '1001'
# I_prev='0101'
r = 1.05
lambda_arrival = [0.2, 0.4, 0.4, 0.9]
lambda_channel = [0.2, 0.2, 0.2, 0.2]
q = [0, 0, 0, 0]
q[0] = Queue.Queue(maxsize=count)
q[1] = Queue.Queue(maxsize=count)
q[2] = Queue.Queue(maxsize=count)
q[3] = Queue.Queue(maxsize=count)
w = [0, 1, 2]
p = {w[0]: I0, w[1]: I1, w[2]: I2}

q_tran = [0, 0, 0, 0]
q_del = [0, 0, 0, 0]
q_arr = [0, 0, 0, 0]
q_tran_record = [[0], [0], [0], [0]]
q_arr_record = [[0], [0], [0], [0]]
q_size_record = [[0], [0], [0], [0]]


def clear_queue():
    for i in range(0, 4):
        while q[i].empty() is False:
            q[i].get()

next_arrival = [0, 0, 0, 0]
channel_active = [0, 0, 0, 0]


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
    for m in range(0, 4):
        while q[m].empty() is False:
            q[m].get()
# end of queues destroy


def Variable_clear():
    for m in range(0, 4):
        q_tran[m] = 0
        q_arr[m] = 0


for i in range(1, loops):
    # print('i',i)
    for m in range(0, 4):
        next_arrival[m] = 2
        channel_active[m] = 2
        lambda_arrival[m] *= r
    temp = [0, 0, 0, 0]
    for n in range(1, count + 1):
        # print(n,'n')
        C_current = [0, 0, 0, 0]
        for m in range(0, 4):
            if channel_active[m] == n:
                channel_active[m] = Channel(m, n)
                # print('n1',n,i)
                C_current[m] = 1
                temp[m] += 1
# begin of schedule
        w[0] = q[1].qsize() * C_current[1] + q[3].qsize() * C_current[3]
        w[1] = q[0].qsize() * C_current[0] + q[2].qsize() * C_current[2]
        w[2] = q[0].qsize() * C_current[0] + q[3].qsize() * C_current[3]
        index = w.index(max(w))
        for m in range(0, 4):
            if p[index][m] == '1' and C_current[m] == 1 and q[m].empty() is False:
                q[m].get()
                q_tran[m] += 1

        for m in range(0, 4):
            if next_arrival[m] == n and q[m].qsize() < count:
                q[m].put(1)
                q_arr[m] += 1
                next_arrival[m] = Arrival(m, n)
    for m in range(0, 4):
        q_tran_record[m].append(q_tran[m])
        q_arr_record[m].append(q_arr[m])
        q_size_record[m].append(q[m].qsize())
        # print(q[m].qsize())
# q_tran_record record the successuffly transmitted packets
# q_arr_record record the arrival packets
# q_size_record record the queue size at the end of every time slots.
#

    Queues_clear()
    Variable_clear()
    for m in range(0, 4):
        temp[m] = 0

# end of schedule

# begin destroy of the queues

plt.figure(1)
plt.xlim(0, loops)
plt.ylim(0, count)

x = [0, 0, 0, 0]
for m in range(0, 4):
    x[m] = np.linspace(0, len(q_size_record[m]), len(q_size_record[m]))
    plt.plot(x[m], q_size_record[m], label=m)

plt.legend()
plt.show()
