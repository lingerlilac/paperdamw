import numpy as np
import matplotlib.pyplot as plt
import Queue
import datetime
import gc
import random
np.random.seed(123)


count = 5000

I = [[0 for col in range(4)] for row in range(3)]
I[0][1] = 1
I[0][3] = 1
I[1][0] = 1
I[1][2] = 1
I[2][0] = 1
I[2][3] = 1
q = [[0 for col in range(4)] for row in range(3)]
for m in range(0, 3):
    for n in range(0, 4):
        q[m][n] = Queue.Queue(maxsize=count)

channel = [[0 for col in range(4)] for row in range(3)]
channel_lambda = [0.5, 0.5, 0.5, 0.5]
arrival_lambda = [0.002, 0.001, 0.001, 0.002]
w = [[0 for col in range(3)] for row in range(3)]
step = 0.005
loops = int(0.80 / step)
qav = [0.0 for col in range(4)]
dav = [0.0 for col in range(4)]
k = [0.25, 0.15, 0.32, 0.28]
K = 10000.0
schedule_recorded = [[0 for col in range(3)] for row in range(3)]
arrival_count = [[0 for col in range(4)] for row in range(3)]
trans_count = [[0 for col in range(4)] for row in range(3)]
schedule = [[0 for col in range(1)] for row in range(3)]
arrival1 = [[0 for col in range(4)] for row in range(3)]
queue_length = [[0 for col in range(loops)] for row in range(3)]


def Clear_queue():
    for m in range(0, 3):
        for n in range(0, 4):
            while q[m][n].empty() is False:
                q[m][n].get()


delayoflinks = [[0.0 for col in range(4)] for row in range(3)]
for h in range(0, loops):
    print(h)
    for i in range(0, count - 1):
        for m in range(0, 3):
            for n in range(0, 4):
                tmp = np.random.binomial(1, channel_lambda[n], 1)
                channel[m][n] = tmp[0]
        tmp = [0.0 for col in range(4)]
        tmp_1 = 0.0
        for m in range(0, 4):
            tmp_1 += k[m] * float(channel[0][m]) * qav[m] * dav[m]
        for m in range(0, 4):
            if tmp_1 != 0:
                tmp[m] = K * k[m] * \
                    float(channel[0][m]) * qav[m] * dav[m] / tmp_1
            else:
                tmp[m] = 0
        w[0][0] = channel[0][1] * q[0][1].qsize() + channel[0][3] * \
            q[0][3].qsize() + tmp[1] + tmp[3]
        w[0][1] = channel[0][0] * q[0][0].qsize() + channel[0][2] * \
            q[0][2].qsize() + tmp[0] + tmp[2]
        w[0][2] = channel[0][0] * q[0][0].qsize() + channel[0][3] * \
            q[0][3].qsize() + tmp[0] + tmp[3]
        hol = [0 for col in range(4)]
        for m in range(0, 4):
            if q[1][m].empty() is True:
                hol[m] = 0
            else:
                hol[m] = i - q[1][m].queue[0]
        w[1][0] = channel[1][1] * hol[1] + channel[1][3] * hol[3]
        w[1][1] = channel[1][0] * hol[0] + channel[1][2] * hol[2]
        w[1][2] = channel[1][0] * hol[0] + channel[1][3] * hol[3]
        w[2][0] = channel[2][1] * q[2][1].qsize() + channel[2][3] * \
            q[2][3].qsize()
        w[2][1] = channel[2][0] * q[2][0].qsize() + channel[2][2] * \
            q[2][2].qsize()
        w[2][2] = channel[2][0] * q[2][0].qsize() + channel[2][3] * \
            q[2][3].qsize()

        ind = [[0 for col in range(1)] for row in range(3)]
        map1 = [[0 for col in range(4)] for row in range(3)]
        for m in range(0, 3):
            tmp = [0, 1, 2]
            t1 = []
            if w[m].count(max(w[m])) == 3:
                t1 = random.sample(tmp, 1)
                ind[m][0] = t1[0]
            elif w[m].count(max(w[m])) == 2:
                tmp.pop(w[m].index(min(w[m])))
                t1 = random.sample(tmp, 1)
                ind[m][0] = t1[0]
            else:
                ind[m][0] = w[m].index(max(w[m]))

        for m in range(0, 3):
            for n in range(0, 4):
                map1[m][n] = I[ind[m][0]][n]
        for m in range(0, 3):
            for n in range(0, 4):
                if map1[m][n] == 1 and q[m][n].empty() is False and channel[m][n] == 1:
                    if m == 0:
                        qav[n] += float(q[m][n].queue[0]) / float(count)
                    tmp3 = q[m][n].get()
                    queue_length[m][h] += 1
                    delayoflinks[m][n] = float(i - tmp3)
                    if m == 0:
                        dav[n] += delayoflinks[m][n] / float(count)
        for m in range(0, 3):
            for n in range(0, 4):
                tmp = np.random.binomial(1, arrival_lambda[n], 1)
                if tmp[0] == 1:
                    q[m][n].put(i)
    arrival_lambda[0] += step / 6.2
    arrival_lambda[3] += step / 1.5
    arrival_lambda[1] += step / 4.8
    arrival_lambda[2] += step / 1.0
    Clear_queue()

tmp = [0, 0, 0]
plt.figure(1)
x = np.linspace(0.0, 0.82, loops)
plt.ylim(0, 2 * count)
plt.xlabel('Input rate up to (0.82, 0.41, 0.82, 0.41)')
plt.ylabel('Total throughputs of algorithms')
plt.plot(x, queue_length[0], label='DAMW')
plt.plot(x, queue_length[1], label='HOL-MW')
plt.plot(x, queue_length[2], label='QL-MW')
plt.legend()
plt.show()

del I, q, channel, channel_lambda, w, k, K, arrival1, schedule_recorded
del arrival_count, arrival_lambda, schedule, queue_length, trans_count
gc.collect()
