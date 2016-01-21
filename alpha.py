import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gc
import platform
np.random.seed(123)
sysstr = platform.system()
print(sysstr)
if sysstr == 'Windows':
    import queue as Queue
elif sysstr == 'Darwin':
    import Queue
q = [[0 for col in range(2)] for row in range(2)]
Z = [[0.0 for col in range(2)] for row in range(2)]
H = [[0.0 for col in range(2)] for row in range(2)]
Y = [0.0 for col in range(2)]
q1 = [0 for col in range(2)]
count = 1000
loops = 100
for m in range(0, 2):
    q1[m] = Queue.Queue(maxsize=2 * count)
V = 100
k = [0.1, 0.1]
w = [0.0 for col in range(2)]
qe = [0.0 for col in range(2)]
alpha = 0.01
R = [0 for col in range(2)]

for n in range(0, 2):
    q[0][n] = Queue.Queue(maxsize=2 * count)
    q[1][n] = Queue.Queue(maxsize=count)
maxvalue = [0.0 for col in range(2)]
max_last = [0.0 for col in range(2)]
r_keep = [[0.0 for col in range(2)] for row in range(2)]
r = [[0.0 for col in range(2)] for row in range(2)]
channel_lambda = [0.4, 0.4]
arrival_lambda = [0.5, 0.5]
channel = [[0 for col in range(2)] for row in range(2)]
schedule = [0 for col in range(2)]
drop_decision = [0 for col in range(2)]
delay_record = [[[0.0 for col in range(loops)]
                 for col in range(2)] for row in range(2)]
delay_every_step = [[0 for col in range(2)] for row in range(2)]
tran_count = [[0 for col in range(2)] for row in range(2)]
drop_count = [0 for col in range(2)]
arrival = [[0 for col in range(2)] for row in range(2)]
max_delay_record = [[[0.0 for col in range(loops)]
                     for col in range(2)] for row in range(2)]
max_delay = [[0 for col in range(2)] for row in range(2)]


def Channel(m, n, i):
    for m in range(0, 2):
        for n in range(0, 2):
            tmp = np.random.binomial(1, channel_lambda[n], 1)
            channel[m][n] = tmp[0]


def Arrival(i):
    for n in range(0, 2):
        # tmp = np.random.binomial(1, arrival_lambda[n], 1)
        # if tmp[0] == 1 and q[0][n].full() is False:
        #     q[0][n].put(i)
        #     # print(q[m][n].qsize(),m,n)
        #     arrival[0][n] += 1
        # elif q[0][n].full() is True:
        #     print('queue', m, n, 'is full')

        if R[m] == 1 and q[1][n].full() is False and q[0][n].full() is False:
            q[1][n].put(i)
            arrival[1][n] += 1
            q[0][n].put(i)
            q1[n].put(-1)
            arrival[0][n] += 1
        elif q[1][n].full() is True:
            print('queue', 1, n, 'is full')
        elif q[0][n].full() is True:
            print('queue', 0, n, 'is full')


def Clear():
    for m in range(0, 2):
        for n in range(0, 2):
            while q[m][n].empty() is False:
                q[m][n].get()
            Z[m][n] = 0.0
            H[m][n] = 0.0
    for m in range(0, 2):
        while q1[m].empty() is False:
            q1[m].get()
    for m in range(0, 1):
        w[m] = 0.0
        qe[m] = 0.0


for j in range(0, loops):
    print(j)
    for i in range(0, count):
        for m in range(0, 2):
            maxvalue[m] = -100000000.0
            max_last[m] = -100000000.0
            R[m] = 0
            for n in range(0, 2):
                r_keep[m][n] = 0.0
        r[0][0] = -1.0
        r[1][0] = -1.0
        while r[0][0] <= 0.8 and r[1][0] <= 0.8:
            r[0][1] = -1.0
            r[1][1] = -1.0
            while r[0][1] <= 0.8 and (r[0][0] + r[0][1]) <= 0.8 and r[1][1] <= 0.8 and (r[1][0] + r[1][1]) <= 0.8:
                if r[0][0] < 0:
                    tmp1 = r[0][0]
                else:
                    tmp1 = math.log(1 + r[0][0])
                if r[0][1] < 0:
                    tmp2 = r[0][1]
                else:
                    tmp2 = math.log(1 + r[0][1])
                tmp3 = V * (tmp1 + tmp2)
                tmp4 = Z[0][0] * r[0][0] + Z[0][1] * r[0][1]
                maxvalue[0] = tmp3 - tmp4
                if maxvalue[0] > max_last[0]:
                    max_last[0] = maxvalue[0]
                    r_keep[0][0] = r[0][0]
                    r_keep[0][1] = r[0][1]
                if r[1][0] < 0:
                    tmp1 = r[1][0]
                else:
                    tmp1 = math.log(1 + r[1][0])
                if r[1][1] < 0:
                    tmp2 = r[1][1]
                else:
                    tmp2 = math.log(1 + r[1][1])
                tmp3 = V * (tmp1 + tmp2)
                tmp4 = Y[0] * r[1][0] + Y[1] * r[1][1]
                maxvalue[1] = tmp3 - tmp4
                if maxvalue[1] > max_last[1]:
                    max_last[1] = maxvalue[1]
                    r_keep[1][0] = r[1][0]
                    r_keep[1][1] = r[1][1]
                r[0][1] += 0.05
                r[1][1] += 0.05
            r[0][0] += 0.05
            r[0][1] += 0.05
        Channel(m, n, i)
        # print(r_keep[0], Z[0], V, maxvalue[0])
        if q1[0].qsize() - q[0][0].qsize() != 0:
            print('impossible1')
        if q1[1].qsize() - q[0][1].qsize() != 0:
            print('impossible2')
        if q1[0].empty() is False:
            tm1 = i - q[0][0].queue[0]
        else:
            tm1 = 0
        if q1[1].empty() is False:
            tm2 = i - q[0][1].queue[0]
        else:
            tm2 = 0
        tmp1 = min(tm1, Z[0][0]) * channel[0][0]
        tmp2 = min(tm2, Z[0][1]) * channel[0][1]
        if tmp1 > tmp2:
            schedule[0] = 0
        elif tmp1 == tmp2:
            tmp = [0, 1]
            tmp3 = random.sample(tmp, 1)
            schedule[0] = tmp3[0]
        else:
            schedule[0] = 1
        for m in range(0, 2):
            tmp = 0
            if q[0][m].empty() is False:
                tmp = float(i - q[0][m].queue[0])
            else:
                tmp = 0.0
            drop_decision[m] = 0
            if Z[0][m] <= tmp:
                drop_decision[m] = 1
        # tm3 = (Z[1][0] - H[1][0]) * (channel[1]
        #                              [0] * k[0] * w[0] * qe[0] - alpha)
        # tm4 = (Z[1][1] - H[1][1]) * (channel[1]
        #                              [1] * k[1] * w[1] * qe[1] - alpha)
        # tm1 = channel[1][1] * k[1] * w[1] * qe[1]
        # tm0 = channel[1][0] * k[0] * w[0] * qe[0]
        tm3 = (Z[1][0] - H[1][0]) * (0.5 * k[0] * w[0] * qe[0] - alpha)
        tm4 = (Z[1][1] - H[1][1]) * (0.5 * k[1] * w[1] * qe[1] - alpha)
        tm1 = 0.5 * k[1] * w[1] * qe[1]
        tm0 = 0.5 * k[0] * w[0] * qe[0]
        #print('w', w,'qe', qe, 'tm0', tm0, 'tm1', tm1)
        #print(tm3, tm4, 'z0', Z[1][0], 'h0', H[1][0], 'z1', Z[1][1], 'h1', H[1][1], j, i, alpha)
        tmp1 = q[1][0].qsize() * channel[1][0] #+ tm3
        tmp2 = q[1][1].qsize() * channel[1][1] #+ tm4
        if tmp1 > tmp2:
            schedule[1] = 0
        elif tmp1 == tmp2:
            tmp = [0, 1]
            tmp3 = random.sample(tmp, 1)
            schedule[1] = tmp3[0]
        else:
            schedule[1] = 1
        for m in range(0, 2):
            tmp1 = q[1][m].qsize()
            tmp = float(tmp1)
            if tmp < Y[m]:
                R[m] = 1
            else:
                R[m] = 0
        if q[0][schedule[0]].empty() is False and channel[0][schedule[0]] == 1 and q1[schedule[0]].empty() is False:
            tmp1 = q1[schedule[0]].get()
            tmp2 = q[0][schedule[0]].get()
            if tmp1 == -1:
                delay_every_step[0][schedule[0]] += i - tmp2  # tmp2
                tmp = i - tmp2  # tmp2
            else:
                delay_every_step[0][schedule[0]] += i - tmp2
                tmp = i - tmp2
            tran_count[0][schedule[0]] += 1
            if tmp > max_delay[0][schedule[0]]:
                max_delay[0][schedule[0]] = tmp
            # print('there')
        if q[1][schedule[1]].empty() is False and channel[1][schedule[1]] == 1:
            tmp = i - q[1][schedule[1]].get()
            delay_every_step[1][schedule[1]] += tmp
            tran_count[1][schedule[1]] += 1
            if tmp > max_delay[1][schedule[1]]:
                max_delay[1][schedule[1]] = tmp
            w[schedule[1]] = float(delay_every_step[1][
                                   schedule[1]]) / tran_count[1][schedule[1]]
            t1 = q[1][schedule[1]].qsize()
            qe[schedule[1]] = float(t1)
            # print('there')
        for m in range(0, 2):
            if drop_decision[m] == 1 and q[0][m].empty() is False:
                tmp = q[0][m].get()
                q1[m].get()
                # q[0][m].put(i)
                # q1[m].put(tmp)
                drop_count[m] += 1
        for m in range(0, 2):
            tmp = Z[0][m] - arrival_lambda[m] + \
                float(drop_decision[m]) + r_keep[0][m]
            if tmp > 0:
                Z[0][m] = tmp
            else:
                Z[0][m] = 0.0
        for m in range(0, 2):
            tmp = Z[1][m] - k[m] * channel[1][m] * w[m] * qe[m] + alpha
            if tmp > 0:
                Z[1][m] = tmp
            else:
                Z[1][m] = 0
        for m in range(0, 2):
            tmp = H[1][m] + k[m] * channel[1][m] * w[m] * qe[m] - alpha
            if tmp > 0:
                H[1][m] = tmp
            else:
                H[1][m] = 0
        for m in range(0, 2):
            tmp = Y[m] - R[m] + r_keep[1][m]
            if tmp > 0:
                Y[m] = tmp
            else:
                Y[m] = 0
        Arrival(i)
    Clear()
    for m in range(0, 2):
        for n in range(0, 2):
            if tran_count[m][n] != 0:
                delay_record[m][n][j] = float(
                    delay_every_step[m][n]) / float(tran_count[m][n])
                max_delay_record[m][n][j] = max_delay[m][n]
            else:
                delay_record[m][n][j] = 0.0
        # print(delay_record[0][m][j])
    for m in range(0, 2):
        drop_count[m] = 0
        for n in range(0, 2):
            delay_every_step[m][n] = 0
            tran_count[m][n] = 0
            max_delay[m][n] = 0
    # print(delay_record[0][m], delay_record[1][0])
    V += 1
plt.figure(1)
x = np.linspace(0, V, loops)
plt.xlabel('V')
plt.ylabel('Max Delay and Average Delay')
plt.plot(x, x + 2)
plt.plot(x, delay_record[0][0], label='Avg, 0, HOL-Neely')
plt.plot(x, delay_record[0][1], label='Avg, 1, HOL-Neely')
plt.plot(x, max_delay_record[0][0], label='Max, 0, HOL-Neely')
plt.plot(x, max_delay_record[0][1], label='Max, 1, HOL-Neely')
plt.plot(x, delay_record[1][0], label='Avg, 0, CL-DAMW')
plt.plot(x, delay_record[1][1], label='Avg, 1, CL-DAMW')
plt.plot(x, max_delay_record[1][0], label='Max, 0, CL-DAMW')
plt.plot(x, max_delay_record[1][1], label='Max, 1, CL-DAMW')
# plt.figure(2)
# y = np.linspace(1, 1001, 50)
# plt.plot(y, Throughput_recorded[0])
# plt.plot(y, Throughput_recorded[1])
plt.legend(loc='upper left')
plt.show()


del q, Z, H, count, V, loops, maxvalue, max_last, r_keep, r
del channel, schedule, channel_lambda, drop_decision
del tran_count, drop_count, arrival_lambda
gc.collect()
