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
count = 1000
loops = 20
V = 0
for m in range(0, 2):
    for n in range(0, 2):
        q[m][n] = Queue.Queue(maxsize=count)
maxvalue = [0.0 for col in range(2)]
max_last = [0.0 for col in range(2)]
r_keep = [[0.0 for col in range(2)] for row in range(2)]
r = [[0.0 for col in range(2)] for row in range(2)]
channel_lambda = [0.2, 0.2]
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


def Arrival(m, n, i):
    for m in range(0, 2):
        for n in range(0, 2):
            tmp = np.random.binomial(1, arrival_lambda[n], 1)
            if tmp[0] == 1 and q[m][n].full() is False:
                q[m][n].put(i)
                # print(q[m][n].qsize(),m,n)
                arrival[m][n] += 1
            elif q[m][n].full() is True:
                print('queue', m, n, 'is full')


def Clear():
    for m in range(0, 2):
        for n in range(0, 2):
            while q[m][n].empty() is False:
                q[m][n].get()
            Z[m][n] = 0.0
            H[m][n] = 0.0


for j in range(0, loops):
    print(j)
    for i in range(0, count):
        for m in range(0, 2):
            maxvalue[m] = -100000000.0
            max_last[m] = -100000000.0
            for n in range(0, 2):
                r_keep[m][n] = 0.0
        r[0][0] = -1.0
        while r[0][0] <= 0.8:
            r[0][1] = -1.0
            while r[0][1] <= 0.8 and (r[0][0] + r[0][1]) <= 0.8:
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
                r[0][1] += 0.05
            r[0][0] += 0.05
        Channel(m, n, i)
        # print(r_keep[0], Z[0], V, maxvalue[0])
        tmp1 = min(H[0][0], Z[0][0]) * channel[0][0]
        tmp2 = min(H[0][1], Z[0][1]) * channel[0][1]
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
        #print(q[0][schedule[0]].qsize(), schedule[0], channel[0][schedule[0]])
        if q[0][schedule[0]].empty() is False and channel[0][schedule[0]] == 1:
            tmp = i - q[0][schedule[0]].get()
            delay_every_step[0][schedule[0]] += tmp
            tran_count[0][schedule[0]] += 1
            if tmp > max_delay[0][schedule[0]]:
                max_delay[0][schedule[0]] = tmp
            # print('there')
        for m in range(0, 2):
            if drop_decision[m] == 1 and q[0][m].empty() is False:
                tmp = q[0][m].get()
                drop_count[m] += 1
        for m in range(0, 2):
            tmp = Z[0][m] - arrival_lambda[m] + \
                float(drop_decision[m]) + r_keep[0][m]
            if tmp > 0:
                Z[0][m] = tmp
            else:
                Z[0][m] = 0.0
        Arrival(m, n, i)
    Clear()
    for m in range(0, 2):
        if tran_count[0][m] != 0:
            delay_record[0][m][j] = float(
                delay_every_step[0][m]) / float(tran_count[0][m])
            max_delay_record[0][m][j] = max_delay[0][m]
            # print('here')
        else:
            delay_record[0][m][j] = 0.0
        #print(delay_record[0][m][j])
    for m in range(0, 2):
        drop_count[m] = 0
        for n in range(0, 2):
            delay_every_step[m][n] = 0
            tran_count[m][n] = 0
            max_delay[m][n] = 0
    for m in range(0, 2):
        if Z[0][m] > V:
            print(r_keep[0], 'r')
    V += 1
plt.figure(1)
x = np.linspace(0, V, loops)
plt.plot(x, x)
plt.plot(x, delay_record[0][0], label='avg00')
plt.plot(x, delay_record[0][1], label='avg01')
plt.plot(x, max_delay_record[0][0], label='max00')
plt.plot(x, max_delay_record[0][1], label='max01')
# plt.figure(2)
# y = np.linspace(1, 1001, 50)
# plt.plot(y, Throughput_recorded[0])
# plt.plot(y, Throughput_recorded[1])
plt.legend()
plt.show()


del q, Z, H, count, V, loops, maxvalue, max_last, r_keep, r
del channel, schedule, channel_lambda, drop_decision
del tran_count, drop_count, arrival_lambda
gc.collect()
