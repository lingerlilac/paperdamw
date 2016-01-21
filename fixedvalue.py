import numpy as np
import matplotlib.pyplot as plt

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
count = 100000

q = [0 for col in range(2)]
arrival_lambda = [0.389, 0.311]
channel_lambda = [0.5, 0.4]
channel = [[0 for col in range(count)] for row in range(2)]
arrival = [[0 for col in range(count)] for row in range(2)]
delay_record = [[0 for col in range(count)] for row in range(2)]
average = [0.0 for col in range(count)]
for m in range(0, 2):
    q[m] = Queue.Queue(maxsize=count)
tmp = [[0 for col in range(count)] for row in range(2)]
tmp1 = [[0 for col in range(count)] for row in range(2)]
for m in range(0, 2):
    tmp[m] = np.random.binomial(1, channel_lambda[m], count)
    tmp1[m] = np.random.binomial(1, arrival_lambda[m], count)
# print(len(tmp[0]),len(tmp1[0]))
for m in range(0, 2):
    for n in range(0, count):
        channel[m][n] = tmp[m][n]
        arrival[m][n] = tmp1[m][n]
# print(float(channel[0].count(1)) / float(channel[0].count(0)),
#      float(channel[1].count(1)) / float(channel[1].count(0)))
for i in range(0, count):
    weight = [0 for col in range(2)]
    for m in range(0, 2):
        if q[m].empty() is False:
            # (i - q[m].queue[0]) * channel[m][i]
            weight[m] = q[m].qsize() * channel[m][i]
        else:
            weight[m] = 0
    schedule = -1
    tmp = [0, 1]
    if weight.count(max(weight)) == 2:
        tmp1 = random.sample(tmp, 1)
        schedule = tmp1[0]
    else:
        schedule = weight.index(max(weight))
    # print(weight, schedule, q[0].qsize(), q[1].qsize(),channel[schedule][i])
    if q[schedule].empty() is False and channel[schedule][i] == 1:
        tmp = q[schedule].get()
    tmp1 = [0, 0]
    for m in range(0, 2):
        if q[m].empty() is True:
            tmp1[m] = 0
        else:
            tmp1[m] = i - q[m].queue[0]
        delay_record[m][i] = tmp1[m]
    for m in range(0, 2):
        if arrival[m][i] == 1 and q[m].full() is False:
            q[m].put(i)
    average[i] = (float(delay_record[0][i]) + float(delay_record[1][i])) / 2.0

plt.figure(1)
x = np.linspace(0, count, count)
plt.ylim(0, count)
plt.xlabel('time step')
plt.ylabel('Max Delay and Average Delay')
#plt.gca().set_yscale('log')
plt.plot(x, delay_record[0], label='HOL delay for l0')
plt.plot(x, delay_record[1], label='HOL delay for l1')
plt.plot(x, average, label='average')

plt.legend(loc='upper left')
plt.show()

del q, delay_record, channel, channel_lambda, arrival_lambda, arrival, tmp, tmp1
gc.collect()
