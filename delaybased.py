from numpy as np
import matplotlib.pylot as plt
import matplotlib.animation as animation
import Queue
import datatime
import time
import math

np.random.seed(123)

# ###begin of definition of slot length T####
# T=10
# ###end of definition of slot length T######

# #######channel and arrival#########
channel = [[0], [0]]
lam = [0.5, 0.6]
# #############end of channel and arrival#############

# ###define of schedule###########
x = 1
# ###end of define of schedule#####

# ###drop##########
D = [0, 0]
# ###en of drop####

# ####begin of auxiliary####
r = [-1.0, -1.0]
# ####end of auxiliary######

# ###begin of queues########
Z[0] = Queue.Queue(maxsize=max_queuelength)
Z[1] = Queue.Queue(maxsize=max_queuelength)
H[0] = Queue.Queue(maxsize=max_queuelength)
H[1] = Queue.Queue(maxsize=max_queuelength)
Q[0] = Queue.Queue(maxsize=max_queuelength)
Q[1] = Queue.Queue(maxsize=max_queuelength)
# ####end of queues########

# ###begin of V#####
V = 1000
# ###end of V#####

#  ###some of the loop variables####
max_queuelength = 1000000

count = 1000000
# ####end of loop variables########

# ###begin of the optimization definition#


def optimization(y):
    g = math.log(1 + y[0]) + math.log(1 + y[1])
# ###end of the optimization definition###

# ###define of initialization in the loop####

keepthemaxaux = 0.0
keepr = [0.0, 0.0]
value = 0.0
# ###########begin loop###############
# ###define of channel process########
mu[[], []]


def channel():
    mu[0] = np.random.binomal(1, 0.5, 1)
    mu[1] = np.random.binomal(1, 0.6, 1)
# ###end of channel process###########

# ###begin of data_arrival#####
Arrival_data = []
# for i in range(0,T):
Arrival_data[0].append(0.0)
Arrival_data[1].append(0.0)


def data_arrival(count, lamb):
    Arrival[0] = np.random.binomal(1, lamb[0], 1)
    Arrival[1] = np.random.binomal(1, lamb[1], 1)
    for i in range(0, 2):
        if Arrival[i] == 1:
            Arrival_data[m] = count

# ###end of data_arrival#####


# ###define of Queues_Update#####
# need(D_l(t),lamda_average, r, T_l, mu_l),count)
def Queues_Update(D, mu, lamb, r, count):

    for m in range(0, 2):
        while Z[m].empty() is False:
            for i in range(0, lamb[m] - D[m] - r[m]):
                Z[m].get()

    for m in range(0, 2):
        while Q[m].empty() is False:
            for i in range(0, mu[m] + D[m]):
                Q[m].get()
        if Arrival_data[m] != 0.0:
            Q[m].put(Arrival_data[m])

    for m in range(0, 2):
        if Q[m].empty is False:
            while H[m].empty() is False:
                for i in range(0, Tl[m](mu[m] + D[m])):
                    H[m].get()
        else:
            while H[m].empty() is False:
                H[m].get()
            if Arrival_data[m] != 0.0:
                H[m].put(1)
# ###end of Queues_Update########


for i in range(0, count):
    initialization()
    for r[0] < 1.0:
        for r[1] < 1.0:
            value = V * optimization(r) - \
                Z[0].qsize() * r[0] + Z[1].qsize() * r[1]
            if value > keepthemaxaux:
                keepthemaxaux = value
                keepr[0] = r[0]
                keepr[1] = r[1]
            r[0] = r[0] + 0.01
        r[1] = r[1] + 0.01
    Data_arrival(count, lamb)
    channel()
    if min(H[0], Z[0]) * channel[0] > min(H[1], Z[1]) * channel[1]:
        x = 1
    else:
        x = 2

    for m in range(0, 2):
        if Z[m] <= H[m]:
            D[m] = 1
        else:
            D[m] = 0

    Queues_Update(D, mu, lamb, keepr, count)


# end loop
