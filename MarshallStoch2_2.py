# -*- coding: utf-8 -*-


#Jessica Marshall
#ECE-302
#Programming Assignment #2

import numpy as np
import matplotlib.pyplot as plt

#Part 2
#Non-Gaussian Detection

############################################################
lambda_0 = 10       #avg number of 0-photons per second
lambda_1 = 5        #avg number of 1-photons per second
num_trials = 1000

np.random.exponential(1/lambda_0)   #generate time between 0-photon bursts
np.random.exponential(1/lambda_1)   #generate time between 1-photon bursts


data = np.random.randint(2, size=num_trials)       #generate bitstream data
time = np.zeros(num_trials)

for i in range(0, data.size):           #determine time between photons in data
    if data[i] == 0:
        time[i] = np.random.exponential(1/lambda_0)
    else:
        time[i] = np.random.exponential(1/lambda_1)

