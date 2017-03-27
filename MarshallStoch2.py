# -*- coding: utf-8 -*-


#Jessica Marshall
#ECE-302
#Programming Assignment #2

import numpy as np
import matplotlib.pyplot as plt

#Part 1
#Radar Detection

############################################################

num_trials = 1000
sigma_2 = 1
std_dev = np.sqrt(sigma_2)  #let standard deviation = 1
A = 1                       #let known constant A = 1
SNR = A/sigma_2
P0 = .8
P1 = .2

gen = np.random.random(num_trials)          #decide if trial is hit or miss
event = np.zeros(num_trials)

X = np.random.normal(0, std_dev, num_trials)        #generate noise
Y = np.zeros(num_trials)

MAP = np.zeros(num_trials)
decision = np.zeros(num_trials)

for i in range(0, num_trials):         
    if gen[i] < P1:     #target present
        event[i] = 1
        Y[i] = A + X[i]
    else:
        event[i] = 0        #target absent
        Y[i] = X[i]

    MAP[i] = np.exp(((-np.square(Y[i]))/(2*np.square(std_dev))) + ((np.square(Y[i] - A))/(2*np.square(std_dev))))
    if MAP[i] < P1/P0:
        decision[i] = 1
    else:
        decision[i] = 0

error = np.logical_xor(event, decision)
percent_error = 100* np.sum(error)/num_trials
print("percent error = ", percent_error, '%')

############################################################
SNR = np.array([1, .5, .05])
MAP_temp = np.zeros(num_trials)
num_thresh = 100
detect = np.zeros(num_trials)
new_thresh = .025           #this value was derived in part 1c

PF = np.zeros((SNR.size, num_thresh))
PD = np.zeros((SNR.size, num_thresh))
point = np.zeros(SNR.size)


for power in range(0, SNR.size):
    for i in range(0, num_trials):
        MAP_temp[i] = np.exp(((-np.square(Y[i]))/(2*np.square(std_dev))) + ((np.square(Y[i] - SNR[power]))/(2*np.square(std_dev))))
    
    thresholds_temp = np.linspace(0, max(MAP_temp), num_thresh)
    #print(thresholds_temp)
    point[power] = np.where(thresholds_temp == min(thresholds_temp, key=lambda x:abs(x-new_thresh)))[0][0]       #point at which thresh = .025
    print("point:", point)
    for j in range(0, num_thresh):
        for k in range(0, num_trials):
            if MAP_temp[k] < thresholds_temp[j]:
                detect[k] = 1
            else:
                detect[k] = 0
        PF[power, j] = np.sum(np.logical_and(np.logical_not(event), detect))/np.sum(np.logical_not(event))  #false positive
        PD[power, j] = np.sum(np.logical_and(event, detect))/np.sum(event)       #true positive
 


plt.figure
lw = 2
plt.plot(PF[0, :], PD[0, :], color='darkred', lw=lw, label='SNR = 1', alpha = .75)
plt.plot(PF[1, :], PD[1, :], color='darkorange', lw=lw, label='SNR = 0.5', alpha = .75)
plt.plot(PF[2, :], PD[2, :], color='darkgreen', lw=lw, label='SNR = 0.05', alpha = .75)

for i in range(0, SNR.size):
    plt.scatter(PF[i, point[i].astype(int)], PD[i, point[i].astype(int)], alpha=.5, s=100, color = 'black', marker = "*")

#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('PF')
plt.ylabel('PD')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()