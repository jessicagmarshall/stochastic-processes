# -*- coding: utf-8 -*-


#Jessica Marshall
#ECE-302
#Programming Assignment #2

import numpy as np
import matplotlib.pyplot as plt

#Part 2
#Non-Gaussian Detection

############################################################
lambda_0 = 5       #avg number of 0-photons per second
lambda_1 = 1005        #avg number of 1-photons per second
num_trials = 1000

#np.random.exponential(1/lambda_0)   #generate time between 0-photon bursts
#np.random.exponential(1/lambda_1)   #generate time between 1-photon bursts


data = np.random.randint(2, size=num_trials).astype('float64')       #generate bitstream data
time = np.zeros(num_trials)

for i in range(0, data.size):           #determine time between photons in data
    if data[i] == 0:
        time[i] = np.random.exponential(1/lambda_0)
    else:
        time[i] = np.random.exponential(1/lambda_1)

############################################################        
MAP_temp = np.zeros(num_trials)
num_thresh = 100
detect = np.zeros(num_trials)

PF = np.zeros(num_thresh)
PD = np.zeros(num_thresh)

for i in range(0, num_trials):
    MAP_temp[i] = (lambda_1/lambda_0)*np.exp((lambda_0-lambda_1)*time[i])
    
thresholds_temp = np.linspace(0, max(MAP_temp), num_thresh)
    #print(thresholds_temp)
#point[power] = np.where(thresholds_temp == min(thresholds_temp, key=lambda x:abs(x-new_thresh)))[0][0]       #point at which thresh = .025
#print("point:", point)
for j in range(0, num_thresh):
    for k in range(0, num_trials):
        if MAP_temp[k] < thresholds_temp[j]:
            detect[k] = 0
        else:
            detect[k] = 1
    PF[j] = np.sum(np.logical_and(np.logical_not(data), detect))/np.sum(np.logical_not(data))  #false positive
    PD[j] = np.sum(np.logical_and(data, detect))/np.sum(data) 
    
plt.figure
lw = 2
plt.plot(PF, PD, color='darkred', label = 'difference in lambdas = 1000', lw=lw, alpha = .75)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('PF')
plt.ylabel('PD')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

############################################################
############################################################
lambda_0 = 5       #avg number of 0-photons per second
lambda_1 = 15        #avg number of 1-photons per second
num_trials = 1000

#np.random.exponential(1/lambda_0)   #generate time between 0-photon bursts
#np.random.exponential(1/lambda_1)   #generate time between 1-photon bursts


data = np.random.randint(2, size=num_trials).astype('float64')       #generate bitstream data
time = np.zeros(num_trials)

for i in range(0, data.size):           #determine time between photons in data
    if data[i] == 0:
        time[i] = np.random.exponential(1/lambda_0)
    else:
        time[i] = np.random.exponential(1/lambda_1)

############################################################        
MAP_temp = np.zeros(num_trials)
num_thresh = 100
detect = np.zeros(num_trials)

PF = np.zeros(num_thresh)
PD = np.zeros(num_thresh)

for i in range(0, num_trials):
    MAP_temp[i] = (lambda_1/lambda_0)*np.exp((lambda_0-lambda_1)*time[i])
    
thresholds_temp = np.linspace(0, max(MAP_temp), num_thresh)
    #print(thresholds_temp)
#point[power] = np.where(thresholds_temp == min(thresholds_temp, key=lambda x:abs(x-new_thresh)))[0][0]       #point at which thresh = .025
#print("point:", point)
for j in range(0, num_thresh):
    for k in range(0, num_trials):
        if MAP_temp[k] < thresholds_temp[j]:
            detect[k] = 0
        else:
            detect[k] = 1
    PF[j] = np.sum(np.logical_and(np.logical_not(data), detect))/np.sum(np.logical_not(data))  #false positive
    PD[j] = np.sum(np.logical_and(data, detect))/np.sum(data) 
    
plt.figure
lw = 2
plt.plot(PF, PD, color='darkorange', label = 'difference in lambdas = 10', lw=lw, alpha = .75)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('PF')
plt.ylabel('PD')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

############################################################
############################################################
lambda_0 = 5       #avg number of 0-photons per second
lambda_1 = 6        #avg number of 1-photons per second
num_trials = 1000

#np.random.exponential(1/lambda_0)   #generate time between 0-photon bursts
#np.random.exponential(1/lambda_1)   #generate time between 1-photon bursts


data = np.random.randint(2, size=num_trials).astype('float64')       #generate bitstream data
time = np.zeros(num_trials)

for i in range(0, data.size):           #determine time between photons in data
    if data[i] == 0:
        time[i] = np.random.exponential(1/lambda_0)
    else:
        time[i] = np.random.exponential(1/lambda_1)

############################################################        
MAP_temp = np.zeros(num_trials)
num_thresh = 100
detect = np.zeros(num_trials)

PF = np.zeros(num_thresh)
PD = np.zeros(num_thresh)

for i in range(0, num_trials):
    MAP_temp[i] = (lambda_1/lambda_0)*np.exp((lambda_0-lambda_1)*time[i])
    
thresholds_temp = np.linspace(0, max(MAP_temp), num_thresh)
    #print(thresholds_temp)
#point[power] = np.where(thresholds_temp == min(thresholds_temp, key=lambda x:abs(x-new_thresh)))[0][0]       #point at which thresh = .025
#print("point:", point)
for j in range(0, num_thresh):
    for k in range(0, num_trials):
        if MAP_temp[k] < thresholds_temp[j]:
            detect[k] = 0
        else:
            detect[k] = 1
    PF[j] = np.sum(np.logical_and(np.logical_not(data), detect))/np.sum(np.logical_not(data))  #false positive
    PD[j] = np.sum(np.logical_and(data, detect))/np.sum(data) 
    
plt.figure
lw = 2
plt.plot(PF, PD, color='darkred', label = 'difference in lambdas = 1', lw=lw, alpha = .75)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('PF')
plt.ylabel('PD')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()