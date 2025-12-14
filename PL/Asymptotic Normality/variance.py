# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:42:32 2023

@author: Ruijian
"""

# for exponential estimator

import numpy as np
import pandas as pd
import copy
import math
import time


import itertools



def calculate_pair_std(num_subject, L, normalized_estimation, outcome_graph, location):

    estimation = normalized_estimation
    variance_num = np.zeros(num_subject)
    variance_den = np.zeros(num_subject)
    for x in range(num_subject):
        for y in range(len(L)):
            data_x = location['l_{0}_{1}'.format(y,x)]
            result = outcome_graph[y][data_x[0]]
            estimator_data = estimation[result]
            for k in range(len(data_x[1])):
                estimator_data_k = estimator_data[k]
                index_x = data_x[1][k]
                estimator_x = estimator_data_k[index_x]
                for i in range(L[y]):
                    if (i != index_x):
                        a = (estimator_x * estimator_data_k[i])/(estimator_data_k[i] + estimator_x)**2
                        variance_den[x] += a
                        variance_num[x] += a 
                    
                        for j in range(L[y]):
                            if (j != i) and (j != index_x):
                                
                                variance_den[x] += estimator_x/(estimator_data_k[i] + estimator_data_k[j] + estimator_x)
                                variance_den[x] -= (estimator_x)**2/ ((estimator_data_k[i] + estimator_x) *( estimator_data_k[j] + estimator_x))
            
                        
    desired_rho =  variance_num/np.sqrt(variance_den)
    return desired_rho

    
def calculate_pair_std_2(num_subject, L, normalized_estimation, outcome_graph, location):

    estimation = normalized_estimation
    variance_num = np.zeros(num_subject)
    variance_den = np.zeros(num_subject)
    for x in range(num_subject):
        for y in range(len(L)):
            data_x = location['l_{0}_{1}'.format(y,x)]
            result = outcome_graph[y][data_x[0]]
            estimator_data = estimation[result]
            for k in range(len(data_x[1])):
                estimator_data_k = estimator_data[k]
                index_x = data_x[1][k]
                estimator_x = estimator_data_k[index_x]
                for i in range(L[y]):
                    if (i != index_x):
                        a = (estimator_x * estimator_data_k[i])/(estimator_data_k[i] + estimator_x)**2
                        variance_den[x] += a
                        variance_num[x] += a 
                    

            
                        
    desired_rho =  variance_num/np.sqrt(variance_den)
    return desired_rho

def calculate_var_PL(estimator_data_k, index_x, truncated_version):
    var = 0
    estimator_data_k_minus_x = np.delete(estimator_data_k, index_x)
    estimator_data_k_x = estimator_data_k[index_x]
    for l in range(truncated_version):
        if l == 0:
            pp = estimator_data_k_x/np.sum(estimator_data_k)
            var +=  pp*(1-pp)

        if l > 0:
            for pati in itertools.permutations(estimator_data_k_minus_x,l):
                #print(pati)
                var_l, minus_part = 1, 0
                for j in range(l):
                    var_l = var_l * pati[j]/(np.sum(estimator_data_k) - minus_part)
                    minus_part += pati[j]
                pp = estimator_data_k_x/(np.sum(estimator_data_k) - minus_part)
                var += var_l * pp * (1-pp)
    return var



def calculate_PL_std(num_subject, L, normalized_estimation, outcome_graph, location, trun=100):

    estimation = normalized_estimation
    variance_num = np.zeros(num_subject)
    variance_den = np.zeros(num_subject)
    for x in range(num_subject):
        for y in range(len(L)):
            # print(x,y)
            data_x = location['l_{0}_{1}'.format(y,x)]
            result = outcome_graph[y][data_x[0]]
            estimator_data = estimation[result]
            for k in range(len(data_x[1])):
                # print(k)
                estimator_data_k = estimator_data[k]
                index_x = data_x[1][k]
                estimator_x = estimator_data_k[index_x]
                truncated_version = len(estimator_data_k)-1
                truncated_version = np.min([truncated_version, trun])
  
    variance_num = variance_den
    desired_rho =  variance_num/np.sqrt(variance_den)
    return desired_rho


def var_1(a,t):
    s=0
    for x in range(t,a):
       s += (x/(x+1))/a
    return s


def var_2(a):
    s= 0.75*(a-1)/(a+1)
    return s





