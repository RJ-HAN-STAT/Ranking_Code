# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:40:06 2023

@author: Ruijian
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
#%%
def calculate_davidson_std(num_subject, theta, normalized_estimation, graph):
    est = normalized_estimation
    n=num_subject
    rho=np.zeros(n)
    delta=np.tile(est,(n,1)).T-np.tile(est,(n,1))
    for x in range(n):
        delta_row=delta[x]
        location=np.where(graph[x]!=0)
        exp_delta=np.exp(delta_row[location])
        #num=theta*exp_delta**(5/2)+(theta**2+4)*(exp_delta**2+exp_delta)+6*theta*exp_delta**(3/2)+theta*exp_delta**(1/2)
        #den=2*(2*(exp_delta+theta*exp_delta**(1/2)+1))**3
        den=exp_delta+theta*(exp_delta**(1/2))+1
        num1=exp_delta*((theta*(exp_delta**(1/2))+2)**2)
        num2=theta*(exp_delta**(1/2))*((1-exp_delta)**2)
        num3=(2*exp_delta+theta*(exp_delta**(1/2)))**2
        rho_=(num1+num2+num3)/(4*(den**3))
        rho[x]=np.sqrt(np.sum(rho_))
        #rho[x]=np.sum(num/den)
    return rho
    


def calculate_RaoKupper_std(num_subject, theta, normalized_estimation, graph):
    est = normalized_estimation
    n=num_subject
    rho=np.zeros(n)
    delta=np.tile(est,(n,1)).T-np.tile(est,(n,1))
    for x in range(n):
        delta_row=delta[x]
        location=np.where(graph[x]!=0)
        exp_delta=np.exp(delta_row[location])
        rho1=(theta**2)*exp_delta/((theta+exp_delta)**3)
        rho2=(theta**2)*(theta**2-1)*exp_delta*((1-(exp_delta**2))**2)/(((exp_delta+theta)*(theta*exp_delta+1))**3)
        rho3=theta**2*exp_delta**2/((theta*exp_delta+1)**3)
        rho[x]=np.sqrt(np.sum(rho1+rho2+rho3))
    return rho


def calculate_Paircardinal_std(num_subject, normalized_estimation, comparison_graph, sigma):
    #est = normalized_estimation
    n=num_subject
    rho=np.zeros(n)
    #delta=np.tile(est,(n,1)).T-np.tile(est,(n,1))
    for x in range(n):
        delta=comparison_graph[x]
        rho[x]=np.sqrt((np.sum(delta))/sigma**2)
    return rho


def calculate_TM_std(num_subject, normalized_estimation, comparison_graph):
    #est = normalized_estimation
    n = num_subject
    rho = np.zeros(n)
    phi = norm.pdf  
    Phi = norm.cdf  # Standard normal CDF Φ(z)
    #delta=np.tile(est,(n,1)).T-np.tile(est,(n,1))
    for i in range(n):
        sum_term = 0.0
        for j in range(n):
            if i == j:
                continue
            delta_ij = normalized_estimation[i] - normalized_estimation[j]  # u_i - u_j
            num_comparisons = comparison_graph[i][j]
            if num_comparisons > 0:
                numerator = phi(delta_ij) ** 2
                denominator = Phi(delta_ij) * (1.0 - Phi(delta_ij))
                sum_term += num_comparisons * (numerator / denominator)
        
        if sum_term > 0:
            rho[i] = 1.0 / sum_term
        else:
            rho[i] = np.inf  # No comparisons -> infinite variance
    return np.sqrt(rho)
