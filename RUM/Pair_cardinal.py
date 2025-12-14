# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:46:00 2023

@author: Ruijian
"""


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from RUM_variance import calculate_Paircardinal_std


def reparameterize(parameter):
    return np.exp(parameter)/np.sum(np.exp(parameter))

def inverse(gamma):
    return np.log(gamma) - np.log(gamma)[0]


repeat = 300
n_list = [500]

sigma=2
# NOTICE THAT: THIS QUESTION WE USE GAMMA RATHER U.
#%%
for n in n_list:
    num_subject = n


    matrix_1 = np.ones((num_subject,num_subject))
    matrix_2 = np.zeros((num_subject,num_subject))
    for i in range(num_subject):
        for j in range(num_subject):
            if i > j:
                matrix_1[i,j], matrix_2[i,j] = 0,1
            elif i == j:
                matrix_1[i,j], matrix_2[i,j] = 0,0

    prob_list = [1/np.sqrt(n)]# (np.log(n))**3/n]
    
    
    m_list = [1, np.log(np.log(n))]           # small to large    
#%%
    for p in range(len(prob_list)):
        for m in range(len(m_list)):
            probability = prob_list[p]
            M_n = m_list[m]
            m = m + 1
            
            
            
            for k in range(repeat):
                np.random.seed(2*k+3*p+4*m+5*n)
                print(n,probability,M_n,k)
                
                parameter_temp = np.random.uniform(-M_n, M_n,size=(num_subject))
                #parameter = np.insert(parameter_left_one,0,0)
                parameter = parameter_temp- np.mean(parameter_temp)
                
                gamma = reparameterize(parameter)
                
                comparison_graph = np.zeros((num_subject,num_subject))
                for i in range(num_subject):
                    pro = np.random.uniform(low = 0, high = probability, size = (num_subject))
                    comparison_graph[i] = np.random.binomial(1, pro + np.log(n)*probability, size = (num_subject))
                
                comparison_graph = comparison_graph*matrix_1
                comparison_graph = comparison_graph + np.transpose(comparison_graph)# make the matrix symmetric            
                
                        
                        
                
            

    
                probability_graph = np.zeros((num_subject,num_subject))
                for i in range(num_subject):
                    probability_graph[i] = np.random.normal(np.log(gamma[i]/gamma),sigma)
                    
                    
                 

                outcome_graph = probability_graph
                outcome_graph = outcome_graph*matrix_1
                
                outcome_graph_2 =  - np.transpose(outcome_graph)
                
                total_outcome_graph = outcome_graph + outcome_graph_2
                
                
                
                
                graph = total_outcome_graph*comparison_graph

# Above is the generate of data       

 #%%     
                initial_K = gamma
                iteration = 10000
                a = 'True'
                
                A = np.zeros((n, n))
                b = np.zeros(n)
                
                for i in range(n):
                    for j in range(n):
                        if graph[i, j] != 0:
                            A[i, i] += 1
                            A[j, j] += 1
                            A[i, j] -= 1
                            A[j, i] -= 1
                            b[i] += graph[i, j]
                            b[j] -= graph[i, j]
                
                # Solving the linear system Au = b
                if np.linalg.matrix_rank(A)==n:
                    estimation = np.linalg.solve(A, b)
                else:
                    A_pinv = np.linalg.pinv(A)
                    estimation = np.dot(A_pinv, b)
                
                


                print(a)
                parameter_estimation = estimation-estimation[0]
                error_infty = np.max(np.abs(parameter - parameter_estimation))
                normalized_estimation=parameter_estimation-np.mean(parameter_estimation)
                print(error_infty)
                #theta=1
                rho=calculate_Paircardinal_std(num_subject, normalized_estimation, comparison_graph, sigma)
                
                normalized_parameter = parameter -np.mean(parameter)
                estimation_error = normalized_estimation - normalized_parameter
                


                overall_CP_95_2 = np.int8(abs(estimation_error*rho) < norm.ppf(0.975))
                
                
                #print('The l2 error is %s'%(np.mean((estimation_error)**2)))
                print('Average CP is %s'%(np.mean(overall_CP_95_2)))
                
                
                std_mean = np.mean(1/rho)

                CP_mean = np.mean(overall_CP_95_2)
                
                QQ=rho[1]*estimation_error[1]
               
                information = [error_infty, std_mean, CP_mean, normalized_estimation[1], normalized_parameter[1], rho[1],QQ]
                
                info = pd.DataFrame(information)
                info = np.transpose(info)
                #info.to_csv('Davidson_%s_%s_%s.csv'%(n, p, m), mode = 'a', index=False, header=False)
                info.to_csv('result/paircardinal/Paircardinal_%s_%s_%s.csv'%(n, p, m), mode = 'a', index=False, header=False)
                
#                del transpose_current_probability, current_probability, up_matrix, down_matrix, g_matrix, comparison_graph
#%%                
for n in n_list:
    for m in range(len(m_list)):
        m=m+1
        data = pd.read_csv('result/paircardinal/Paircardinal_%s_%s_%s.csv'%(n, p, m),header=None)
        stats_data= data.loc[:,[0,1,2]]
        print('n=%s, m=%s'%(n,m) )
        print(stats_data.describe())
    #%%
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')


#fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
for n in n_list:
    m=1
    data = pd.read_csv('result/paircardinal/Paircardinal_%s_%s_%s.csv'%(n, p, m),header=None)
    QQ_=data[3]-data[4]
    QQ=QQ_*data[5]
    stats.probplot(QQ, dist=stats.norm, plot=plt, fit=False)
    print(n)
    ax.legend('n=%s'%(n))




# Add on y=x line
ax.plot([-2, 2.5], [-2, 2.5], c='black', ls='--',alpha=0.5)

plt.title('QQ−plot of the 2st coordinate of MLE of Paired Cardinal model',fontsize=15)
fig.savefig('result/paircardinal/QQpaired.png', dpi=300 )        
