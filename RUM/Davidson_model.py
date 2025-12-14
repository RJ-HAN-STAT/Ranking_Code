# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import pandas as pd
from scipy.stats import norm
from RUM_variance import calculate_davidson_std


def reparameterize(parameter):
    return np.exp(parameter)/np.sum(np.exp(parameter))

def inverse(gamma):
    return np.log(gamma) - np.log(gamma)[0]


repeat = 300
n_list = [500]
# NOTICE THAT: THIS QUESTION WE USE GAMMA RATHER U.

#%%

# NOTICE THAT: THIS QUESTION WE USE GAMMA RATHER U.
for n in n_list:
    num_subject = n

# NOTICE THAT: THIS QUESTION WE USE GAMMA RATHER U.


    matrix_1 = np.ones((num_subject,num_subject))
    matrix_2 = np.zeros((num_subject,num_subject))
    for i in range(num_subject):
        for j in range(num_subject):
            if i > j:
                matrix_1[i,j], matrix_2[i,j] = 0,1
            elif i == j:
                matrix_1[i,j], matrix_2[i,j] = 0,0

    prob_list = [1/np.sqrt(n)]# (np.log(n))**3/n]
    
    
    m_list = [1, np.log(np.log(n))]        # small to large    
#%%
    for p in range(len(prob_list)):
        for m in range(len(m_list)):
            probability = prob_list[p]
            M_n = m_list[m]
            m = m + 1
            
            
            
            for k in range(repeat):
                np.random.seed(k+1)
                print(n,probability,M_n,k)
                
                #parameter_left_one = np.random.uniform(-M_n, M_n,size=(num_subject-1))
                #parameter = np.insert(parameter_left_one,0,0)
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
                    probability_graph[i] = gamma[i]/gamma
        
                win_vs_notwin_graph = probability_graph/(probability_graph + 1 + np.sqrt(probability_graph))   
                win_vs_notwin_graph = np.random.binomial(1, win_vs_notwin_graph, size = (num_subject,num_subject))  # win is one 
                notwin_vs_win_graph = 1 - win_vs_notwin_graph       # no win is one
                
                
                
                tie_vs_loss_graph_under_no_win_graph = np.sqrt(probability_graph)/(1 + np.sqrt(probability_graph))   
                tie_vs_loss_graph_under_no_win_graph = np.random.binomial(1, tie_vs_loss_graph_under_no_win_graph, size = (num_subject,num_subject))  # tie is one
                tie_vs_loss_graph_under_no_win_graph = tie_vs_loss_graph_under_no_win_graph + 1    # tie is two while loss is one
                
                outcome_graph = tie_vs_loss_graph_under_no_win_graph * notwin_vs_win_graph + win_vs_notwin_graph * 3  # loss is one, tie is two, win is three
                outcome_graph = outcome_graph*matrix_1
                
                outcome_graph_2 =  4 - np.transpose(outcome_graph)
                outcome_graph_2 = outcome_graph_2*matrix_2
                
                total_outcome_graph = outcome_graph + outcome_graph_2
                
                
                
                
                graph = total_outcome_graph*comparison_graph
                
              
                
                
                
                win = np.sum((graph == 3),axis=1)
                tie = np.sum((graph == 2),axis=1)
                loss = np.sum((graph == 1),axis=1)
    
# Above is the generate of data       

 #%%     
                initial_K = gamma
                iteration = 10000
                a = 'True'

                for K in range(iteration):
                    last = initial_K.copy()
                    current_probability = np.zeros((num_subject,num_subject))
                    for i in range(num_subject):
                        current_probability[i] = initial_K[i]/initial_K        # ij term = gamma_i/gamma_j
            
                    transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j/gamma_i
                    
                    up_matrix = (2 + np.sqrt(transpose_current_probability)) * comparison_graph
                    down_matrix = (current_probability + 1 + np.sqrt(current_probability)) * initial_K  # It is true!
                    g_matrix = up_matrix/down_matrix
                    
                    g_matrix_sum = np.sum(g_matrix,axis=1)
                    initial_K = (2*win + tie)/g_matrix_sum
                    initial_K = initial_K/np.sum(initial_K)
 #                   print(objective(initial_K))  
        
        
                    if np.max(np.abs(last - initial_K))/np.max(np.abs(gamma)) < 1e-5:
                        print(np.max(np.abs(last - initial_K))/np.max(np.abs(gamma)))
                        print('Converge')
                        estimation = initial_K
                        break




                print(a)
                parameter_estimation = inverse(estimation)
                error_infty = np.max(np.abs(parameter - parameter_estimation))
                normalized_estimation=parameter_estimation-np.mean(parameter_estimation)
                print(error_infty)
                theta=1
                rho=calculate_davidson_std(num_subject, theta, normalized_estimation, graph)
                
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
                info.to_csv('result/davidson/Davidson_%s_%s_%s.csv'%(n, p, m), mode = 'a', index=False, header=False)
                
                del transpose_current_probability, current_probability, up_matrix, down_matrix, g_matrix, comparison_graph
#%%                
for n in n_list:
    for m in range(len(m_list)):
        m=m+1
        data = pd.read_csv('result/davidson/Davidson_%s_%s_%s.csv'%(n, p, m),header=None)
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
    data = pd.read_csv('result/davidson/Davidson_%s_%s_%s.csv'%(n, p, m),header=None)
    QQ_=data[3]-data[4]
    QQ=QQ_*data[5]
    stats.probplot(QQ, dist=stats.norm, plot=plt, fit=False)
    print(n)
    ax.legend('n=%s'%(n))





# Add on y=x line
ax.plot([-2, 2.5], [-2, 2.5], c='black', ls='--',alpha=0.5)

plt.title('QQ−plot of the 2st coordinate of MLE of Davidson model',fontsize=15)
fig.savefig('result/davidson/QQdavidson.png', dpi=300 )





