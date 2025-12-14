#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:59:18 2023

@author: Ruijian
"""

import numpy as np
import pandas as pd
import copy
import math
import time
from scipy.stats import norm
import os

os.makedirs('simulation_HSBM')


# from variance import*
def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)
#np.random.seed(12345)

repeat = 300

n_list = [200, 400, 600, 800, 1000]

#sample multiple comparison data
L = [5]                                         # number of subjects in one comparison

comparison_graph = L.copy()
outcome_graph = L.copy()

two_SBM_prob = [0.5, 0.2, 0.3]    # p_11, p_12, p_22


tol = 1e-10

for n in n_list:
####################   Sample data
    cum_time_1, cum_time_2, cum_time_3, cum_time_4 = 0, 0, 0, 0
    cum_time_var_1, cum_time_var_2, cum_time_var_3, cum_time_var_4 = 0, 0, 0, 0
    for _ in range(repeat):
        num_subject = n
        num_comparison = n*int((np.log(n)**3)/10)
        M_n = 0.5 
        community = [int(n/2.5), n - int(n/2.5)]

        parameter_left_one = np.random.uniform(-M_n, M_n,size=(num_subject-1))
        parameter = np.insert(parameter_left_one,0,0)
        parameter = np.exp(parameter)
        pair_matrix = np.zeros((num_subject,num_subject))
        pair_matrix_champion = np.zeros((num_subject,num_subject))

        multiple_win = np.zeros(num_subject)
        champion = np.zeros(num_subject)
        champion_all = np.zeros((L[0]-1,num_subject))

        for i in range(len(L)):
            comparison_graph[i] = np.zeros((int(num_comparison/len(L)),L[i]),dtype = int)
            outcome_graph[i] = np.zeros((int(num_comparison/len(L)),L[i]),dtype = int)


        for j in range(len(L)):
            for i in range(int(num_comparison/len(L))):
                ### choose community:
                determine_prob = np.random.uniform(size = 1) 
                if determine_prob < two_SBM_prob[0]:
                    comparison_graph_JI = np.random.choice(range(community[0]),size = L[j],replace=False)
                elif determine_prob > two_SBM_prob[0] + two_SBM_prob[1]:    
                    comparison_graph_JI = np.random.choice(range(community[0],community[0]+community[1]),size = L[j],replace=False)
                else:
                    community_1_size = np.random.choice(range(1,L[j]))
                    comparison_graph_JI1 = np.random.choice(range(community[0]),size = community_1_size,replace=False)
                    comparison_graph_JI2 = np.random.choice(range(community[0],community[0]+community[1]),size = L[j] - community_1_size,replace=False)
                    comparison_graph_JI = np.concatenate([comparison_graph_JI1, comparison_graph_JI2], axis= 0)
                
                
                comparison_graph[j][i] = comparison_graph_JI
                
                
                current_comp = comparison_graph[j][i]
                current_parameter = parameter[current_comp]
                order = []
                for k in range(len(current_parameter)):
                    winner = np.random.choice(len(current_parameter), p = current_parameter/np.sum(current_parameter))
                    order.append(current_comp[winner])
            
                    delete = np.delete(current_comp,winner)
                    if k == 0:
                        pair_matrix_champion[current_comp[winner]][delete] += 1 
                    
                    if len(delete) > 0:
                        pair_matrix[current_comp[winner]][delete] += 1
                        multiple_win[current_comp[winner]] += 1
                        champion_all[k:L[-1],current_comp[winner]] += 1

                    if k == 0:
                        champion[current_comp[winner]] += 1
                        
                    current_comp = delete
                    current_parameter = np.delete(current_parameter,winner)
                outcome_graph[j][i] = np.array(order)


####################   Pairwise
        initial_K = np.ones(num_subject)
        iteration = 10000
        a = 'True'
        win = np.sum(pair_matrix,axis=1)
        
        

        pair_comparsion_graph = np.transpose(pair_matrix) + pair_matrix
        
        
        
        location={}                                    # location information
        for x in range(num_subject):
            for y in range(len(L)):
                location["l_{0}_{1}".format(y,x)]= np.where(outcome_graph[y]==x)
        
        # Estimation
        start_time = time.time()

        for K in range(iteration):
            last = initial_K.copy()
            current_probability = np.zeros((num_subject,num_subject))
            for i in range(num_subject):
                current_probability[i] = initial_K[i] + initial_K        
        
            g_matrix = pair_comparsion_graph/current_probability  
            g_matrix_sum = np.sum(g_matrix,axis=1)
    
            initial_K = (win)/g_matrix_sum
            initial_K = initial_K/initial_K[0]

            if np.max(np.abs(last - initial_K)) < tol:
                print('Pairwise Converge')
                estimation = initial_K
                break
        time_1 = time.time() - start_time
        cum_time_1 += time.time() - start_time


        
        normalized_estimation = np.exp(np.log(estimation) -np.mean(np.log(estimation)))

        normalized_parameter = np.exp(np.log(parameter) -np.mean(np.log(parameter)))
        estimation_error_1 = np.log(normalized_estimation) - np.log(normalized_parameter)
        



        
        
        print('*'*7)
        print('pairwise case')
        print('Cumulative Computation time for estimation is %s'%(cum_time_1))
        print('The entry-wise error is %s'%(np.max(np.abs(estimation_error_1))))
        print('The l2 error is %s'%(np.mean((estimation_error_1)**2)))
      
        
        
        error_infty_1 = np.max(np.abs(estimation_error_1))

        error_l2_1 = np.mean((estimation_error_1)**2)


        information = [error_infty_1, error_l2_1, time_1]
        info = pd.DataFrame(information)
        info = np.transpose(info)
        info.to_csv('simulation_HSBM/PL_qmle_%s_with_%s.csv'%(n, num_comparison), mode = 'a', index=False, header=False)
                    
      
        
####################    Full Multiple 

        initial_K_2 = np.ones(num_subject)
        iteration_2 = 10000
        a = 'True'
        m_win = multiple_win

        start_time = time.time()
        for K in range(iteration_2):
            last = initial_K_2.copy()
            current_down = np.zeros((num_subject))
            for y in range(len(L)):
                current_parameter = initial_K_2[outcome_graph[y]]
                cum_1 = np.cumsum(current_parameter[:,::-1], axis = 1)[:,::-1]
                cum_2 = np.cumsum(1/cum_1, axis = 1)
                cum_2[:,np.shape(cum_2)[1]-1] = cum_2[:,np.shape(cum_2)[1]-2]    
                for x in range(num_subject):    
                    current_down[x] += np.sum(cum_2[location['l_{0}_{1}'.format(y,x)]])
        
      
            initial_K_2 = m_win/current_down
            initial_K_2 = initial_K_2/initial_K_2[0]
        
            if np.max(np.abs(last - initial_K_2)) < tol:
                print('Multiple Converge')
                print(np.max(np.abs(last - initial_K_2)))
                estimation_2 = initial_K_2
                break     
        time_2 = time.time() - start_time

        cum_time_2 += time.time() - start_time


        
        normalized_estimation_2 = np.exp(np.log(estimation_2) -np.mean(np.log(estimation_2)))
        normalized_parameter_2 = np.exp(np.log(parameter) -np.mean(np.log(parameter)))
        estimation_error_2 = np.log(normalized_estimation_2) - np.log(normalized_parameter_2)
        

        
        
        print('*'*7)
        print('multiple case')
        print('Cumulative Computation time for estimation is %s'%(cum_time_2))
        print('The entry-wise error is %s'%(np.max(np.abs(estimation_error_2))))
        print('The l2 error is %s'%(np.mean((estimation_error_2)**2)))
        
        
        error_infty_2 = np.max(np.abs(estimation_error_2))

        error_l2_2 = np.mean((estimation_error_2)**2)


        information = [error_infty_2, error_l2_2, time_2]
        info = pd.DataFrame(information)
        info = np.transpose(info)
        info.to_csv('simulation_HSBM/PL_full_%s_with_%s.csv'%(n, num_comparison), mode = 'a', index=False, header=False)
            
            
            

####################    Top Multiple 

        for truncated in range(1, 3):
            initial_K_3 = np.ones(num_subject)
            iteration_2 = 10000
            a = 'True'
            m_win = champion_all[truncated-1]#multiple_win
    
            start_time = time.time()
            for K in range(iteration_2):
                last = initial_K_2.copy()
                current_down = np.zeros((num_subject))
                for y in range(len(L)):
                    current_parameter = initial_K_2[outcome_graph[y]]
                    cum_1 = np.cumsum(current_parameter[:,::-1], axis = 1)[:,::-1]
                    cum_2 = np.cumsum(1/cum_1, axis = 1)
                    
                    a = L[y] - truncated + 1
                    for t in range(L[y] - truncated):
                        
                        cum_2[:,np.shape(cum_2)[1]-a+t+1] = cum_2[:,np.shape(cum_2)[1]-a]
                        
                    for x in range(num_subject):    
                        current_down[x] += np.sum(cum_2[location['l_{0}_{1}'.format(y,x)]])
            
          
                initial_K_2 = m_win/current_down
                initial_K_2 = initial_K_2/initial_K_2[0]
            
                if np.max(np.abs(last - initial_K_2)) < tol:
                    print('Multiple Converge')
                    print(np.max(np.abs(last - initial_K_2)))
                    estimation_2 = initial_K_2
                    break     
            time_2 = time.time() - start_time
    

            
            normalized_estimation_2 = np.exp(np.log(estimation_2) -np.mean(np.log(estimation_2)))
            normalized_parameter_2 = np.exp(np.log(parameter) -np.mean(np.log(parameter)))
            estimation_error_2 = np.log(normalized_estimation_2) - np.log(normalized_parameter_2)

            
            
            print('*'*7)
            print('truncated is %s'%(truncated))
            print('multiple case')
            print('The entry-wise error is %s'%(np.max(np.abs(estimation_error_2))))
            print('The l2 error is %s'%(np.mean((estimation_error_2)**2)))


            error_infty_2 = np.max(np.abs(estimation_error_2))

            error_l2_2 = np.mean((estimation_error_2)**2)


    
            information = [error_infty_2, error_l2_2, time_2]
            info = pd.DataFrame(information)
            info = np.transpose(info)
            info.to_csv('simulation_HSBM/PL_top_%s_%s_with_%s.csv'%(truncated,n, num_comparison), mode = 'a', index=False, header=False)
            
            






