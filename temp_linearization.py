# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:50:37 2025

@author: wengerj
Objectif du code: réaliser la regression libneaire de T_end(fctT_start)

1:Calculer T_end pour Plusieurs Tstart et Plusieurs lignes, en fct du Debit
    On aura ainsi F(D) en entree du pb d'opti'
    
2:Creer le pb d'opti
    fct obectif=carre de l'erreur sommé sur tout le code'
    trouver la pente de la regression a partir d'un point'
Idee: faire 
"""



import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt


R = 4.7
C = 4000
Ta = 20
D1=20
T1=60
T_list=np.linspace(40, 60,10).tolist()
D_list=np.linspace(1, D1,200).tolist()
L_list=[8000,6000,5000,4000,10000,50]

def plot_test_T_end(T,L,seuil):
    D_min=2
    D_list=np.linspace(D_min,200,200)
    Ta=20
    C=4000
    R=4.7
    T_true=[]
    L=8000
    (alpha_cold_low_values, beta_cold_low_values, alpha_cold_high_values, beta_cold_high_values,
           alpha_hot_low_values,beta_hot_low_values,alpha_hot_high_values,beta_hot_high_values) = regression_lineraire_cold_hot_low_high(1,200, seuil, 30, 40, 60, 70)
    
    for D in D_list:
        T_true+=[Ta + (T - Ta) * np.exp(-L / (R * C * D))]
    
    T_lin=[]
    D_list_low = np.linspace(D_min,seuil,seuil)
    D_list_high=np.linspace(seuil,200,200-seuil)
    for D in D_list_low:
        T_lin+=[T+beta_hot_low_values[0]+alpha_hot_low_values[0]*D]
    
    for D in D_list_high:
        T_lin+=[T+beta_hot_high_values[0]+alpha_hot_high_values[0]*D]

    plt.plot(D_list,T_true)
    plt.plot(D_list,T_lin)
    plt.show()
    return()

def T_End_Values(D, L, T_start):
    return Ta + (T_start - Ta) * np.exp(-L / (R * C * D))


def regression_lineraire_cold_hot(D_min=3,D_max=130,T_min=40,T_max=100,T1=55):
    
    ##Vraies valeurs cold
    T_list_cold=np.linspace(T_min, T1,10).tolist()
    D_list=np.linspace(D_min, D_max,200).tolist()
    T_end_true_cold=[]
    for L in L_list:
        temp1=[]
        for D in D_list:
            temp2=[]
            for T_start in T_list_cold:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_cold+=[temp1]
    
    m = gp.Model("regression temp low")
    alpha_cold = m.addVars(len(L_list), lb=-100, ub=100, name="alpha_cold")
    beta_cold = m.addVars(len(L_list),lb=-100,ub=100,name="beta_cold")
    # D1 = m.addVars(1, lb=0, ub=50, name="beta")
    
    m.setObjective(
        gp.quicksum(
            ((T_end_true_cold[iL][iD][iT] - (T_list_cold[iT]+beta_cold[iL]+D_list[iD]*alpha_cold[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list))
            for iT in range(len(T_list))
        ))
    m.setParam('OutputFlag', 0)
    m.optimize()
    alpha_cold_values = [alpha_cold[i].X for i in range(len(L_list))]
    beta_cold_values = [beta_cold[i].X for i in range(len(L_list))]
    
    
    
    ##Vraies valeurs hot
    T_list_hot=np.linspace(T1, T_max,10).tolist()
    D_list=np.linspace(D_min, D_max,200).tolist()
    T_end_true_hot=[]
    for L in L_list:
        temp1=[]
        for D in D_list:
            temp2=[]
            for T_start in T_list_hot:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_hot+=[temp1]
    
    m1 = gp.Model("regression temp high")
    alpha_hot = m1.addVars(len(L_list), lb=-1, ub=1, name="alpha_hot")
    beta_hot = m1.addVars(len(L_list),lb=-20,ub=20,name="beta_hot")
    # D1 = m.addVars(1, lb=0, ub=50, name="beta")
    
    m1.setObjective(
        gp.quicksum(
            ((T_end_true_hot[iL][iD][iT] - (T_list_hot[iT]+beta_hot[iL]+D_list[iD]*alpha_hot[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list))
            for iT in range(len(T_list_hot))
        ))
    m1.setParam('OutputFlag', 0)
    m1.optimize()
    
    
    
    alpha_hot_values = [alpha_hot[i].X for i in range(len(L_list))]
    beta_hot_values = [beta_hot[i].X for i in range(len(L_list))]
    
    return(alpha_cold_values,beta_cold_values,alpha_hot_values,beta_hot_values)

# alpha_cold_values,beta_cold_values,alpha_hot_values,beta_hot_values = regression_lineraire_cold_hot(2,200,40,95,55)


# alpha_cold_values,beta_cold_values,alpha_hot_values,beta_hot_values


def regression_lineraire_cold_hot_low_high(D_min,D_max,D1,T1,T2,T3,T4):
    R = 4.7
    C = 4000
    Ta = 20
    L_list=[8000,6000,5000,4000,10000,50]
    D_min=D_min+0.0001
    ##Vraies valeurs cold
    T_list_cold=np.linspace(T1, T2,10).tolist()
    D_list_low=np.linspace(D_min, D1,D1).tolist()
    D_list_high=np.linspace(D1, D_max,200-D1).tolist()

    T_end_true_cold_low=[]
    for L in L_list:
        temp1=[]
        for D in D_list_low:
            temp2=[]
            for T_start in T_list_cold:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_cold_low+=[temp1]
    
    T_end_true_cold_high=[]
    for L in L_list:
        temp1=[]
        for D in D_list_high:
            temp2=[]
            for T_start in T_list_cold:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_cold_high+=[temp1]
    
    m = gp.Model("regression temp low")


    alpha_cold_low = m.addVars(len(L_list), lb=-1000, ub=1000, name="alpha_cold_low")
    beta_cold_low = m.addVars(len(L_list),lb=-1000,ub=1000,name="beta_cold_low")
    alpha_cold_high = m.addVars(len(L_list), lb=-1000, ub=1000, name="alpha_cold_high")
    beta_cold_high = m.addVars(len(L_list),lb=-1000,ub=1000,name="beta_cold_high")
    
    # D1 = m.addVars(1, lb=0, ub=50, name="beta")
    
    m.setObjective(
        gp.quicksum(
            ((T_end_true_cold_low[iL][iD][iT] - (T_list_cold[iT]+beta_cold_low[iL]+D_list_low[iD]*alpha_cold_low[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list_low))
            for iT in range(len(T_end_true_cold_low))
        )+
        gp.quicksum(
            ((T_end_true_cold_high[iL][iD][iT] - (T_list_cold[iT]+beta_cold_high[iL]+D_list_high[iD]*alpha_cold_high[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list_high))
            for iT in range(len(T_end_true_cold_high))
        )
        )
    m.addConstrs((beta_cold_low[iL]+D1*alpha_cold_low[iL] == 
                beta_cold_high[iL]+D1*alpha_cold_high[iL] for iL in range(len(L_list))),
                name = 'continuity')
    # m.addConstr()
    m.addConstrs((beta_cold_low[iL]+D1*alpha_cold_low[iL]<=-0.001 for iL in range(len(L_list))),name="sign 1")
    m.addConstrs((beta_cold_high[iL]+D_max*alpha_cold_high[iL]<=-0.001 for iL in range(len(L_list))),name="sign 1")
    m.setParam('OutputFlag', 0)
    m.optimize()
    alpha_cold_low_values = [np.round(alpha_cold_low[i].X,7) for i in range(len(L_list))]
    beta_cold_low_values = [np.round(beta_cold_low[i].X,7) for i in range(len(L_list))]
    alpha_cold_high_values=[np.round(alpha_cold_high[i].X,7) for i in range(len(L_list))]
    beta_cold_high_values=[np.round(beta_cold_high[i].X,7) for i in range(len(L_list))]
    
    
    ##Vraies valeurs hot
    T_list_hot=np.linspace(T3, T4,10).tolist()
    T_end_true_hot_low=[]
    for L in L_list:
        temp1=[]
        for D in D_list_low:
            temp2=[]
            for T_start in T_list_hot:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_hot_low+=[temp1]
    
    T_end_true_hot_high=[]
    for L in L_list:
        temp1=[]
        for D in D_list_high:
            temp2=[]
            for T_start in T_list_hot:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_hot_high+=[temp1]
    
    
    m = gp.Model("regression temp low")


    alpha_hot_low = m.addVars(len(L_list), lb=-1000, ub=1000, name="alpha_hot_low")
    beta_hot_low = m.addVars(len(L_list),lb=-100,ub=100,name="beta_hot_low")
    alpha_hot_high = m.addVars(len(L_list), lb=-1000, ub=1000, name="alpha_hot_high")
    beta_hot_high = m.addVars(len(L_list),lb=-100,ub=100,name="beta_hot_high")
    
    # D1 = m.addVars(1, lb=0, ub=50, name="beta")
    
    m.setObjective(
        gp.quicksum(
            ((T_end_true_hot_low[iL][iD][iT] - (T_list_hot[iT]+beta_hot_low[iL]+D_list_low[iD]*alpha_hot_low[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list_low))
            for iT in range(len(T_end_true_hot_low))
        )+
        gp.quicksum(
            ((T_end_true_hot_high[iL][iD][iT] - (T_list_hot[iT]+beta_hot_high[iL]+D_list_high[iD]*alpha_hot_high[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list_high))
            for iT in range(len(T_end_true_hot_high))
        )
        )
    m.addConstrs((beta_hot_low[iL]+D1*alpha_hot_low[iL] == 
                beta_hot_high[iL]+D1*alpha_hot_high[iL] for iL in range(len(L_list))),
                name = 'continuity2')
    
    m.addConstrs((beta_hot_low[iL]+D1*alpha_hot_low[iL]<=-0.001 for iL in range(len(L_list))),name="sign 1 hot")
    m.addConstrs((beta_hot_high[iL]+D_max*alpha_hot_high[iL]<=-0.001 for iL in range(len(L_list))),name="sign 2 hot")
    m.setParam('OutputFlag', 0)
    m.optimize()
    alpha_hot_low_values = [np.round(alpha_hot_low[i].X,7) for i in range(len(L_list))]
    beta_hot_low_values = [np.round(beta_hot_low[i].X,7) for i in range(len(L_list))]
    alpha_hot_high_values=[np.round(alpha_hot_high[i].X,7) for i in range(len(L_list))]
    beta_hot_high_values=[np.round(beta_hot_high[i].X,7) for i in range(len(L_list))]
    

    
    return(alpha_cold_low_values, beta_cold_low_values, alpha_cold_high_values, beta_cold_high_values,
           alpha_hot_low_values,beta_hot_low_values,alpha_hot_high_values,beta_hot_high_values)
        
#%% plotting
def regression_lineraire_cold_hot_low_high_plot(D_min,D_max,D1,T1,T2,T3,T4):
    R = 4.7
    C = 4000
    Ta = 20
    L_list=[8000,6000,5000,4000,10000,50]
    D_min=D_min+0.0001
    ##Vraies valeurs cold
    T_list_cold=np.linspace(T1, T2,100).tolist()
    D_list_low=np.linspace(D_min, D1,D1).tolist()
    D_list_high=np.linspace(D1, D_max,D_max-D1).tolist()

    T_end_true_cold_low=[]
    for L in L_list:
        temp1=[]
        for D in D_list_low:
            temp2=[]
            for T_start in T_list_cold:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_cold_low+=[temp1]
    
    T_end_true_cold_high=[]
    for L in L_list:
        temp1=[]
        for D in D_list_high:
            temp2=[]
            for T_start in T_list_cold:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_cold_high+=[temp1]
    
    # print(T_end_true_cold_high)

    m = gp.Model("regression temp low")


    alpha_cold_low = m.addVars(len(L_list), lb=-1, ub=0.01, name="alpha_cold")
    beta_cold_low = m.addVars(len(L_list),lb=-100,ub=100,name="beta_cold")
    alpha_cold_high = m.addVars(len(L_list), lb=-1, ub=0.01, name="alpha_cold")
    beta_cold_high = m.addVars(len(L_list),lb=-100,ub=100,name="beta_cold")
    
    # D1 = m.addVars(1, lb=0, ub=50, name="beta")
    
    m.setObjective(
        gp.quicksum(
            ((T_end_true_cold_low[iL][iD][iT] - (T_list_cold[iT]+beta_cold_low[iL]+D_list_low[iD]*alpha_cold_low[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list_low))
            for iT in range(len(T_end_true_cold_low))
        )+
        gp.quicksum(
            ((T_end_true_cold_high[iL][iD][iT] - (T_list_cold[iT]+beta_cold_high[iL]+D_list_high[iD]*alpha_cold_high[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list_high))
            for iT in range(len(T_end_true_cold_high))
        )
        )
    m.addConstrs((beta_cold_low[iL]+D1*alpha_cold_low[iL] == 
                beta_cold_high[iL]+D1*alpha_cold_high[iL] for iL in range(len(L_list))),
                name = 'continuity')
    m.addConstrs((beta_cold_low[iL]+D1*alpha_cold_low[iL]<=-0.001 for iL in range(len(L_list))),name="sign 1")
    m.addConstrs((beta_cold_high[iL]+D_max*alpha_cold_high[iL]<=-0.001 for iL in range(len(L_list))),name="sign 1")
    m.setParam('OutputFlag', 0)
    m.optimize()
    alpha_cold_low_values = [np.round(alpha_cold_low[i].X,7) for i in range(len(L_list))]
    beta_cold_low_values = [np.round(beta_cold_low[i].X,7) for i in range(len(L_list))]
    alpha_cold_high_values=[np.round(alpha_cold_high[i].X,7) for i in range(len(L_list))]
    beta_cold_high_values=[np.round(beta_cold_high[i].X,7) for i in range(len(L_list))]
    
    
    
    
    
    
    ##Vraies valeurs hot
    T_list_hot=np.linspace(T3, T4,100).tolist()
    T_end_true_hot_low=[]
    for L in L_list:
        temp1=[]
        for D in D_list_low:
            temp2=[]
            for T_start in T_list_hot:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_hot_low+=[temp1]
    
    T_end_true_hot_high=[]
    for L in L_list:
        temp1=[]
        for D in D_list_high:
            temp2=[]
            for T_start in T_list_hot:
                temp2+=[Ta + (T_start - Ta) * np.exp(-L / (R * C * D))]
            temp1+=[temp2]
        T_end_true_hot_high+=[temp1]
    
    
    m = gp.Model("regression temp low")

    
    alpha_hot_low = m.addVars(len(L_list), lb=-1, ub=0.00001, name="alpha_hot")
    beta_hot_low = m.addVars(len(L_list),lb=-100,ub=100,name="beta_hot")
    alpha_hot_high = m.addVars(len(L_list), lb=-1, ub=0.0001, name="alpha_hot")
    beta_hot_high = m.addVars(len(L_list),lb=-100,ub=100,name="beta_hot")
    
    # D1 = m.addVars(1, lb=0, ub=50, name="beta")
    #HOT
    m.setObjective(
        gp.quicksum(
            ((T_end_true_hot_low[iL][iD][iT] - (T_list_hot[iT]+beta_hot_low[iL]+D_list_low[iD]*alpha_hot_low[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list_low))
            for iT in range(len(T_end_true_hot_low))
        )+
        gp.quicksum(
            ((T_end_true_hot_high[iL][iD][iT] - (T_list_hot[iT]+beta_hot_high[iL]+D_list_high[iD]*alpha_hot_high[iL])) ** 2)
            for iL in range(len(L_list))
            for iD in range(len(D_list_high))
            for iT in range(len(T_end_true_hot_high))
        )
        )
    m.addConstrs((beta_hot_low[iL]+D1*alpha_hot_low[iL] == 
                beta_hot_high[iL]+D1*alpha_hot_high[iL] for iL in range(len(L_list))),
                name = 'continuity')
    m.addConstrs((beta_hot_low[iL]+D1*alpha_hot_low[iL]<=-0.001 for iL in range(len(L_list))),name="sign 1 hot")
    m.addConstrs((beta_hot_high[iL]+D_max*alpha_hot_high[iL]<=-0.001 for iL in range(len(L_list))),name="sign 2 hot")
    m.setParam('OutputFlag', 0)
    m.optimize()
    alpha_hot_low_values = [np.round(alpha_hot_low[i].X,7) for i in range(len(L_list))]
    beta_hot_low_values = [np.round(beta_hot_low[i].X,7) for i in range(len(L_list))]
    alpha_hot_high_values=[np.round(alpha_hot_high[i].X,7) for i in range(len(L_list))]
    beta_hot_high_values=[np.round(beta_hot_high[i].X,7) for i in range(len(L_list))]
    
    plt.plot(D_list_low+D_list_high,T_end_true_hot_low[0][:][0]+T_end_true_hot_high[0][:][0])
    plt.show()
    
    return(T_end_true_cold_low)
    
#%%

threshold = 20
D_min = 0.001
D_max = 220


(alpha_cold_low, beta_cold_low, 
 alpha_cold_high, beta_cold_high,
 alpha_hot_low,beta_hot_low,
 alpha_hot_high,beta_hot_high) =regression_lineraire_cold_hot_low_high(D_min,D_max, threshold, 40, 40, 60, 70)

# print("ypts_cold")

ypts_cold = [(
    beta_cold_low[i] + alpha_cold_low[i] * D_min,
    beta_cold_low[i] + alpha_cold_low[i] * threshold ,
    beta_cold_high[i] + alpha_cold_high[i] * D_max) for i in range(len(alpha_cold_low))
]
# print(ypts_cold)

# print("ypts_hot")

ypts_hot = [(
    beta_hot_low[i] + alpha_hot_low[i] * D_min,
    beta_hot_low[i] + alpha_hot_low[i] * threshold ,
    beta_hot_high[i] + alpha_hot_high[i] * D_max) for i in range(len(alpha_hot_low))
]
# print(ypts_hot)







#%%



# max_errors_per_D = []
# def plot_results(alpha_values,beta_values)    :
#     for iD, D in enumerate(D_list):
#         max_error = 0
#         for iL, L in enumerate(L_list):
#             for iT, T_start in enumerate(T_list):
#                 T_true = T_end_true[iL][iD][iT]
#                 T_pred = T_start+beta_values[iL]+D_list[iD]*alpha_values[iL]
#                 error = abs(T_true - T_pred)
#                 if error > max_error:
#                     max_error = error
#         max_errors_per_D.append(max_error)
    
#     # Plotting
#     plt.figure(figsize=(8, 5))
#     plt.plot(D_list, max_errors_per_D, label='Max Error per D', color='red')
#     plt.xlabel("D")
#     plt.ylabel("Max Absolute Error")
#     plt.title("Max Prediction Error vs D")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
    
    
    
#     for iL, L in enumerate(L_list):
#         plt.figure(figsize=(10, 6))
#         for iT, T_start in enumerate(T_list):
#             T_true_curve = [T_end_true[iL][iD][iT] for iD in range(len(D_list))]
#             T_pred_curve = [T_start+beta_values[iL]+D_list[iD]*alpha_values[iL] for iD in range(len(D_list))]
    
#             plt.plot(D_list, T_true_curve, linestyle='-', label=f'True T_start={T_start:.1f}°C')
#             plt.plot(D_list, T_pred_curve, linestyle='--', label=f'Approx T_start={T_start:.1f}°C')
    
#         plt.title(f"True vs Approximated Temperature for L = {L}")
#         plt.xlabel("D")
#         plt.ylabel("Final Temperature")
#         # plt.ylim(bottom=0)
#         # plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#     return()

# (alpha_values,beta_values)




# m=gp.Model("regression")
# alpha=m.addVars(len(L_list), lb=0, ub=1)


# m.setObjectiveN(gp.quicksum(
#                     gp.quicksum(
#                         gp.quicksum(
#                             gp.abs_((T_end_true[L][D][T_start]-(Ta + alpha[L]*D*(T_start-Ta)))
#                                             *
#                                             (T_end_true[L][D][T_start]-(Ta + alpha[L]*D*(T_start-Ta))))
                
#                 for L in range(len(L_list))
#                 for D in range(len(D_list))) for T_start in range(len(T_list)))))










































