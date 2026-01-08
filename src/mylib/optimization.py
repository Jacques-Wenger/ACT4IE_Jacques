import random
import time

import numpy as np
from global_parameters import *
from data_import import *
from utils import *
import gurobipy as gp

def catalogue(choix):
    # choix=[liste des choix de lignes]

    # Caractérisitques des renforcements
    # Liste des résistances/réactances linéques des trois sections alu
    r_lin = [0.125 /2 ,0.125 /3 ,0.125 /4]
    # On ajoute les multiples de section max au listes r_list, x_list et s_max
    x_lin = [0.1, 0.1, 0.1]
    # Liste des longueurs
    longueurs = [30, 30, 10, 1, 8, 5, 20, 4, 5]
    prix_poser = 110000
    prix_materiau = [ 24, 36, 48]
    # Liste des résistances/réactances totales initiales en pu
    r_initial = [0.486533681141036,
                 0.48935267328094,
                 0.0626681726907631,
                 0.02652243752467,
                 0.0749738690751891,
                 0.257590457080734,
                 0.369003788703729,
                 0.187243746420511,
                 0.015624999999999998  ]# 8eme renforcee en i = 5
    x_init = [0.794196331415867,
              0.793222256649126,
              0.27492491472259,
              0.0268822024456043,
              0.0627791922540079,
              0.137010845009378,
              0.170831514968032,
              0.108669117099955,
              0.0375  ]# 8eme renforcee en i = 5

    # calcul des couts d'installation des lignes
    r_tot = []
    x_tot = []
    prix_tot = []
    for j in range(len(longueurs)):
        tempr = [r_initial[j ] *Zref]
        tempx = [x_init[j ] *Zref]
        tempprix = [0]
        for k in range(len(r_lin)):
            tempr += [r_lin[k ] *longueurs[j]]
            tempx += [x_lin[k ] *longueurs[j]]
            tempprix += [(prix_poser +prix_materiau[k] *1000 ) *longueurs[j]]
        r_tot += [tempr]
        x_tot += [tempx]
        prix_tot += [tempprix]

    # Liste des résistances/réactances totales en pu
    r_list = [[ j /Zref for j in i] for i in r_tot]
    x_list = [[ j /Zref for j in i] for i in x_tot]

    # Liste des Imax
    Imax_list = [405 *2 ,405 *3 ,405 *4]
    imax_list = [ i /Iref for i in Imax_list]

    L_max = [ i**2 for i in imax_list]

    # Matrice des Lmax: Pour chaque ligne j, L_Matrix[j][0]= Lmax de la ligne initiale
    Lmax_initial = [0.4191563, 0.4191563,  0.4191563,  0.4191563,
                    0.62122889, 0.2251666, 0.46418962, 0.2251666,  1.9410062400000005] # 8eme renforcee en i = 5

    L_Matrix = []
    for j in range(len(longueurs)):
        L_Matrix += [[Lmax_initial[j]]]
        L_Matrix[j] += L_max

    # Output: choix des lignes
    R = []
    X = []
    CAPEX = []
    Lmax = []

    choix_list = []
    for i in range(len(choix)):
        choix_list += [int(choix[i])]

    # updated_catalogue = [0,4,5,6]
    for i in range(len(r_list)):
        # for i in updated_catalogue:
        R += [r_list[i][choix_list[i]]]
        X += [x_list[i][choix_list[i]]]
        CAPEX += [prix_tot[i][choix_list[i]]]
        Lmax += [L_Matrix[i][choix_list[i]]]
    return R, X, Lmax, CAPEX

def create_curt_model (elec_grid,thermal_grid,choix=[0]*9,coeff_tload=1.0,
                       coeff_pv_prod=1.0, HP_MAX_ELEC=3000000,
                       curt=True,storage=False,outputFlag=False):
    # Caractéristiques des lignes
    R, X, Lmax, CAPEX = catalogue(choix)
    # topologie reseau de chaleur
    Bh, tLoadMatrix, thermalGridMatrix, thermalNodesMatrix,thermal_losses_coeffs= thermal_grid
    alpha_cold_values, beta_cold_values, alpha_hot_values, beta_hot_values = thermal_losses_coeffs

    B,  elecLoadProfilePMatrix, elecLoadProfileQMatrix, elecProdProfilePMatrix, N = elec_grid
    model = gp.Model("thermique")

    setT = range(tLoadMatrix.shape[0])
    # setT = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    # time=23
    # setT=[time]
    buses = range(tLoadMatrix.shape[1])
    thlines = range(thermalGridMatrix.shape[0])
    nbuses = tLoadMatrix.shape[1]
    nlines = thermalGridMatrix.shape[0]
    # tLoadMatrix=0.7
    # electrique
    el_lines = B.shape[1]
    el_lines = range(el_lines)
    w_min = 0.95 ** 2
    w_max = 1.05 ** 2
    eLPMatrix = elecLoadProfilePMatrix.T
    eLQMatrix = elecLoadProfileQMatrix.T
    ePPMatrix = elecProdProfilePMatrix.T

    ###Coeff modulations pv tload
    tLoadMatrix = coeff_tload * tLoadMatrix
    ePPMatrix[9, :] = coeff_pv_prod * ePPMatrix[9, :]
    ePPMatrix[10, :] = coeff_pv_prod * ePPMatrix[10, :]
    # ePPMatrix[9]=0
    # Conversion de B en listes de lignes arrivantes et partantes
    l_in = [[] for _ in buses]
    l_out = [[] for _ in buses]

    for i in buses:
        for j in el_lines:
            if B[i, j] == 1:
                l_out[i] += [j]
            if B[i, j] == -1:
                l_in[i] += [j]

    n_start = [[] for _ in el_lines]
    n_end = [[] for _ in el_lines]

    for i in buses:
        for j in el_lines:
            if B[i, j] == 1:
                n_start[j] = i
            if B[i, j] == -1:
                n_end[j] = i
    Cp = 4180
    # Pref = 30_000_000
    # Cp = Cp / 1e6
    # Pref = Pref / 1e6
    # tLoadMatrix = tLoadMatrix / 1e6

    # nt = tLoadMatrix.shape[0]
    nt = 24
    W = model.addVars(nbuses, nt, lb=w_min, ub=w_max, name="V_squared")
    W.Start = 1
    L = model.addVars(el_lines, nt, lb=0, ub=2, name="I_squared")
    L.Start = 0
    P_line = model.addVars(el_lines, nt, lb=-1, ub=1, name="Line_P_power")
    P_line.Start = 0
    Q_line = model.addVars(el_lines, nt, lb=-1, ub=1, name="Line_Q_power")
    Q_line.Start = 0
    P_slack = model.addVars(nbuses, nt, lb=-1, ub=1, name="P_slack")
    P_slack.Start = 0
    Q_slack = model.addVars(nbuses, nt, lb=-1, ub=1, name="Q_slack")
    Q_slack.Start = 0
    P_gen = model.addVars(nbuses, nt, lb=0, ub=1, name="P_gen")
    P_gen.Start = 0
    Hp_P = model.addVars(nbuses, nt, lb=0, ub=2, name="hp_elec_P_load")
    Hp_P.Start = 0
    Hp_Q = model.addVars(nbuses, nt, lb=0, ub=2, name="hp_elec_Q_load")
    Hp_Q.Start = 0

    # thermal buses
    clientBuses = [2, 4, 5, 6]
    discBuses = [12]
    prodBuses = [10, 11]

    thermalbuses = [2, 4, 5, 6, 10, 11, 12]

    D_ech = model.addVars(nbuses, nt, lb=-90, ub=80, name="D_ech")
    D_ech.Start = 0
    D_hot = model.addVars(nlines, nt, lb=0, ub=200, name="D_hot")
    D_hot.Start = 0
    D_cold = model.addVars(nlines, nt, lb=0, ub=200, name="D_cold")
    D_cold.Start = 0

    T_ech_hot = model.addVars(nbuses, nt, lb=40, ub=100, name="T_ech_hot")
    T_ech_hot.Start = 95
    T_ech_cold = model.addVars(nbuses, nt, lb=0, ub=50, name="T_ech_cold")
    T_ech_cold.Start = 40

    T_cold_start = model.addVars(nlines, nt, lb=20, ub=50, name="T_cold_start")
    T_cold_start.Start = 40
    T_cold_end = model.addVars(nlines, nt, lb=20, ub=50, name="T_cold_end")
    T_cold_end.Start = 40

    T_hot_start = model.addVars(nlines, nt, lb=30, ub=100, name="T_hot_start")
    T_hot_start.Start = 95
    T_hot_end = model.addVars(nlines, nt, lb=30, ub=100, name="T_hot_end")
    T_hot_end.Start = 95

    p_th_gen = model.addVars(len(buses), nt, lb=0, ub=1, name="p_th_gen")
    p_th_gen.Start = 1 / 2

    p_th_charge = model.addVars(nbuses, nt, lb=0, ub=1, name="p_th_charge")
    p_th_charge.Start = 0

    p_th_store = model.addVars(nbuses, nt, lb=0, ub=1, name="p_th_store")
    p_th_store.Start = 0

    p_th_inj = model.addVars(nbuses, nt, lb=0, ub=1, name="p_th_inj")
    p_th_inj.Start = 0

    epsilon = model.addVar(lb=0, ub=1, name="epsilon")

    # total_pv_prod=0
    # for i in [9,10]:
    #     for t in setT:
    #         total_pv_prod+=ePPMatrix[i,t]

    Cost_curt = 0.083
    # model.setObjective(gp.quicksum(p_th_gen[i,t] for i in [11] for t in setT))
    Cost_joules = 0.1  # /kWh
    model.setObjective(gp.quicksum(
        ePPMatrix[i, t] - P_gen[i, t] for i in [9, 10] for t in setT) * 365 * 30000 * Cost_curt + epsilon * 100000
                       )

    # Matrice d'incidence du réseau chaud
    Bh = np.array(Bh)

    # paramétrage de échangeurs
    tk = 273
    # Matrice d'incidence du réseau froid, nomé et décalré dans le meme ordre que son symétrique froid
    Bc = -Bh

    # liste lignes chaud arrivant
    lhi = [[] for _ in range(nbuses)]
    # liste lignes chaud sortant
    lho = [[] for _ in range(nbuses)]

    # liste des noeuds de starts: hot
    nhs = [[] for _ in range(nlines)]
    # liste des noeuds de end: hot
    nhe = [[] for _ in range(nlines)]

    # liste lignes froid arrivant
    lci = [[] for _ in range(nbuses)]
    # liste lignes foid sortant:
    lco = [[] for _ in range(nbuses)]

    # liste des noeuds de starts: cold
    ncs = [[] for _ in range(nlines)]
    # liste des noeuds de end: cold
    nce = [[] for _ in range(nlines)]

    for i in range(Bh.shape[0]):
        for j in range(Bh.shape[1]):
            if Bh[i, j] == 1:  # ligne chaude j sort de noeud i
                lho[i] += [j]
                nhs[j] += [i]
            if Bh[i, j] == -1:
                lhi[i] += [j]
                nhe[j] += [i]
    for i in range(Bh.shape[0]):
        for j in range(Bh.shape[1]):
            if Bc[i, j] == 1:  # ligne chaude j sort de noeud i
                lco[i] += [j]
                ncs[j] += [i]
            if Bc[i, j] == -1:
                lci[i] += [j]
                nce[j] += [i]

    ##### Declaration et ajout des contraintes ############
    # model.addConstrs(T_hot_end[i, t] >= 40 for i in thlines for t in setT)
    # model.addConstrs(T_hot_start[i, t] >= 40 for i in thlines for t in setT)

    # Signe de D_ech en Fonction du cas
    model.addConstrs(
        (D_ech[i, t] >= 0 for i in clientBuses for t in setT), name="D_ech_clientBuses")
    model.addConstrs(
        (D_ech[i, t] <= 0 for i in prodBuses for t in setT), name="D_ech_prodBuses")
    model.addConstrs(
        (D_ech[i, t] == 0 for i in discBuses for t in setT), name="D_ech_discBuses")

    # Egalité de température
    [model.addConstrs((T_hot_start[j, t] == T_hot_start[k, t]
                       for j in lho[i] for k in lho[i] for t in setT), name="Temp_disconnected_hot") for i in discBuses]

    [model.addConstrs((T_cold_start[j, t] == T_cold_start[k, t]
                       for j in lco[i] for k in lco[i] for t in setT), name="Temp_disconnected_cold") for i in
     discBuses]

    [model.addConstrs((T_hot_start[j, t] == T_ech_hot[i, t]
                       for j in lho[i] for t in setT), name="Temp_client_hot") for i in clientBuses]

    [model.addConstrs((T_hot_end[j, t] == T_ech_hot[i, t]
                       for j in lhi[i] for t in setT), name="Temp_client_hot") for i in clientBuses]

    [model.addConstrs((T_cold_start[j, t] == T_ech_cold[i, t]
                       for j in lco[i] for t in setT), name="Temp_prod_cold") for i in prodBuses]

    [model.addConstrs((T_cold_end[j, t] == T_ech_cold[i, t]
                       for j in lci[i] for t in setT), name="Temp_prod_cold") for i in prodBuses]

    # Power Balance
    model.addConstrs((gp.quicksum(D_hot[j, t] * (T_hot_end[j, t] + tk) for j in lhi[i])
                      ==
                      gp.quicksum(D_hot[j, t] * (T_hot_start[j, t] + tk)
                                  for j in lho[i])
                      + D_ech[i, t] * (T_ech_hot[i, t] + tk) for i in buses for t in setT), name='hot_power_balance')

    model.addConstrs((gp.quicksum(D_cold[j, t] * (T_cold_end[j, t] + tk) for j in lci[i])
                      + D_ech[i, t] * (T_ech_cold[i, t] + tk)
                      ==
                      gp.quicksum(D_cold[j, t] * (T_cold_start[j, t] + tk)
                                  for j in lco[i])
                      for i in buses for t in setT), name='cold_power_balance')
    # Flow Balance
    model.addConstrs((gp.quicksum(D_hot[j, t] for j in lhi[i])
                      ==
                      gp.quicksum(D_hot[j, t] for j in lho[i])
                      + D_ech[i, t] for i in buses for t in setT), name='hot_flow_balance')

    model.addConstrs((gp.quicksum(D_cold[j, t] for j in lci[i])
                      + D_ech[i, t]
                      ==
                      gp.quicksum(D_cold[j, t] for j in lco[i]) for i in buses for t in setT), name='cold_flow_balance')

    # Heat Losses

    # alpha_cold_values,beta_cold_values,alpha_hot_values,beta_hot_values
    model.addConstrs(((T_hot_end[i, t] == T_hot_start[i, t] + beta_hot_values[i] + D_hot[i, t] * alpha_hot_values[i])
                      for i in thlines for t in setT), name='temp_losses_hot')

    model.addConstrs((T_cold_end[i, t] == T_cold_start[i, t] + beta_cold_values[i] + D_cold[i, t] * alpha_cold_values[i]
                      for i in thlines for t in setT), name='temp_losses_cold')

    # extraction de puissance par échangeur
    model.addConstrs((D_ech[i, t] * (T_ech_hot[i, t] -
                                     T_ech_cold[i, t]) * Cp / Pref == tLoadMatrix[t, i] for i in clientBuses for t in
                      setT), name='power_extraction_client')
    # injection de puissance par échangeur
    model.addConstrs((D_ech[i, t] * (T_ech_hot[i, t] -
                                     T_ech_cold[i, t]) * Cp / Pref == -(p_th_inj[i, t] + p_th_store[i, t]) for i in
                      prodBuses for t in setT), name='power_injection_client')

    # Limite de production
    model.addConstrs(p_th_gen[i, t] <= 1 for i in [11] for t in setT)

    model.addConstrs(p_th_gen[i, t] == 0 for i in clientBuses for t in setT)
    model.addConstrs(p_th_gen[i, t] == 0 for i in discBuses for t in setT)
    model.addConstrs(p_th_gen[i, t] == 0 for i in [0, 1, 2, 3, 4, 7, 9, 8] for t in setT)
    # reglage des echangeurs
    model.addConstrs(T_ech_hot[i, t] == 95 for i in prodBuses for t in setT)
    model.addConstrs(T_ech_cold[i, t] == 40 for i in clientBuses for t in setT)

    # Q_gen=model.addVars(nbuses,nt,lb=0,ub=2,name="Nodal_Q_slack")
    # print(ePPMatrix[11, :])
    slack = [0, 1, 8]
    non_slack = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12]

    # Réglages des
    model.addConstrs(
        (P_slack[i, t] == 0 for i in non_slack for t in setT), name="non_slack_P")
    model.addConstrs(
        (Q_slack[i, t] == 0 for i in non_slack for t in setT), name="non_slack_Q")
    model.addConstrs(
        (W[i, t] == 1 for i in slack for t in setT), name="slack_W")

    model.addConstrs((W[n_start[i], t] * L[i, t] == P_line[i, t] ** 2 + Q_line[i, t]
                      ** 2 for i in el_lines for t in setT), name="Branch_Apparent_Power")
    model.addConstrs((W[n_start[i], t] - W[n_end[i], t] - 2 * (R[i] * P_line[i, t] + X[i] * Q_line[i, t]) + (
                R[i] ** 2 + X[i] ** 2) * L[i, t] == 0
                      for i in el_lines for t in setT), name="voltage_drop")

    model.addConstrs((P_slack[i, t] + gp.quicksum(P_line[j, t] - R[j] * L[j, t] for j in l_in[i]) - gp.quicksum(
        P_line[j, t] for j in l_out[i])
                      - eLPMatrix[i, t] + P_gen[i, t] - Hp_P[i, t]
                      == 0 for i in buses for t in setT), name="nodal_P_balance")

    model.addConstrs((Q_slack[i, t] + gp.quicksum(Q_line[j, t] - X[j] * L[j, t] for j in l_in[i]) - gp.quicksum(
        Q_line[j, t] for j in l_out[i])
                      - eLQMatrix[i, t] - Hp_Q[i, t] == 0 for i in buses for t in setT), name="nodal_Q_balance")

    # LIMITES de sécurités
    model.addConstrs(
        (L[i, t] <= Lmax[i] for i in el_lines for t in setT), "current_upper_limit")
    model.addConstrs(
        (0 <= L[i, t] for i in el_lines for t in setT), "current_lower_limit")

    # Limite de production
    el_prodBuses = [9, 10, 11]
    nonProdBuses = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]
    model.addConstrs((P_gen[i, t] <= ePPMatrix[i, t]
                      for i in el_prodBuses for t in setT), name="prod_ub_P")
    model.addConstrs(
        (P_gen[i, t] >= 0 for i in el_prodBuses for t in setT), name="prod_ub_P")
    model.addConstrs(
        (P_gen[i, t] == 0 for i in nonProdBuses for t in setT), name="non_prod_P")

    # COUPLAGE DES RESEAU:
    CHP = [11]
    HP = [10]
    nonHP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    ratio_chp = 2
    model.addConstrs((P_gen[i, t] - p_th_gen[i, t] / (ratio_chp)
                      == 0 for i in CHP for t in setT), "chp_coupling")

    model.addConstrs(
        (Hp_P[i, t] == 0 for i in nonHP for t in setT), name="non_Heat_Pump")

    # Couplage Pompe à Chaleur
    model.addConstrs((Hp_Q[i, t] == 0.43 * Hp_P[i, t]
                      for i in buses for t in setT), name="heat_pump_reactive_Power")
    model.addConstrs((p_th_gen[i, t] == (1 / 0.9) * Hp_P[i, t] * HP_COP for i in [10]
                      for t in setT), name="heat_pump_active_Power")
    model.addConstrs((Hp_P[i, t] <= HP_MAX_ELEC /
                      Pref for i in HP for t in setT), name="heat_pump_max_P")

    # model.setParam('TimeLimit', 20)
    # model.setParam('MaxC')
    # model.setParam("MIPGap", 0.039)
    # model.setParam('Threads', 8)
    model.addConstrs(p_th_gen[11, t] >= 0.096 for t in setT)

    store_nodes = [10]
    non_stor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    model.addConstrs((p_th_charge[i, t] * p_th_store[i, t] <= epsilon for i in store_nodes for t in setT),
                     name="store_constraint")
    setT2 = range(1, 24)
    eff1 = 0.9
    eff2 = 0.9

    if storage:
        E_store = model.addVars(nbuses, nt, lb=0, ub=1, name="E_stor")
        E_store.Start = 0
        model.addConstrs((p_th_gen[i, t] == p_th_inj[i, t] + p_th_charge[i, t] for i in buses for t in setT))

        model.addConstrs((p_th_charge[i, t] == 0 for i in non_stor for t in setT))
        model.addConstrs((p_th_store[i, t] == 0 for i in non_stor for t in setT))

        model.addConstrs(
            (E_store[i, t] == E_store[i, t - 1] + eff1 * p_th_charge[i, t - 1] - p_th_store[i, t - 1] / eff2 for i in
             store_nodes for t in setT2), name="storage SOC")
        model.addConstrs((E_store[i, 0] == 0.2) for i in store_nodes)
        model.addConstrs((E_store[i, 23] == 0.2) for i in store_nodes)
        model.addConstrs((E_store[i, t] == 0) for i in non_stor for t in setT)
    else:
        model.addConstrs((p_th_charge[i, t] == 0 for i in buses for t in setT))
        model.addConstrs((p_th_store[i, t] == 0 for i in buses for t in setT))

    model.setParam('OutputFlag', outputFlag)
    model.setParam("MIPGap", 0.053)
    model.setParam("NonConvex", 2)
    model.setParam("TimeLimit", 90)
    model.setParam("NumericFocus", 1)
    model.setParam("MIPFocus", 1)
    model.setParam("FeasibilityTol", 1e-4)
    model.setParam("OptimalityTol", 1e-4)
    model.setParam("IntFeasTol", 1e-5)
    model.setParam("BarConvTol", 1e-6)
    model.setParam("BarQCPConvTol", 1e-4)
    return model

def make_fitness_TOTEX(elec_grid,thermal_grid,curt=True,storage=False,coeff_tload=1,coeff_pvprod=1,
                       HP_MAX_ELEC=3000000,outputFlag=False):
    def fitness_fct_wrapper(choix):
        m = create_curt_model(elec_grid,thermal_grid,choix,coeff_tload,
                              coeff_pvprod,HP_MAX_ELEC,storage,outputFlag=False)
        m.optimize()
        df = df_from_model(m)
        #OPEX curtailment costs and joule losses
        joule = joule_from_df(df,choix)
        curt = curt_from_df(df,choix,coeff_pvprod)
        #CAPEX
        capex = capex_from_x(choix)
        #TOTEX
        totex = joule + curt + capex
        return totex
    return fitness_fct_wrapper


def simulated_annealing_memory(initial, blackbox, n_vars=9, n_levels=4,
                               init_temp=200000, final_temp=10000, cooling_rate=0.8,
                               steps_per_temp=30, max_no_improve=100):


    cache = {}  # <---- NEW: stores solution → value

    def evaluate(sol):
        """Return cached blackbox evaluation."""
        key = tuple(sol)
        if key not in cache:
            cache[key] = blackbox(sol)
        return cache[key]

    # Initialize
    current = initial[:]
    current_val = evaluate(current)

    best = current[:]
    best_val = current_val

    start_time = time.time()
    T = init_temp
    no_improve = 0
    iteration = 0

    print(f"Initial solution: f={best_val:.3f}, x={current}")

    while T > final_temp and no_improve < max_no_improve:
        for _ in range(steps_per_temp):

            # Random perturbation
            candidate = current[:]
            idx = random.randint(0, n_vars - 1)
            candidate[idx] = random.randint(0, n_levels - 1)

            # Evaluate using caching
            candidate_val = evaluate(candidate)

            delta = candidate_val - current_val
            accept = (delta < 0) or (random.random() < np.exp(-delta / T))

            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            if accept:
                current, current_val = candidate, candidate_val

                if candidate_val < best_val:
                    best, best_val = candidate[:], candidate_val
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            if best > candidate[:]:
                print(
                    f"Iter {iteration:4d} | T={T:.3f} | best={best_val:.3f} | "
                    f"x={current} | {minutes:02d}min {seconds:02d}s\r"
                )
            # else:
            #     print(
            #         f"Iter {iteration:4d} | T={T:.3f} | best={best_val:.3f} | "
            #         f"x={candidate[:]} | {minutes:02d}min {seconds:02d}s\r"
            #     )
            iteration += 1

        T *= cooling_rate
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"Optimization finished.Total iter = {iteration}")
    print(f"Best solution: f={best_val:.3f}, x={best}| {minutes:02d}min {seconds:02d}s\n")
    return best, float(best_val)
