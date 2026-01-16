import os
from global_parameters import *
import numpy as np
import seaborn as sns
# from optimization import *
from gurobipy import GRB
import re
import pandas as pd
from data_import import  *

def reinforcement_details(choix):
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


def natural_sort_key(s):
    # split string into parts: digits as numbers, other as strings
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def df_from_model(m):
    if m.status == GRB.OPTIMAL:
        records = []
        for v in m.getVars():
            if '[' in v.VarName:
                base, idx_str = v.VarName.split('[', 1)
                idx = [int(x) for x in idx_str[:-1].split(',')]
            else:
                base, idx = v.VarName, []

            # assume last index is always Time
            if len(idx) == 2:
                entity, t = idx
                colname = f"{base}[{entity}]"
            elif len(idx) == 1:  # if variable has only time
                t = idx[0]
                colname = base
            else:  # scalar (no time)
                t = 0
                colname = base

            records.append((t, colname, v.X))

        # Build dataframe
        df = pd.DataFrame(records, columns=["Time", "Variable", "Value"])
        df = df.pivot(index="Time", columns="Variable", values="Value").reset_index()
        df = df.drop(df.columns[0], axis=1)

        # natural sort the columns
        df = df[sorted(df.columns, key=natural_sort_key)]
        return df
    else:
        print(f"Model status is not optimal (status={m.status})")
        return None

def save_M_df(M_df, folder="results"):
    """
    Saves each DataFrame in M_df as a CSV file named:
    solution_thXX_pvYY.csv
    """
    os.makedirs(folder, exist_ok=True)
    th_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    pv_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    for i, th in enumerate(th_values[:len(M_df)]):
        for j, pv in enumerate(pv_values[:len(M_df[0])]):
            df = M_df[i][j]

            # Build filename with consistent formatting
            th_str = f"{th:.2f}".replace(".", "")
            pv_str = f"{pv:.2f}".replace(".", "")

            filename = f"solution_th{th_str}_pv{pv_str}.csv"
            path = os.path.join(folder, filename)

            df.to_csv(path, index=False)

    print(f"Saved {len(th_values) * len(pv_values)} DataFrames in '{folder}'")


def save_M_x(M_x, folder="results_Mx"):
    """
    Saves each 2D slice M_x[i][j] as a CSV file
    named using the same scheme as save_M_df():
    solution_thXX_pvYY.csv
    """
    os.makedirs(folder, exist_ok=True)

    th_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    pv_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    for i, th in enumerate(th_values[:len(M_x)]):
        for j, pv in enumerate(pv_values[:len(M_x[0])]):

            mat = M_x[i][j]           # this is a 2D list
            df = pd.DataFrame(mat)    # convert to DataFrame

            th_str = f"{th:.2f}".replace(".", "")
            pv_str = f"{pv:.2f}".replace(".", "")

            filename = f"x_th{th_str}_pv{pv_str}.csv"
            path = os.path.join(folder, filename)

            df.to_csv(path, index=False)

    print(f"Saved {len(th_values) * len(pv_values)} matrices in '{folder}'")


def load_M_x(folder="results_Mx"):
    """
    Loads CSV files saved by save_M_x()
    and returns them as a 3D list-of-lists-of-lists,
    cleaning out missing entries.
    """
    th_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    pv_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    # 2D structure
    M_x_loaded = [[None for _ in pv_values] for _ in th_values]

    for i, th in enumerate(th_values):
        for j, pv in enumerate(pv_values):

            th_str = f"{th:.2f}".replace(".", "")
            pv_str = f"{pv:.2f}".replace(".", "")

            filename = f"x_th{th_str}_pv{pv_str}.csv"
            path = os.path.join(folder, filename)

            if os.path.exists(path):
                df = pd.read_csv(path)
                M_x_loaded[i][j] = df.squeeze(axis=1).tolist()

    # 🔥 Remove None entries inside rows
    cleaned = [
        [cell for cell in row if cell is not None]
        for row in M_x_loaded
    ]

    # 🔥 Remove empty rows
    cleaned = [row for row in cleaned if len(row) > 0]

    return cleaned


def load_M_df(folder="results"):
    """
    Loads CSVs into a 2D list-of-lists and removes all None entries.
    """
    th_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    pv_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    M_df_loaded = [[None for _ in pv_values] for _ in th_values]

    for i, th in enumerate(th_values):
        for j, pv in enumerate(pv_values):

            th_str = f"{th:.2f}".replace(".", "")
            pv_str = f"{pv:.2f}".replace(".", "")

            filename = f"solution_th{th_str}_pv{pv_str}.csv"
            path = os.path.join(folder, filename)

            if os.path.exists(path):
                M_df_loaded[i][j] = pd.read_csv(path)

    # 🔥 Remove every None entry in each row
    cleaned = [
        [cell for cell in row if cell is not None]
        for row in M_df_loaded
    ]

    # 🔥 Remove empty rows completely (optional)
    cleaned = [row for row in cleaned if len(row) > 0]

    return cleaned


def joule_from_df(df, x):
    setT = range(24)
    Cost_joules = 0.1
    i_act = 0.035
    R, X, Lmax, CAPEX = reinforcement_details(x)
    daily_J_losses = sum(df[f"I_squared[{i}]"][t] * R[i]
                         for i in range(len(R)) for t in setT)
    J_cost_total = daily_J_losses * Cost_joules * 365 * 30000
    J_Cost_act = sum((J_cost_total) / ((1 + i_act) ** n) for n in range(31))
    return (J_Cost_act)


def curt_from_df(df, x, coeff_pv_prod):
    setT = range(24)
    Cost_curt = 0.083
    i_act = 0.035
    (B,  elecLoadProfilePMatrix, elecLoadProfileQMatrix, elecProdProfilePMatrix, N) = import_elec_data(
        'data/linePREDIS.csv', 'data/loadprofileP.csv', 'data/loadprofileQ.csv',
        'data/ElecprodProfileP.csv')
    ePPMatrix = elecProdProfilePMatrix.T
    ePPMatrix[9, :] = coeff_pv_prod * ePPMatrix[9, :]
    ePPMatrix[10, :] = coeff_pv_prod * ePPMatrix[10, :]

    daily_curt = sum(ePPMatrix[i, t] - df[f"P_gen[{i}]"][t]
                     for i in [9, 10] for t in setT) * 365 * 30000 * Cost_curt
    curt_act = sum((daily_curt) / ((1 + i_act) ** n) for n in range(31))
    return (curt_act)


def co2_from_df(df):
    P_slack_minus = []
    P_slack_plus = []

    co2_grid = 0.032  # kg/kWh
    co2_gaz = 0.583  # kg/kWh
    co2_PV = 0.043  # kg/kWh
    setT = range(24)
    for i in range(12):
        P_slack_minus_i = []
        P_slack_plus_i = []
        for t in setT:
            if df[f"P_slack[{i}]"][t] < 0:
                # print(m.getVarByName(f"P_slack[{i},{t}]").X)
                P_slack_minus_i += [df[f"P_slack[{i}]"][t]]
                P_slack_plus_i += [0]
            else:
                P_slack_plus_i += [df[f"P_slack[{i}]"][t]]
                P_slack_minus_i += [0]
        P_slack_minus += [P_slack_minus_i]
        P_slack_plus += [P_slack_plus_i]

    CO2_daily = (co2_grid * sum(P_slack_plus[i][t] for i in [0, 1, 8] for t in setT)
                 +
                 co2_gaz * sum(df[f"p_th_gen[11]"][t] + df[f"P_gen[11]"]
            [t] + P_slack_minus[1][t] for t in setT)
                 +
                 co2_PV * sum(df[f"P_gen[9]"][t] + df[f"P_gen[10]"]
            [t] + P_slack_minus[8][t] for t in setT)
                 ) * 30_000
    # indirect emissions

    CO2_total_emission = CO2_daily * 356 * 30
    return CO2_total_emission


def capex_from_x(x):
    R, X, Lmax, CAPEX = reinforcement_details(x)
    return (sum(CAPEX))


def M_CO2_from_M_df(M_df, M_x, M_coeffs):
    return [[co2_from_df(M_df[i][j]) for j in range(len(M_df[0]))] for i in range(len(M_df))]


def M_CAPEX_from_M_df(M_df, M_x, M_coeffs):
    return [[capex_from_x(M_x[i][j]) for j in range(len(M_df[0]))] for i in range(len(M_df))]


def M_JOULES_from_M_df(M_df, M_x, M_coeffs):
    return [[joule_from_df(M_df[i][j], M_x[i][j]) for j in range(len(M_df[0]))] for i in range(len(M_df))]


def M_CURT_from_M_df(M_df, M_x, M_coeffs):
    return [[curt_from_df(M_df[i][j], M_x[i][j], M_coeffs[i][j][1]) for j in range(len(M_df[0]))] for i in
            range(len(M_df))]


def M_TOTEX_from_M_df(M_df, M_x, M_coeffs):
    M_curt = M_CURT_from_M_df(M_df, M_x, M_coeffs)

    M_joules = M_JOULES_from_M_df(M_df, M_x, M_coeffs)

    M_capex = M_CAPEX_from_M_df(M_df, M_x, M_coeffs)
    M_totex = [[M_curt[i][j] + M_joules[i][j] + M_capex[i][j]
                for j in range(len(M_df[0]))] for i in range(len(M_df))]
    return (M_totex)


def plot_all_5_df(M_df, M_x, name):
    coeff_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    M_coeffs = []
    for a in coeff_list:
        L_coeffs = []
        for b in coeff_list:
            L_coeffs += [(a, b)]
        M_coeffs += [L_coeffs]

    M_curt = M_CURT_from_M_df(M_df, M_x, M_coeffs)
    M_joules = M_JOULES_from_M_df(M_df, M_x, M_coeffs)
    M_capex = M_CAPEX_from_M_df(M_df, M_x, M_coeffs)
    M_totex = [[M_curt[i][j] + M_joules[i][j] + M_capex[i][j]
                for j in range(len(M_df[0]))] for i in range(len(M_df))]

    M_CO2 = M_CO2_from_M_df(M_df, M_x, M_coeffs)

    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row - 3 plots
    data_list = [
        ([[M_totex[i][j] / 1_000_000 for j in range(len(M_df[0]))] for i in range(len(M_df))], "(a) TOTEX (M€)", 13),
        ([[M_capex[i][j] / 1_000_000 for j in range(len(M_df[0]))] for i in range(len(M_df))], "(b) CAPEX (M€)", 13),
        ([[M_CO2[i][j] / 1_000_000_000 for j in range(len(M_CO2[0]))] for i in range(len(M_CO2))],
         "(c) Total CO2 emissions (Mteq)", 3),
    ]

    # Bottom row - 2 plots
    data_list += [
        ([[M_joules[i][j] / 1_000_000 for j in range(len(M_df[0]))] for i in range(len(M_df))],
         "(d) Actualized Joule losses (M€)", 6),
        ([[M_curt[i][j] / 1_000_000 for j in range(len(M_df[0]))] for i in range(len(M_df))],
         "(e) Actualized curtailment cost (M€)", 6),
    ]

    tick_labels = ["", 0.6, "", 0.8, "", 1.0, "", 1.2, "", 1.4, "", 1.6]
    full_ticks = np.linspace(0.5, 11.5, 12)
    tick_positions = full_ticks
    tick_labels_half = tick_labels

    # Plot top row (3 plots)
    for idx, (data, title, vmax) in enumerate(data_list[:3]):
        ax = axes[0, idx]
        sns.heatmap(data, cmap="coolwarm", annot=False, vmin=0, vmax=vmax, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_half, fontsize=10)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels_half, fontsize=10)
        ax.set_xlabel(r'$\beta_{pv}$', fontsize=14)
        ax.set_ylabel(r'$\beta_{th}$', fontsize=14)

        # Add asterisks
        for i in range(len(M_capex)):
            for j in range(len(M_capex[0])):
                if M_capex[i][j] != 0:
                    ax.text(j + 0.05, i + 0.95, "*", color="black", ha="left", va="bottom", fontsize=12)

    # Plot bottom row (2 plots, centered)
    for idx, (data, title, vmax) in enumerate(data_list[3:]):
        ax = axes[1, idx]
        sns.heatmap(data, cmap="coolwarm", annot=False, vmin=0, vmax=vmax, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_half, fontsize=10)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels_half, fontsize=10)
        ax.set_xlabel(r'$\beta_{pv}$', fontsize=14)
        ax.set_ylabel(r'$\beta_{th}$', fontsize=14)

        # Add asterisks
        for i in range(len(M_capex)):
            for j in range(len(M_capex[0])):
                if M_capex[i][j] != 0:
                    ax.text(j + 0.05, i + 0.95, "*", color="black", ha="left", va="bottom", fontsize=12)

    # Hide the 6th subplot (bottom right)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()