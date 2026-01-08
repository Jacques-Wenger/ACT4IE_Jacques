import numpy as np
from temp_linearization import *
def import_thermal_data(tLoadProfile_name, thermalGrid_file, thermalNodes_file):
    tLoadMatrix = np.loadtxt(tLoadProfile_name, delimiter=';')
    thermalGridMatrix = np.loadtxt(thermalGrid_file, delimiter=',')
    thermalNodesMatrix = np.loadtxt(thermalNodes_file, delimiter=";")
    buses = tLoadMatrix.shape[1]
    th_lines = thermalGridMatrix.shape[0]
    Bh = np.zeros((buses, th_lines))
    for i in range(th_lines):
        Bh[int(thermalGridMatrix[i, 0]), i] = 1
        Bh[int(thermalGridMatrix[i, 1]), i] = -1
    thermal_losses_coeffs = regression_lineraire_cold_hot()

    return Bh, tLoadMatrix, thermalGridMatrix, thermalNodesMatrix,thermal_losses_coeffs


def import_elec_data(lineFile, elecLoadProfileP, elecLoadProfileQ, elecProdProfileP):
    lineMatrix = np.loadtxt(lineFile, delimiter=';')
    elecLoadProfilePMatrix = np.loadtxt(elecLoadProfileP, delimiter=';')
    elecLoadProfileQMatrix = np.loadtxt(elecLoadProfileQ, delimiter=';')
    elecProdProfilePMatrix = np.loadtxt(elecProdProfileP, delimiter=';')
    V = elecLoadProfilePMatrix.shape[1]
    E = lineMatrix.shape[0]
    B = np.zeros((V, E))
    for i in range(E):
        if lineMatrix[i, 5] == 1:
            B[int(lineMatrix[i, 0]), i] = 1
            B[int(lineMatrix[i, 1]), i] = -1
    N = np.eye(V) #slack matrix
    N[0, 0] = 0
    N[1, 1] = 0
    N[8, 8] = 0
    return B,  elecLoadProfilePMatrix, elecLoadProfileQMatrix, elecProdProfilePMatrix, N