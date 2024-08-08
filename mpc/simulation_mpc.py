import numpy as np
import scipy.sparse as sparse
from scipy.io import loadmat
import sim_mpc_OSQP
import diamond_I_configuration_v5 as DI

# Load data from .mat files
system_path = '../data/mpc_data_02092022_nd8.mat'
mat_data = loadmat(system_path)

fname_RM = '../orms/GoldenBPMResp_DIAD.mat'
RM_data = loadmat(fname_RM)
#initialise parameters
n_samples = 100
n_delay = 9
RMorigx = RM_data['Rmat'][0][0][0]
square_config = True
id_to_bpm_x, id_to_cm_x, id_to_bpm_y, id_to_cm_y = DI.diamond_I_configuration_v5(RMorigx, RMorigx, square_config)
RMx = RMorigx[np.ix_(id_to_bpm_x, id_to_cm_x)]
#Current issue, RM matrix is too small 167x167
#Shrinking state transitions a solution? Currently 168x168


TOT_BPM = np.size(RMorigx,0)
dist = np.ones((TOT_BPM,1)) * np.ones((1,n_samples)) * 10
Ap = mat_data['Ap_x']
Bp = mat_data['Bp_x']
Cp = mat_data['Cp_x']
Ao = mat_data['Ao_x']
Bo = mat_data['Bo_x']
Co = mat_data['Co_x']
Ad = mat_data['Ad_x']
Cd = mat_data['Cd_x']
LxN_obs = mat_data['Kfx_x']
Lxd_obs = mat_data['Kfd_x']
P_mpc = mat_data['P_x']
R_mpc = mat_data['Rlqr_x']
J_mpc = np.transpose(Bo) @ P_mpc @ Ao + R_mpc
ny = np.size(RMx, 0)
nu = np.size(RMx, 1)
nx = nu

S_sp_pinv = np.linalg.pinv( np.block([
    [np.eye(nx) - Ao, -Bo],
    [Co, np.zeros((ny,nu))]
]) )


q_mat_x0 = np.transpose(Bo) @ P_mpc @ Ao
q_mat_xd =(-(np.hstack( ((np.transpose(Bo) @ P_mpc) , R_mpc))) @ np.vstack((S_sp_pinv_x,S_sp_pinv_u))@ -Cd

)








