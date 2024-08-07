import numpy as np
import scipy.sparse as sparse
from scipy.io import loadmat
import sim_mpc_OSQP

# Load data from .mat files
system_path = '../data/mpc_data_02092022_nd8.mat'
mat_data = loadmat(system_path)

fname_RM = '../orms/GoldenBPMResp_DIAD.mat'
RM_data = loadmat(fname_RM)
#initialise parameters
n_samples = 100
n_delay = 9
RMorigx = RM_data['Rmat'][0][0][0]
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










