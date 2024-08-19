import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim

import numpy as np
import numpy.random as npr

import os
import matplotlib.pyplot as plt

import pickle as pkl
import scipy.sparse as sparse
from scipy.io import loadmat
import sim_mpc as sim
import diamond_I_configuration_v5 as DI
import pandas as pd
import time
start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


size_file_path = '../data/sizes.pkl'
with open(size_file_path, 'rb') as f:
    sizes = pkl.load(f)

n_state = sizes['n_state']
n_ctrl = sizes['n_ctrl']
hidden_size = sizes['hidden_size']
n_sc = sizes['n_sc']

model_path = '../data/model/model.ckpt'
nnparams = torch.load(model_path)

# options
fname_RM = '../orms/GoldenBPMResp_DIAD.mat'
pick_dir = 1
dirs = ['horizontal','vertical']
pick_direction = dirs[pick_dir]
do_step = True
sim_IMC = False
use_FGM = True

#Hardlimits
fname_correctors = '../data/corrector_data.csv'
correctors = pd.read_csv(fname_correctors)
hardlimits = correctors['MaxAmps'].iloc[:172]
hardlimits = hardlimits.to_numpy().reshape(172, 1)

#Configure Diamond-I storage ring
Rmat = loadmat(fname_RM)
RMorigx = Rmat['Rmat'][0][0][0]
ny_x = np.size(RMorigx, 0)
nu_x = np.size(RMorigx, 1) 
RMorigy = Rmat['Rmat'][1][1][0]
ny_y = np.size(RMorigy, 0) 
nu_y = np.size(RMorigy, 1)
assert ny_x == ny_y
assert nu_x == nu_y
TOT_BPM = np.size(RMorigx, 0)
TOT_CM = np.size(RMorigx, 1)
square_config = True
id_to_bpm_x, id_to_cm_x, id_to_bpm_y, id_to_cm_y = DI.diamond_I_configuration_v5(RMorigx, RMorigy, square_config)

#first n_include BPMs and CMs active for testing

n_include = 8


id_to_bpm_x = id_to_bpm_x[:n_include]
id_to_cm_x = id_to_cm_x[:n_include]
id_to_bpm_y = id_to_bpm_y[:n_include]
id_to_cm_y = id_to_cm_y[:n_include]


RMx = RMorigx[np.ix_(id_to_bpm_x, id_to_cm_x)]
RMy = RMorigy[np.ix_(id_to_bpm_y, id_to_cm_y)]



#Observer and Regulator
n_delay = 8

fname = f'../data/mpc_data_13092022_nd{n_delay}.mat'


mat_data = loadmat(fname)


Fs = 10000
Ts = 1/Fs
# fname = fname = f'../data/mpc_data_13092022_nd{n_delay}.mat'
# mat_data.update(loadmat(fname))

if pick_direction == 'vertical':
    id_to_bpm = id_to_bpm_y
    id_to_cm = id_to_cm_y
    RM = RMy
    aI_Hz = 700
    #Observer
    Ao = mat_data['Ao_y'][:n_include,:n_include]
    Bo = mat_data['Bo_y'][:n_include,:n_include]
    Co = mat_data['Co_y'][:n_include,:n_include]
    Ad = mat_data['Ad_y'][:n_include,:n_include]
    Cd = mat_data['Cd_y'][:n_include,:n_include]
    #Plant with all BPMs and CMs
    Ap = mat_data['Ap_y']
    Bp = mat_data['Bp_y']
    Cp = mat_data['Cp_y']
    Kfd = mat_data['Kfd_y'][:n_include,:n_include] #Observer gain for disturbance
    Kfx = mat_data['Kfx_y'][:n_include,:n_include] #Observer gain for state
    P_mpc = mat_data['P_y'][:n_include,:n_include] #Terminal cost
    Q_mpc = mat_data['Qlqr_y'][:n_include,:n_include] #State weighting
    R_mpc = mat_data['Rlqr_y'][:n_include,:n_include] #Input weighting
    #SOFB

else:
    id_to_bpm = id_to_bpm_x
    id_to_cm = id_to_cm_x
    RM = RMx
    aI_Hz = 500
    #Observer
    Ao = mat_data['Ao_x'][:n_include,:n_include]
    Bo = mat_data['Bo_x'][:n_include,:n_include]
    Co = mat_data['Co_x'][:n_include,:n_include]
    Ad = mat_data['Ad_x'][:n_include,:n_include]
    Cd = mat_data['Cd_x'][:n_include,:n_include]
    #Plant with all BPMs and CMs
    Ap = mat_data['Ap_x']
    Bp = mat_data['Bp_x']
    Cp = mat_data['Cp_x']
    Kfd = mat_data['Kfd_x'][:n_include,:n_include]
    Kfx = mat_data['Kfx_x'][:n_include,:n_include]
    P_mpc = mat_data['P_x'][:n_include,:n_include]
    Q_mpc = mat_data['Qlqr_x'][:n_include,:n_include]
    R_mpc = mat_data['Rlqr_x'][:n_include,:n_include]
    #SOFB

ny = np.size(RM, 0)
nu = np.size(RM, 1)
nx = nu

#Observer
Lxd_obs = Kfd
Lx8_obs = Kfx
S_sp_pinv = np.linalg.pinv(np.block([[np.eye(nx) - Ao, -Bo], [Co, np.zeros((ny, nu))]]))
S_sp_pinv = S_sp_pinv[:,nx:]

#MPC
horizon = 1
u_rate_scalar = 1*1000
u_rate = np.ones((nu, 1)) * u_rate_scalar
u_max = hardlimits[id_to_cm] * 1000
y_max_scalar = 200
y_max = np.ones((id_to_bpm.size, 1)) * y_max_scalar
J_mpc = np.transpose(Bo) @ P_mpc @ Bo + R_mpc
S_sp_pinv_x = S_sp_pinv[:nx,:]
S_sp_pinv_u = S_sp_pinv[nx:,:]
q_mat_x0 = np.transpose(Bo) @ P_mpc @ Ao
q_mat_xd = (np.hstack((Bo.T @ P_mpc, R_mpc)) @ np.vstack((S_sp_pinv_x, S_sp_pinv_u)) @ Cd)
q_mat = np.hstack((q_mat_x0, q_mat_xd))

#Set up FGM
beta_fgm = 0

if use_FGM:
    eigmax = np.max(np.linalg.eigvals(J_mpc))
    eigmin = np.min(np.linalg.eigvals(J_mpc))   
    J_mpc = np.eye(J_mpc.shape[0]) - J_mpc / eigmax
    beta_fgm = (np.sqrt(eigmax) - np.sqrt(eigmin)) / (np.sqrt(eigmax) + np.sqrt(eigmin))
    q_mat = np.hstack((q_mat_x0, q_mat_xd)) / eigmax


#Rate limiter on VME processors
if pick_direction == 'vertical':
    mat_data.update(loadmat('../data/awrSSy.mat'))

else:
    mat_data.update(loadmat('../data/awrSSx.mat'))

#Measurement Data
if do_step:
    n_samples = 2000
    doff = np.ones((TOT_BPM, 1)) * np.ones((1, n_samples)) * 10
    don = doff

else:
    print("no data at this time")

n_samples = 6000

#Simulates multiple modes of disturbance to get training data
train = False

trainmodes = [1,2,3,4,5,6]

n_tests = len(trainmodes)

u_sim_train = np.zeros((n_samples * n_tests, n_include))
xd_obs_train = np.zeros((n_samples * n_tests, n_include))
x0_obs_train = np.zeros((n_samples * n_tests, n_include))
y_sim_train = np.zeros((n_samples * n_tests , n_include))


if train:

    for imode in trainmodes:
        k = 1

        if pick_direction == 'vertical':
            UR, SR, VR = np.linalg.svd(RMy)
        else:
            UR, SR, VR = np.linalg.svd(RMx)

        mag_u = 10
        tmp = UR[:, imode - 1] * SR[imode - 1] * mag_u
        doff_tmp = np.zeros((TOT_BPM, 1))
        doff_tmp[id_to_bpm] = tmp[:, np.newaxis]
        doff = doff_tmp * np.ones((1,n_samples))
        don = doff
        y_max = np.ones((id_to_bpm.size, 1)) * 850
        #Simulation

        endt = n_samples*Ts - Ts
        Lsim = n_samples*Ts
        t = np.arange(0, endt + Ts, Ts)

        SOFB_setpoints = 0* np.array([0.01619904302060604, -0.3885023295879364, 0.6992301344871521, -1.7483879327774048, 2.3418004512786865, -1.5496907234191895, 1.1642824411392212, -1.0826029777526855, -3.5085201263427734, 2.117027997970581, -0.7213708162307739, -0.016367638483643532, -0.8390787839889526, 0.9059506058692932, -0.44695019721984863, -0.6361433267593384, 0.5771667957305908, -2.4178178310394287, 1.3213233947753906, -0.41284069418907166, -0.01976143568754196, 0.28217950463294983, 0.617164134979248, -0.7120676636695862, 1.0387179851531982, 0.4763958752155304, -0.5612682104110718, -0.6951331496238708, 0.02955007180571556, 0.11709681898355484, -0.044195499271154404, -0.45408889651298523, 0.38928934931755066, -0.3819846510887146, -0.11148449033498764, 0.3617706894874573, -0.4519191384315491, -0.20117510855197906, -0.6386896371841431, 0.7802280187606812, -0.22534973919391632, -1.4564428329467773, 1.1819654703140259, -0.7721765637397766, 0.6452361941337585, 1.4174635410308838, -1.711761713027954, 0.6994525194168091, -0.8488370776176453, 1.4066176414489746, 0.07518509030342102, -0.070978082716465, -1.5602132081985474, -0.5684106945991516, 2.698157548904419, -0.6943356990814209, -0.13617253303527832, 0.0965278148651123, -0.4131055474281311, 1.202785849571228, 0.6013646721839905, -2.7761335372924805, 0.7537578344345093, -0.4265475571155548, 2.0365219116210938, -5, 0.4749950170516968, -0.6516678929328918, 0.4227065145969391, -0.342603862285614, 0.27613964676856995, -0.6612204313278198, 0.8361185193061829, 0.4531811773777008, -0.8470264673233032, 1.780594825744629, -1.05325448513031, -0.3227340281009674, 0.2727269232273102, 0.3892798125743866, -0.42446842789649963, 0.97515869140625, -0.010264929383993149, -1.4114340543746948, 0.8374415040016174, -0.3478398621082306, 0.3624727129936218, -0.4587855637073517, 0.7216208577156067, -1.558065414428711, 1.8873403072357178, -0.8834448456764221, 0.4664323329925537, 0.13014689087867737, -1.6719253063201904, 0.13990825414657593, -0.062160685658454895, 0.30241018533706665, -1.0267870426177979, 2.59081768989563, -0.38700228929519653, -0.176784947514534, 0.65232914686203, -0.5602529644966125, 0.9080629944801331, -1.3520822525024414, 0.5552564263343811, 0.6984527707099915, -0.4484162926673889, -0.9921460747718811, 1.0305604934692383, -1.0470114946365356, -0.5739063620567322, 2.045651912689209, -0.7531506419181824, -0.05252191796898842, 0.23519375920295715, 0.2581924498081207, 0.7864719033241272, -0.04462754726409912, -0.7147292494773865, -0.3432796895503998, 0.1804860681295395, -0.1353411078453064, -0.3737372159957886, 0.03207216039299965, 1.3494750261306763, -1.823861002922058, 1.0277291536331177, -0.6163777112960815, -0.30168670415878296, 0.0707305371761322, 0.496951699256897, -0.13955195248126984, -0.736243486404419, 0.7516483068466187, -0.19038543105125427, -0.06523245573043823, -0.02522803097963333, 0.26334622502326965, -0.4521408975124359, 1.0672394037246704, 0.1539207249879837, -0.3190993666648865, -0.2580198645591736, 0.21354863047599792, 1.6735446453094482, -2.5382795333862305, 2.207550287246704, -0.5185757875442505, 0.3553980588912964, 0.691810131072998, -0.9267514944076538, -0.46967199444770813, 1.5809837579727173, -3.0611188411712646, 0.9965386390686035, 0.07120700925588608, 0.05444963276386261, -0.4420344829559326, 0.759113609790802, -0.33723288774490356, -0.0011737275635823607, 0.19979232549667358, -0.414171427488327, -1.1921398639678955, 1.0765595436096191, -0.31856679916381836, -1.37770414352417, 2.4016101360321045, 0.011450201272964478, -0.6292238831520081])
        SOFB_setpoints = np.zeros((1,172))
        SOFB_setp = np.transpose(SOFB_setpoints[:,id_to_cm])
        SOFB_setp = np.where(SOFB_setp > u_max, u_max, SOFB_setp)
        SOFB_setp = np.where(SOFB_setp < -u_max, -u_max, SOFB_setp)


        mpc = sim.Mpc(
            n_samples, n_delay, doff,
            Ap, Bp, Cp, 
            Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
            J_mpc, q_mat, y_max,
            u_max, u_rate,
            id_to_bpm, id_to_cm,
            mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
            SOFB_setp, beta_fgm)



        y_sim_fgm ,u_sim_fgm, x0_obs_fgm, xd_obs_fgm = mpc.sim_mpc(use_FGM)
        u_sim_train[(k-1)*n_samples:k*n_samples, :] = u_sim_fgm[:,id_to_cm]
        xd_obs_train[(k-1)*n_samples:k*n_samples, :] = xd_obs_fgm
        x0_obs_train[(k-1)*n_samples:k*n_samples, :] = x0_obs_fgm
        y_sim_train[(k-1)*n_samples:k*n_samples, :] = y_sim_fgm[:,id_to_bpm]
        k+=1

else:
    if pick_direction == 'vertical':
        UR, SR, VR = np.linalg.svd(RMy)
    else:
        UR, SR, VR = np.linalg.svd(RMx)

    imode =1
    mag_u = 10
    tmp = UR[:, imode - 1] * SR[imode - 1] * mag_u
    doff_tmp = np.zeros((TOT_BPM, 1))
    doff_tmp[id_to_bpm] = tmp[:, np.newaxis]
    doff = doff_tmp * np.ones((1,n_samples))
    don = doff
    y_max = np.ones((id_to_bpm.size, 1)) * 850
    #Simulation

    endt = n_samples*Ts - Ts
    Lsim = n_samples*Ts
    t = np.arange(0, endt + Ts, Ts)

    SOFB_setpoints = 0* np.array([0.01619904302060604, -0.3885023295879364, 0.6992301344871521, -1.7483879327774048, 2.3418004512786865, -1.5496907234191895, 1.1642824411392212, -1.0826029777526855, -3.5085201263427734, 2.117027997970581, -0.7213708162307739, -0.016367638483643532, -0.8390787839889526, 0.9059506058692932, -0.44695019721984863, -0.6361433267593384, 0.5771667957305908, -2.4178178310394287, 1.3213233947753906, -0.41284069418907166, -0.01976143568754196, 0.28217950463294983, 0.617164134979248, -0.7120676636695862, 1.0387179851531982, 0.4763958752155304, -0.5612682104110718, -0.6951331496238708, 0.02955007180571556, 0.11709681898355484, -0.044195499271154404, -0.45408889651298523, 0.38928934931755066, -0.3819846510887146, -0.11148449033498764, 0.3617706894874573, -0.4519191384315491, -0.20117510855197906, -0.6386896371841431, 0.7802280187606812, -0.22534973919391632, -1.4564428329467773, 1.1819654703140259, -0.7721765637397766, 0.6452361941337585, 1.4174635410308838, -1.711761713027954, 0.6994525194168091, -0.8488370776176453, 1.4066176414489746, 0.07518509030342102, -0.070978082716465, -1.5602132081985474, -0.5684106945991516, 2.698157548904419, -0.6943356990814209, -0.13617253303527832, 0.0965278148651123, -0.4131055474281311, 1.202785849571228, 0.6013646721839905, -2.7761335372924805, 0.7537578344345093, -0.4265475571155548, 2.0365219116210938, -5, 0.4749950170516968, -0.6516678929328918, 0.4227065145969391, -0.342603862285614, 0.27613964676856995, -0.6612204313278198, 0.8361185193061829, 0.4531811773777008, -0.8470264673233032, 1.780594825744629, -1.05325448513031, -0.3227340281009674, 0.2727269232273102, 0.3892798125743866, -0.42446842789649963, 0.97515869140625, -0.010264929383993149, -1.4114340543746948, 0.8374415040016174, -0.3478398621082306, 0.3624727129936218, -0.4587855637073517, 0.7216208577156067, -1.558065414428711, 1.8873403072357178, -0.8834448456764221, 0.4664323329925537, 0.13014689087867737, -1.6719253063201904, 0.13990825414657593, -0.062160685658454895, 0.30241018533706665, -1.0267870426177979, 2.59081768989563, -0.38700228929519653, -0.176784947514534, 0.65232914686203, -0.5602529644966125, 0.9080629944801331, -1.3520822525024414, 0.5552564263343811, 0.6984527707099915, -0.4484162926673889, -0.9921460747718811, 1.0305604934692383, -1.0470114946365356, -0.5739063620567322, 2.045651912689209, -0.7531506419181824, -0.05252191796898842, 0.23519375920295715, 0.2581924498081207, 0.7864719033241272, -0.04462754726409912, -0.7147292494773865, -0.3432796895503998, 0.1804860681295395, -0.1353411078453064, -0.3737372159957886, 0.03207216039299965, 1.3494750261306763, -1.823861002922058, 1.0277291536331177, -0.6163777112960815, -0.30168670415878296, 0.0707305371761322, 0.496951699256897, -0.13955195248126984, -0.736243486404419, 0.7516483068466187, -0.19038543105125427, -0.06523245573043823, -0.02522803097963333, 0.26334622502326965, -0.4521408975124359, 1.0672394037246704, 0.1539207249879837, -0.3190993666648865, -0.2580198645591736, 0.21354863047599792, 1.6735446453094482, -2.5382795333862305, 2.207550287246704, -0.5185757875442505, 0.3553980588912964, 0.691810131072998, -0.9267514944076538, -0.46967199444770813, 1.5809837579727173, -3.0611188411712646, 0.9965386390686035, 0.07120700925588608, 0.05444963276386261, -0.4420344829559326, 0.759113609790802, -0.33723288774490356, -0.0011737275635823607, 0.19979232549667358, -0.414171427488327, -1.1921398639678955, 1.0765595436096191, -0.31856679916381836, -1.37770414352417, 2.4016101360321045, 0.011450201272964478, -0.6292238831520081])
    SOFB_setpoints = np.zeros((1,172))
    SOFB_setp = np.transpose(SOFB_setpoints[:,id_to_cm])
    SOFB_setp = np.where(SOFB_setp > u_max, u_max, SOFB_setp)
    SOFB_setp = np.where(SOFB_setp < -u_max, -u_max, SOFB_setp)


    mpc = sim.Mpc(
        n_samples, n_delay, doff,
        Ap, Bp, Cp, 
        Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
        J_mpc, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
        SOFB_setp, beta_fgm)



    y_sim_fgm ,u_sim_fgm, x0_obs_fgm, xd_obs_fgm = mpc.sim_mpc(use_FGM)

#Save training data

np.savez('../data/simresults.npz' , y_sim=y_sim_train, u_sim=u_sim_train, x0_obs=x0_obs_train, xd_obs=xd_obs_train)

#Toggle to simulate with nn controller and graph comparisons

compare = False

if compare:
    mpc = sim.Mpc(
        n_samples, n_delay, doff,
        Ap, Bp, Cp, 
        Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
        J_mpc, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
        SOFB_setp, beta_fgm)
    y_sim_nn ,u_sim_nn, x0_obs_nn, xd_obs_nn = mpc.sim_nn(n_state, hidden_size, n_ctrl, nnparams, device)

    y_err = (y_sim_fgm - y_sim_nn) 
    u_err = (u_sim_fgm - u_sim_nn) 





#Unpack Loss Data

loss_data = np.load('../data/model/modelloss.npz')


# Plotting


scale_u = 0.001

fig, axs = plt.subplots(2, 5, figsize=(15, 8))

# Subplot 1: Disturbance
axs[0, 0].plot(doff[id_to_bpm, :].T)
axs[0, 0].set_title('Disturbance')

# Subplot 2: Disturbance Mode Space
axs[0, 1].plot((UR.T @ doff[id_to_bpm, :]).T)
axs[0, 1].set_title('Disturbance Mode Space')

# Subplot 3: Input
axs[0, 2].plot(u_sim_fgm[:, id_to_cm] * scale_u, linestyle='-')  # solid line for u_sim_fgm
if compare:
    axs[0, 2].plot(u_sim_nn[:, id_to_cm] * scale_u, linestyle='--')  # dashed line for u_sim_nn
axs[0, 2].set_title('Input')

# # Subplot 4: % Error in Input
if compare:
    axs[0, 3].plot(u_err[:, id_to_cm] * scale_u, linestyle='-')  # solid line for u_sim_fgm
    axs[0, 3].set_title('Input Error')

# # Subplot 5: Input Mode Space
axs[1, 0].plot(scale_u * u_sim_fgm[:, id_to_cm] @ VR, linestyle='-')  # solid line for u_sim_fgm
if compare:
    axs[1, 0].plot(scale_u * u_sim_nn[:, id_to_cm] @ VR, linestyle='--')  # dashed line for u_sim_nn
axs[1, 0].set_title('Input Mode Space')

# Subplot 6: Output
axs[1, 1].plot(y_sim_fgm[:, id_to_bpm], linestyle='-')  # solid line for y_sim_fgm
if compare:
    axs[1, 1].plot(y_sim_nn[:, id_to_bpm], linestyle='--')  # dashed line for y_sim_nn
axs[1, 1].set_title('Output')

# Subplot 7: Output Mode Space
axs[1, 2].plot(y_sim_fgm[:, id_to_bpm] @ UR, linestyle='-')  # solid line for y_sim_fgm
if compare:
    axs[1, 2].plot(y_sim_nn[:, id_to_bpm] @ UR, linestyle='--')  # dashed line for y_sim_nn
axs[1, 2].set_title('Output Mode Space')


# Subplot 8: % Error in Output
if compare:
    axs[1, 3].plot(y_err[:, id_to_bpm], linestyle='-')  # solid line for y_sim_fgm
axs[1, 3].set_title('Output Error')

#Suplot 9 : Loss
if compare:
    axs[1, 4].plot(loss_data['epochs'], loss_data['epoch_losses'])
    axs[1, 4].set_xlabel('Epoch')
    axs[1, 4].set_ylabel('Loss')
    axs[1, 4].set_title('Training Loss')

plt.suptitle('imode: ' + str(imode) + ' | Direction: ' + pick_direction + ' | n states: ' + str(n_include) )

print("Time taken: ", time.time() - start)

# # Adjust layout for better spacing
plt.tight_layout()
plt.show()











