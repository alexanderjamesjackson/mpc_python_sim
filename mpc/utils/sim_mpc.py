import osqp
import numpy as np
import scipy.sparse as sparse
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as rand

class Mpc:
    def __init__(
        self, n_samples, n_delay, dist,
        Ap, Bp, Cp,
        Ao, Bo, Co, Ad, Cd, LxN_obs, Lxd_obs,
        J_MPC, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        A_awr, B_awr, C_awr, D_awr,
        SOFB_setp,beta_FGM,
        ol_mode = False, dtype = np.float64, hil_mode = True):

        #Initialise solver parameters
        self.MAX_ITER  = 20
        self.beta_fgm = beta_FGM
        self.n_samples = n_samples
        self.n_delay = n_delay
        self.dist = dist
        self.dtype = dtype
        self.Ap = Ap.astype(dtype)
        self.Bp = Bp.astype(dtype)
        self.Cp = Cp.astype(dtype)
        self.Ao = Ao.astype(dtype)
        self.Bo = Bo.astype(dtype)
        self.Co = Co.astype(dtype)
        self.Ad = Ad.astype(dtype)
        self.Cd = Cd.astype(dtype)
        self.LxN_obs = LxN_obs.astype(dtype)
        self.Lxd_obs = Lxd_obs.astype(dtype)
        self.J_FGM = J_MPC.astype(dtype)
        self.J = sparse.csc_matrix(J_MPC, dtype=dtype)
        self.u_max = u_max.astype(dtype)
        self.u_rate = u_rate.astype(dtype)
        self.id_to_bpm = id_to_bpm
        self.id_to_cm = id_to_cm
        self.A_awr = A_awr.astype(dtype)
        self.B_awr = B_awr.astype(dtype)
        self.C_awr = C_awr.astype(dtype)
        self.D_awr = D_awr.astype(dtype)
        self.SOFB_setp = SOFB_setp
        self.ol_mode = ol_mode
        self.hil_mode = hil_mode


        #extract dimensions
        self.nx_plant, self.nu_plant = np.size(Bp,0) , np.size(Bp,1)
        self.ny_plant = np.size(Cp,0)
        self.nx_obs, self.nu_obs = np.size(Bo,0) , np.size(Bo,1)
        self.ny_obs = np.size(Co,0)

        #Variables for plant
        np.random.seed(42)
        self.x_sim = np.zeros((self.nx_plant, 1))
        self.y_sim = np.zeros((self.ny_plant, n_samples))
        self.u_sim = np.zeros((self.nu_plant, n_samples))
        self.u_sim_expert = np.zeros((self.nu_plant, n_samples))

        #Variables for AWR
        self.ny_awr , self.nx_awr = np.size(C_awr,0) , np.size(C_awr,1)
        self.x_awr = np.zeros((self.nx_awr, 1))
        self.y_awr = np.zeros((self.ny_awr, 1))

        #Variables for observer
        self.x_obs_old = np.zeros((n_delay, self.nx_obs, 1), dtype=dtype)
        self.x_obs_new = np.zeros((n_delay + 1, self.nx_obs, 1), dtype=dtype)
        self.xd_obs_old = np.zeros((self.nx_obs, 1), dtype=dtype)
        self.xd_obs_new = np.zeros((self.nx_obs, 1), dtype=dtype)
        self.x0_obs = np.zeros((n_samples, self.nx_obs, 1))
        self.xd_obs = np.zeros((n_samples, self.nx_obs, 1))
        self.Apow = np.zeros((n_delay + 1, self.nx_obs, self.nx_obs), dtype=dtype)
        
        #Compute elementwise powers in advance
        for i in range(n_delay + 1):
            self.Apow[i] = (self.Ao ** i).astype(dtype)

        
        self.q_mat = q_mat.astype(dtype)
        self.y_max = y_max.astype(dtype)
        self.u_max = u_max.astype(dtype)
        self.y_mat = (self.Co @ self.Ao).astype(dtype)
        self.y_meas = np.zeros((self.ny_obs, 1), dtype=dtype)

        #Setup constraints

        self.A_constr = sparse.csr_matrix(np.vstack((np.eye(self.ny_obs), Co @ Bo)), dtype=dtype)

        self.l_constr = np.vstack(( 
            np.maximum(-u_max - SOFB_setp, -u_rate + self.y_awr),
            -y_max - self.y_mat @ self.x_obs_new[0]
        ))

        self.u_constr = np.vstack((
            np.minimum(u_max - SOFB_setp, u_rate + self.y_awr),
            y_max - self.y_mat @ self.x_obs_new[0]
        ))

        #Variables for FGM
        self.z_new = np.zeros((self.nu_obs, 1), dtype=dtype)   


    #Initialises OSQP solver
    def setupOSQP(self, verbose,
        polish,
        adaptive_rho):
        
        settings = {
            'verbose': verbose,
            'polish': polish,
            'adaptive_rho': adaptive_rho,
            'max_iter': self.MAX_ITER,
            'check_termination': self.MAX_ITER
        }

        self.osqp_solver = osqp.OSQP()
        self.osqp_solver.setup(P=self.J, q=np.zeros(np.size(self.q_mat, 0)),
                       A=self.A_constr, l=self.l_constr, u=self.u_constr, **settings)


    def update_y_sim(self, k):
        if self.ol_mode:
            self.y_sim[:, k:k+1] = self.dist[:, k:k+1]
        else:
            self.y_sim[:, k:k+1] = (self.Cp @ self.x_sim) + self.dist[:, k:k+1]

    def update_y_meas(self, k):
        if self.hil_mode ==  True:
            self.y_meas = (np.round(self.y_sim[self.id_to_bpm, k - self.n_delay][:, np.newaxis] * 1000, 0) * 0.001).astype(self.dtype)
        else:
            self.y_meas = self.y_sim[self.id_to_bpm, k-self.n_delay][:, np.newaxis].astype(self.dtype)

    def calc_obs_state(self, k):
        self.x_obs_new[0] = (self.Ao @ self.x_obs_old[0]) + (self.Bo @ self.u_sim[self.id_to_cm, k-1][:, np.newaxis].astype(self.dtype))
        self.xd_obs_new = self.Ad @ self.xd_obs_old
        for i in range(1, self.n_delay + 1):
            self.x_obs_new[i] = self.x_obs_old[i-1]

    def update_obs_measurement(self):
        delta_y = self.y_meas - self.Co @ self.x_obs_new[self.n_delay] - self.Cd @ self.xd_obs_new
        delta_xN = self.LxN_obs @ delta_y
        delta_xd = self.Lxd_obs @ delta_y
        self.xd_obs_new = self.xd_obs_new + delta_xd

        for i in range(self.n_delay + 1):
            self.x_obs_new[i] = self.x_obs_new[i] + self.Apow[self.n_delay - i] @ delta_xN

    def update_obs_state(self,k):
        self.xd_obs_old = self.xd_obs_new
        self.x_obs_old = self.x_obs_new[:self.n_delay][:][:]
        self.x0_obs[k] = self.x_obs_new[0]
        self.xd_obs[k] = self.xd_obs_new

    def update_q_limits(self):
        ####
        self.q = self.q_mat @ np.vstack((self.x_obs_new[0], self.xd_obs_new))
        if self.use_FGM:
            self.l_constr = np.maximum(-self.u_max - self.SOFB_setp, -self.u_rate + self.y_awr)
            self.u_constr = np.minimum(self.u_max - self.SOFB_setp, self.u_rate + self.y_awr)
        else:
        
            if self.dtype == np.float32:

                self.l_constr = np.vstack((
                    np.maximum(-self.u_max - self.SOFB_setp.astype(self.dtype), -self.u_rate + self.y_awr.astype(self.dtype)),
                    -self.y_max - self.y_mat @ self.x_obs_new[0].astype(self.dtype)
                ))
                self.u_constr = np.vstack((
                    np.minimum(self.u_max - self.SOFB_setp.astype(self.dtype), self.u_rate + self.y_awr.astype(self.dtype)),
                    self.y_max - self.y_mat @ self.x_obs_new[0].astype(self.dtype)
                ))
            else:
                self.l_constr = np.vstack((
                    np.maximum(-self.u_max - self.SOFB_setp.astype(self.dtype), -self.u_rate + self.y_awr.astype(self.dtype)),
                    -self.y_max - self.y_mat @ self.x_obs_new[0].astype(self.dtype) - self.xd_obs_new.astype(self.dtype)
                ))
                self.u_constr = np.vstack((
                    np.minimum(self.u_max - self.SOFB_setp.astype(self.dtype), self.u_rate + self.y_awr.astype(self.dtype)),
                    self.y_max - self.y_mat @ self.x_obs_new[0].astype(self.dtype) - self.xd_obs_new.astype(self.dtype)
                ))


    #Solves the problem using OSQP and updates the AWR state
    def solveOSQP_update_awr(self,k):
        self.osqp_solver.update(q=self.q, l=self.l_constr, u=self.u_constr)
        result = self.osqp_solver.solve()
        osqp_result = result.x[:self.nu_obs][:, np.newaxis]
        assert result.info.status_val in [1, 2], "OSQP solver did not find an optimal solution."
        if self.hil_mode == True:
            osqp_result = np.round(osqp_result * 1000000,0) * 0.000001
        self.u_sim[self.id_to_cm, k] = osqp_result.flatten()

        self.x_awr = self.A_awr @ self.x_awr + self.B_awr @ osqp_result.astype(np.float64)
        self.y_awr = self.C_awr @ self.x_awr + self.D_awr @ osqp_result.astype(np.float64)


    def update_plant_state(self,k):
        self.x_sim = self.Ap @ self.x_sim + self.Bp @ self.u_sim[:, k:k+1]


    #Solves the problem using FGM and updates the AWR state
    def solveFGM_update_awr(self, k):

        out_global = self.u_sim[self.id_to_cm, k-1][:, np.newaxis]
        for i in range(self.MAX_ITER):
            self.z_old = self.z_new
            t = self.J_FGM @ out_global - self.q
            self.z_new = np.maximum(self.l_constr, np.minimum(self.u_constr, t))
            out_global = (1+self.beta_fgm) * self.z_new - self.beta_fgm * self.z_old
        fgm_result = self.z_new

        #sign = rand.choice([-1, 1])
        self.u_sim[self.id_to_cm, k] = fgm_result.flatten() #+ np.ones(self.id_to_cm.size) * 0.08 * sign

        self.x_awr = self.A_awr @ self.x_awr + self.B_awr @ fgm_result.astype(np.float64)
        self.y_awr = self.C_awr @ self.x_awr + self.D_awr @ fgm_result.astype(np.float64)

    #Function to simulate the MPC

    def sim_mpc(self, use_fgm):
        self.use_FGM = use_fgm
        self.setupOSQP(False, False, False)

        for k in range(self.n_samples):
            #Measurements
            if k % 100 == 0:
                print(f"Simulation progress: {k/self.n_samples*100:.2f}%")

            self.update_y_sim(k)
            if k >= self.n_delay:
                self.update_y_meas(k)
                self.calc_obs_state(k)
                self.update_obs_measurement()
                self.update_obs_state(k)
                self.update_q_limits()
                if use_fgm:
                    self.solveFGM_update_awr(k)
                else:
                    self.solveOSQP_update_awr(k)
            self.update_plant_state(k)

        self.u_sim = np.transpose(self.u_sim)
        self.y_sim = np.transpose(self.y_sim)
        self.x0_obs = self.x0_obs.reshape((self.n_samples, self.nx_obs))
        self.xd_obs = self.xd_obs.reshape((self.n_samples, self.nx_obs))
        return self.y_sim, self.u_sim, self.x0_obs, self.xd_obs
    

    #Solves the MPC problem using nn
    def solvenn(self,k):
        x_aug = torch.tensor(np.transpose(np.vstack((self.x_obs_new[0], self.xd_obs_new)))).float().to(self.device)
        self.u_nn = self.nn_controller(x_aug).detach().numpy()
        self.u_nn = np.transpose(self.u_nn)
        #######
        self.u_sim[self.id_to_cm,k] = self.u_nn.flatten()
    
    #Solves the MPC problem using rnn
    def solvernn(self,k):
        concat_states = np.hstack((self.x0_obs, self.xd_obs))
        x_seq = concat_states[k-self.sequence_length+1:k+1]
        x_seq = x_seq.transpose(2,0,1)
        x_aug = torch.tensor(x_seq).float().to(self.device)
        self.u_nn = self.nn_controller(x_aug).detach().numpy()
        self.u_nn = np.transpose(self.u_nn)
        self.u_sim[self.id_to_cm,k] = self.u_nn.flatten()

    #Updates the AWR state using nn
    def nn_update_awr(self):
        self.x_awr = self.A_awr @ self.x_awr + self.B_awr @ self.u_nn
        self.y_awr = self.C_awr @ self.x_awr + self.D_awr @ self.u_nn

    def solveFGM_expert(self, k):

        out_global = self.u_sim[self.id_to_cm, k-1][:, np.newaxis]
        for i in range(self.MAX_ITER):
            self.z_old = self.z_new
            t = self.J_FGM @ out_global - self.q
            self.z_new = np.maximum(self.l_constr, np.minimum(self.u_constr, t))
            out_global = (1+self.beta_fgm) * self.z_new - self.beta_fgm * self.z_old
        fgm_result = self.z_new

        self.u_sim_expert[self.id_to_cm, k] = fgm_result.flatten()


    #Simulate using nn
    def sim_nn(self,model,device, sequence_length = 0, RNN = False):
        self.sequence_length = sequence_length
        self.device = device
        self.nn_controller = model
        self.nn_controller.eval()
        self.nn_controller.to(torch.device(device))

        for k in range(self.n_samples):
            #Measurements
            if k % 100 == 0:
                print(f"Simulation progress: {k/self.n_samples*100:.2f}%")

            self.update_y_sim(k)
            if k >= self.n_delay:
                self.update_y_meas(k)
                self.calc_obs_state(k)
                self.update_obs_measurement()
                self.update_obs_state(k)
                if RNN:
                    self.solvernn(k)
                else:
                    self.solvenn(k)
                self.nn_update_awr()
            self.update_plant_state(k)

        self.u_sim = np.transpose(self.u_sim)
        self.y_sim = np.transpose(self.y_sim)
        self.x0_obs = self.x0_obs.reshape((self.n_samples, self.nx_obs))
        self.xd_obs = self.xd_obs.reshape((self.n_samples, self.nx_obs))
        return self.y_sim, self.u_sim, self.x0_obs, self.xd_obs


    def sim_dagger(self, model, device):
        self.use_FGM = True
        self.device = device
        self.nn_controller = model
        self.nn_controller.eval()
        self.nn_controller.to(torch.device(device))

        for k in range(self.n_samples):
            #Measurements
            if k % 100 == 0:
                print(f"Simulation progress: {k/self.n_samples*100:.2f}%")

            self.update_y_sim(k)
            if k >= self.n_delay:
                self.update_y_meas(k)
                self.calc_obs_state(k)
                self.update_obs_measurement()
                self.update_obs_state(k)
                self.update_q_limits()
                self.solveFGM_expert(k)
                self.solvenn(k)
                self.nn_update_awr()
            self.update_plant_state(k)

        self.u_sim_expert = np.transpose(self.u_sim_expert)
        self.x0_obs = self.x0_obs.reshape((self.n_samples, self.nx_obs))
        self.xd_obs = self.xd_obs.reshape((self.n_samples, self.nx_obs))
        return self.u_sim_expert, self.x0_obs, self.xd_obs







