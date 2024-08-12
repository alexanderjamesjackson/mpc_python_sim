import osqp
import numpy as np
import scipy.sparse as sparse

def sim_mpc_OSQP(
        n_samples, n_delay, dist,
        Ap, Bp, Cp,
        Ao, Bo, Co, Ad, Cd, LxN_obs, Lxd_obs,
        J_MPC, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        A_awr, B_awr, C_awr, D_awr,
        SOFB_setp,
        ol_mode = False):
    
    assert(n_delay == 8) or (n_delay == 9)
    use_single = False
    hil_mode = True

    #Extract plant and observer dimensions
    nx_plant, nu_plant = np.size(Bp,0) , np.size(Bp,1)
    ny_plant = np.size(Cp,0)
    nx_obs, nu_obs = np.size(Bo,0) , np.size(Bo,1)
    ny_obs = np.size(Co,0)

    #Variables for plant
    x_sim_new, x_sim_old = np.zeros((nx_plant, 1)) , np.zeros((nx_plant, 1))
    y_sim = np.zeros((ny_plant, n_samples))
    u_sim = np.zeros((nu_plant, n_samples))

    #Variables for AWR
    ny_awr , nx_awr = np.size(C_awr,0) , np.size(C_awr,1)
    x_awr_new = np.zeros((nx_awr, 1))
    y_awr = np.zeros((ny_awr, 1))

    #Variables for observer
    if use_single == True:
        dtype = np.float32
    else:
        dtype = np.float64
    
    x0_obs_new = np.zeros((nx_obs, 1), dtype=dtype)
    x0_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    x1_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    x2_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    x3_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    x4_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    x5_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    x6_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    x7_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    x8_obs_old = np.zeros((nx_obs, 1), dtype=dtype)
    xd_obs_old = np.zeros((ny_obs, 1), dtype=dtype)
    Apow1 = (Ao ** 1).astype(dtype)
    Apow2 = (Ao ** 2).astype(dtype)
    Apow3 = (Ao ** 3).astype(dtype)
    Apow4 = (Ao ** 4).astype(dtype)
    Apow5 = (Ao ** 5).astype(dtype)
    Apow6 = (Ao ** 6).astype(dtype)
    Apow7 = (Ao ** 7).astype(dtype)
    Apow8 = (Ao ** 8).astype(dtype)
    Apow9 = (Ao ** 9).astype(dtype)

    Ao = Ao.astype(dtype)
    Bo = Bo.astype(dtype)
    Co = Co.astype(dtype)
    Ad = Ad.astype(dtype)
    Cd = Cd.astype(dtype)
    LxN_obs = LxN_obs.astype(dtype)
    Lxd_obs = Lxd_obs.astype(dtype)

    #Variables Solver
    J = sparse.csc_matrix(J_MPC, dtype=dtype)
    q_mat = q_mat.astype(dtype)
    y_max = y_max.astype(dtype)
    u_max = u_max.astype(dtype)
    y_mat = (Co@Ao).astype(dtype)


    MAX_ITER = 20
    settings = {
        'verbose': False,
        'polish': False,
        'adaptive_rho': False,
        'max_iter': MAX_ITER,
        'check_termination': MAX_ITER
    }
    
    A_constr = sparse.csr_matrix(np.vstack((np.eye(ny_obs), Co @ Bo)), dtype=dtype)

    l_constr = np.vstack((
        np.maximum(-u_max - SOFB_setp, -u_rate + y_awr),
        -y_max - y_mat @ x0_obs_new
    ))
    u_constr = np.vstack((
        np.minimum(u_max - SOFB_setp, u_rate + y_awr),
        y_max - y_mat @ x0_obs_new
    ))

    osqp_solver = osqp.OSQP()
    osqp_solver.setup(P=J, q=np.zeros(np.size(q_mat, 0)),
                       A=A_constr, l=l_constr, u=u_constr, **settings)
    
    x0_obs = np.zeros((nx_obs, n_samples))
    xd_obs = np.zeros((ny_obs, n_samples))
    for k in range(n_samples):
        #Measurements
        if k % 100 == 0:
            print(f"Simulation progress: {k/n_samples*100:.2f}%")
            
        if ol_mode == True:
            y_sim[:, k:k+1] = dist[:, k:k+1] #
        else:
            y_sim[:, k:k+1] = (Cp @ x_sim_new) + dist[:, k:k+1]
       
        if k > n_delay:
            
            if hil_mode ==  True:
                y_meas = (np.round(y_sim[id_to_bpm, k - n_delay][:, np.newaxis] * 1000, 0) * 0.001).astype(dtype)
            else:
                y_meas = y_sim[id_to_bpm, k-n_delay].astype(dtype)

            #Observer - State update
            x0_obs_new = (Ao @ x0_obs_old) + (Bo @ u_sim[id_to_cm, k-1][:, np.newaxis].astype(dtype))
            xd_obs_new = Ad @ xd_obs_old
            x1_obs_new = x0_obs_old
            x2_obs_new = x1_obs_old
            x3_obs_new = x2_obs_old
            x4_obs_new = x3_obs_old
            x5_obs_new = x4_obs_old
            x6_obs_new = x5_obs_old
            x7_obs_new = x6_obs_old
            x8_obs_new = x7_obs_old
            if(n_delay == 9):
                x9_obs_new = x8_obs_old
            #Observer - Measurement update
            if(n_delay == 9):
                delta_y = y_meas - Co @ x9_obs_new - Cd @ xd_obs_new
            else:
                delta_y = y_meas - Co @ x8_obs_new - Cd @ xd_obs_new
            delta_xN = LxN_obs @ delta_y
            delta_xd = Lxd_obs @ delta_y
            xd_obs_new = xd_obs_new + delta_xd
            if (n_delay == 9):
                x9_obs_new = x9_obs_new + delta_xN
                x8_obs_new = x8_obs_new + Apow1 @ delta_xN
                x7_obs_new = x7_obs_new + Apow2 @ delta_xN
                x6_obs_new = x6_obs_new + Apow3 @ delta_xN
                x5_obs_new = x5_obs_new + Apow4 @ delta_xN
                x4_obs_new = x4_obs_new + Apow5 @ delta_xN
                x3_obs_new = x3_obs_new + Apow6 @ delta_xN
                x2_obs_new = x2_obs_new + Apow7 @ delta_xN
                x1_obs_new = x1_obs_new + Apow8 @ delta_xN
                x0_obs_new = x0_obs_new + Apow9 @ delta_xN
            else:
                x8_obs_new = x8_obs_new + delta_xN
                x7_obs_new = x7_obs_new + Apow1 @ delta_xN
                x6_obs_new = x6_obs_new + Apow2 @ delta_xN
                x5_obs_new = x5_obs_new + Apow3 @ delta_xN
                x4_obs_new = x4_obs_new + Apow4 @ delta_xN
                x3_obs_new = x3_obs_new + Apow5 @ delta_xN
                x2_obs_new = x2_obs_new + Apow6 @ delta_xN
                x1_obs_new = x1_obs_new + Apow7 @ delta_xN
                x0_obs_new = x0_obs_new + Apow8 @ delta_xN
            #Update observer states
            xd_obs_old = xd_obs_new
            x0_obs_old = x0_obs_new
            x1_obs_old = x1_obs_new
            x2_obs_old = x2_obs_new
            x3_obs_old = x3_obs_new
            x4_obs_old = x4_obs_new
            x5_obs_old = x5_obs_new
            x6_obs_old = x6_obs_new
            x7_obs_old = x7_obs_new
            if (n_delay == 9):
                x8_obs_old = x8_obs_new
            x0_obs[:,k:k+1] = x0_obs_new
            xd_obs[:,k:k+1] = xd_obs_new

            #Compute q-vector
            q = q_mat @ np.vstack((x0_obs_new, xd_obs_new))

            #Compute lower and upper limits
            if dtype == np.float32:
                l_constr = np.vstack((
                    np.maximum(-u_max - SOFB_setp.astype(dtype), -u_rate + y_awr.astype(dtype)),
                    -y_max - y_mat @ x0_obs_new.astype(dtype)
                ))
                u_constr = np.vstack((
                    np.minimum(u_max - SOFB_setp.astype(dtype), u_rate + y_awr.astype(dtype)),
                    y_max - y_mat @ x0_obs_new.astype(dtype)
                ))
            else:
                # print(y_awr.shape)
                # print(np.maximum(-u_max - SOFB_setp.astype(dtype), -u_rate + y_awr.astype(dtype)).shape)
                # print((-y_max - y_mat @ x0_obs_new.astype(dtype) - xd_obs_new.astype(dtype)).shape)
                l_constr = np.vstack((
                    np.maximum(-u_max - SOFB_setp.astype(dtype), -u_rate + y_awr.astype(dtype)),
                    -y_max - y_mat @ x0_obs_new.astype(dtype) - xd_obs_new.astype(dtype)
                ))
                u_constr = np.vstack((
                    np.minimum(u_max - SOFB_setp.astype(dtype), u_rate + y_awr.astype(dtype)),
                    y_max - y_mat @ x0_obs_new.astype(dtype) - xd_obs_new.astype(dtype)
                ))

            
            osqp_solver.update(l=l_constr, u=u_constr, q=q)
            result = osqp_solver.solve()
            osqp_result = result.x[:nu_obs][:, np.newaxis]
            assert result.info.status_val in [1, 2], "OSQP solver did not find an optimal solution."

            if hil_mode == True:
                osqp_result = np.round(osqp_result * 1000000,0) * 0.000001
            u_sim[id_to_cm, k][:, np.newaxis] = osqp_result.astype(np.float64)
            #AWR
            x_awr_old = x_awr_new
            x_awr_new = A_awr @ x_awr_old + B_awr @ osqp_result.astype(np.float64)
            y_awr = C_awr @ x_awr_new + D_awr @ osqp_result.astype(np.float64)
        #Plant
        x_sim_old = x_sim_new
        x_sim_new = Ap @ x_sim_old + Bp @ u_sim[:, k:k+1]
    u_sim = np.transpose(u_sim)
    y_sim = np.transpose(y_sim)
    return y_sim, u_sim
