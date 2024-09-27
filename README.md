mpc:
    utils:
        model.py - allows for design of neural network and loss function

        processdata.py - helper function for constructing datasets

        diamond_I_configuration_v5.py - helper function for initialising DLS actuators and sensors

        sim_mpc.py - implements simulation with various controllers


    simulation_mpc.py - Executable, runs full simulation with MPC solving with OSQP or FGM as specified


    test_model.py - Executable:
                    Runs partial simulation with first n states of system 
                    Runs comparison between nn and mpc
                    Uses DAGGER for generating datasets
                    Generates datasets from various disturbances 

    train.py - Executable, trains network with data saved from test_model.py

    evaluateModels.py - Executable, evaluates model performance over various trajectories

    lqr.py - Executable, generate LQR training data based on random sampling

    datatoexpert.py - Executable, converts u_train.pt etc to expert data file

    




