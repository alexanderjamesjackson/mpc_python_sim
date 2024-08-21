mpc:

    model.py - allows for design of neural network and loss function

    processdata.py - helper function for constructing datasets

    diamond_I_configuration_v5.py - helper function for initialising DLS actuators and sensors

    sim_mpc.py - implements control scheme based on neural network or other solver


    simulation_mpc.py - Executable, runs full simulation solving with OSQP or FGM as specified

    test_model.py - Executable:
                    Runs partial simulation with first n states of system 
                    Saves training data with varied disturbances
                    Runs comparison between nn and mpc

    train.py - Executable, trains network defined in model.py with data saved from test_model.py


