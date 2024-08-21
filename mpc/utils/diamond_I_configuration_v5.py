import numpy as np
#Corrected for 0-based indexing
def diamond_I_configuration_v5(RMorigx, RMorigy, square_config=False):
    TOT_BPM = 173
    TOT_CM = 172

    # This is the configuration implemented on the 13.09.2022 at DLS
    if not square_config:
        bad_bpm_x = np.array([32, 75, 78, 120, 137, 155])
        bad_bpm_y = np.array([18, 25, 62, 75, 78, 83, 137, 141])
        bad_cm_x = np.array([76])
        bad_cm_y = np.array([76])
    else:
        bad_bpm_x = np.array([32, 75, 78, 120, 137, 155])
        bad_bpm_y = np.array([18, 25, 62, 75, 78, 83, 137, 141])
        bad_cm_x = np.array([76, 32, 120, 137, 155])
        bad_cm_y = np.array([76, 18, 25, 62, 83, 137, 141])

    bad_bpm_x = np.sort(bad_bpm_x)
    bad_bpm_y = np.sort(bad_bpm_y)
    bad_cm_x = np.sort(bad_cm_x)
    bad_cm_y = np.sort(bad_cm_y)

    # Generate IDs as used by the FOFB
    id_to_bpm_x = np.arange(TOT_BPM)
    id_to_bpm_x = np.delete(id_to_bpm_x, bad_bpm_x)
    id_to_bpm_y = np.arange(TOT_BPM)
    id_to_bpm_y = np.delete(id_to_bpm_y, bad_bpm_y)
    id_to_cm_x = np.arange(TOT_CM)
    id_to_cm_x = np.delete(id_to_cm_x, bad_cm_x)
    id_to_cm_y = np.arange(TOT_CM)
    id_to_cm_y = np.delete(id_to_cm_y, bad_cm_y)

    # Ensure full rank
    assert np.linalg.matrix_rank(RMorigx[np.ix_(id_to_bpm_x, id_to_cm_x)]) == len(id_to_bpm_x)
    assert np.linalg.matrix_rank(RMorigy[np.ix_(id_to_bpm_y, id_to_cm_y)]) == len(id_to_bpm_y)

    for i in bad_bpm_x:
        assert i not in id_to_bpm_x
    for i in bad_cm_x:
        assert i not in id_to_cm_x
    for i in bad_bpm_y:
        assert i not in id_to_bpm_y
    for i in bad_cm_y:
        assert i not in id_to_cm_y

    return id_to_bpm_x, id_to_cm_x, id_to_bpm_y, id_to_cm_y

