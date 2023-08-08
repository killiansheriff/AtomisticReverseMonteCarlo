import numpy as np

from rMC_old import rMC

if __name__ == "__main__":
    rmc = rMC()

    # Set wc target 1-pij/cj
    target_wc = np.array(
        [
            [0.32719603, -0.19925471, -0.12794131],
            [-0.19925471, 0.06350427, 0.13575045],
            [-0.12794131, 0.13575045, -0.00762235],
        ]
    )
    rmc.set_target_wc(target_wc)

    # Set data from dump
    rmc.set_data_from_dump("fcc_random.dump")

    # Alternatively you create a random solid solution from ASE
    # rmc.set_data_from_ase(
    #     crystal_structure="fcc",
    #     dimension=(10, 10, 10),
    #     concentrations=(1 / 3, 1 / 3, 1 / 3),
    #     elements=np.array(["Ni", "Co", "Cr"]),
    #     lat_params={"a": 3.57},
    # )

    # Run rmc
    rmc.run(
        nneigh=12,  # number of neighbors to compute WC parameters
        T=1e-9,  # rMC temperature
        tol_percent_diff=np.ones((3, 3)) * 1,  # max percent tolerence allowed before stopping
        save_file_name="fcc_wc.dump",
    )
