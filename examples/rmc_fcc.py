import numpy as np

from rMC import rMC

if __name__ == "__main__":
    rmc = rMC(
        target_wc=np.load(
            "/home/ksheriff/PAPERS/first_paper/03_mtp/data/eca_id_temperature/300K/wc_3x3.npy"
        ),
    )
    # rmc.set_data_from_dump("fcc_random.dump")
    # rmc.set_data_from_dump(
    #     "/home/ksheriff/PAPERS/first_paper/03_mtp/data/dumps/dumps_rss_mtp_mc/RSS_relaxation_20_1_600K.dump"
    # )

    data = rmc.data_from_ase(
        crystal_structure="fcc",
        dimension=(10, 10, 10),
        concentrations=(1 / 3, 1 / 3, 1 / 3),
        elements=("Ni", "Co", "Cr"),
    )
    rmc.run(nneigh=12, T=1e-9, tol_percent_diff=np.ones((3, 3)) * 1, save_file_name="fcc_wc.dump")
