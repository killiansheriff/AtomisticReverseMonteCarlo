import numpy as np
from ovito.io import export_file, import_file

from AtomisticReverseMonteCarlo import AtomisticReverseMonteCarlo

mod = AtomisticReverseMonteCarlo(
    nneigh=12,  # number of neighbors to compute WC parameters
    T=1e-9,  # rMC temperature
    target_wc=[  # wc target 1-pij/cj
        [0.32719603, -0.19925471, -0.12794131],
        [-0.19925471, 0.06350427, 0.13575045],
        [-0.12794131, 0.13575045, -0.00762235],
    ],
    tol_percent_diff=np.ones((3, 3)).tolist(),  # max percent tolerence allowed before stopping
)

pipeline = import_file("fcc_random.dump")
pipeline.modifiers.append(mod)
data = pipeline.compute()

export_file(
    data,
    "fcc_wc.dump",
    "lammps/dump",
    columns=[
        "Particle Identifier",
        "Particle Type",
        "Position.X",
        "Position.Y",
        "Position.Z",
    ],
)
