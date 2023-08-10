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
    tol_percent_diff=[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],  # max percent tolerence allowed before stopping
    save_rate=1000,
    seed=123,
    max_iter=None,  # infinity number of iter
)

pipeline = import_file("fcc_random.dump")
pipeline.modifiers.append(mod)
data = pipeline.compute()

print(f'Target Warren-Cowley parameters: \n {data.attributes["Target Warren-Cowley parameters"]}')
print(f'Warren-Cowley parameters: \n {data.attributes["Warren-Cowley parameters"]}')
print(f'Warren-Cowley Percent error: \n {data.attributes["Warren-Cowley percent error"]}')

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
