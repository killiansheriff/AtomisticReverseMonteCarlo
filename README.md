# Atomistic Reverse Monte-Carlo 
OVITO Python modifier to generate bulk crystal structures with target Warren-Cowley parameters. 

## Usage 
Here's an example on how to use the code to create the ``fcc_wc.dump`` file which has Warren-Cowley parameters that falls within a 1% difference of the targeted ones:

```python 
from ovito.io import export_file, import_file

from AtomisticReverseMonteCarlo import AtomisticReverseMonteCarlo

mod = AtomisticReverseMonteCarlo(
    nneigh=12,                                                          # number of neighbors to compute WC parameters (12 1NN in fcc)
    T=1e-9,                                                             # rMC temperature
    target_wc=[                                                         # wc target 1-pij/cj
        [0.32719603, -0.19925471, -0.12794131],
        [-0.19925471, 0.06350427, 0.13575045],
        [-0.12794131, 0.13575045, -0.00762235],
    ],
    tol_percent_diff=[                                                  # max percent tolerence allowed before stopping
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],                          
    save_rate=100000,                                                   # Save rate
)

# Load the intial snapshot 
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
```
The script can be found in the ``examples`` directory.

## Installation
For a standalone Python package or Conda environment, please use:
```bash
pip install --user AtomisticReverseMonteCarlo
```

For *OVITO PRO* built-in Python interpreter, please use:
```bash
ovitos -m pip install --user AtomisticReverseMonteCarlo
```

If you want to install the lastest git commit, please replace ``AtomisticReverseMonteCarlo`` with ``git+https://github.com/killiansheriff/AtomisticReverseMonteCarlo``.

![](media/ovito_pro_desktop.png)

## Contact
If any questions, feel free to contact me (ksheriff at mit dot edu).

## References & citing 
If you use this repository in your work, please cite ``ibtex entry to follow``.