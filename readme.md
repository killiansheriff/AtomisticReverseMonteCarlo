# Atomistic Reverse Monte-Carlo 
OVITO Python modifier to generate bulk crystal structures with target Warren-Cowley parameters. 

## Usage 
Here's an example on how to use the code, the code can be found in ``examples/``:

```python 
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
    save_rate=100000,
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
```

## Installation
For a standalone Python package or Conda environment, please use:
```bash
pip install --user AtomisticReverseMonteCarlo
```

For *OVITO PRO* built-in Python interpreter, please use:
```bash
ovitos -m pip install --user AtomisticReverseMonteCarlo
```

If you want to install the lastest git commit, please replace ``AtomisticReverseMonteCarlo`` by ``git+https://github.com/killiansheriff/Atomistic-Reverse-Monte-Carlo``.

![](media/ovito_pro_desktop.png)

## Contact
If any questions, feel free to contact me (ksheriff at mit dot edu).