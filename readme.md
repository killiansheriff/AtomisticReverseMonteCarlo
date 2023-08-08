# Atomistic Reverse Monte-Carlo 
``rMC`` is a python package which allows generating bulk crystal structures with target Warren-Cowley parameters. 

## Usage 
Here's an example of how to use rMC. Additional examples can be found in the ``examples`` folder.  

```python 
import numpy as np
from rMC import rMC

rmc = rMC()

# Set rMC target 1-pij/cj
target_wc = np.array([[0.349, -0.239, -0.112], [-0.239, 0.059, 0.18], [-0.112, 0.18, -0.068]])
rmc.set_target_wc(target_wc)

# Set data from dump
rmc.set_data_from_dump("fcc_random.dump")

# Alternatively can also create intial config from ase
data = rmc.set_data_from_ase(
    crystal_structure="fcc",
    dimension=(10, 10, 10),
    concentrations=(1 / 3, 1 / 3, 1 / 3),
    elements=np.array(["Ni", "Co", "Cr"]),
    lat_params={"a": 3.57},
)

# Run rmc
rmc.run(
    nneigh=12,
    T=1e-9,
    tol_percent_diff=np.ones((3, 3)) * 1,
    save_file_name="fcc_wc.dump",
)


## Installation
For a standalone Python package or Conda environment, please use:
```bash
pip install --user rMC
```

For *OVITO PRO* built-in Python interpreter, please use:
```bash
ovitos -m pip install --user rMC
```

If you want to install the lastest git commit, please replace ``rMC`` by ``git+https://github.com/killiansheriff/Atomistic-Reverse-Monte-Carlo``.

![](media/ovito_pro_desktop.png)

## Contact
If any questions, feel free to contact me (ksheriff at mit dot edu).



```