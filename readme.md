# Atomistic Reverse Monte-Carlo 
``rMC`` is a python package which allows generating bulk crystal structures with target Warren-Cowley parameters. 

## Installation 
You can install rMC through PyPy using:
```bash
pip install rMC
```
## Utilisation 
Here's an example of how to use rMC. Additional examples can be found in the ``examples`` folder.  

```python 
import numpy as np
from rMC import rMC

rmc = rMC()

# Set rMC target
target_wc = np.array([[0.349, -0.239, -0.112], [-0.238, 0.059, 0.18], [-0.111, 0.18, -0.068]])
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

```