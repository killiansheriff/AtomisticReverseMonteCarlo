from functools import partial
from itertools import product

import numpy as np
from ase.build import bulk
from ovito.data import NearestNeighborFinder
from ovito.io import export_file, import_file
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource


class rMC:
    def __init__(self):
        print("Let's do some rMC!")

    def set_target_wc(self, target_wc):
        self.target_wc = target_wc

    def set_data_from_dump(self, dump_file):
        self.pipeline = import_file(dump_file)
        self.data = self.pipeline.compute()

    def set_data_from_ase(
        self,
        crystal_structure,
        dimension,
        concentrations,
        elements,
        lat_params,
    ):
        structure = bulk(elements[0], crystal_structure, **lat_params)
        structure = structure.repeat(dimension)

        # Get the number of atoms in the lattice
        num_atoms = len(structure)

        # Random putting of elements
        atom_types = np.zeros(num_atoms, dtype=int)
        N_atoms = [0]
        for i, concentration in enumerate(concentrations):
            N = int(num_atoms * concentration)

            atom_types[N_atoms[-1] : N_atoms[-1] + N] = i
            N_atoms.append(N_atoms[-1] + N)

        # shuffle
        np.random.shuffle(atom_types)

        structure.set_chemical_symbols(elements[atom_types])
        self.data = ase_to_ovito(structure)
        self.pipeline = Pipeline(source=StaticSource(data=self.data))

    def set_data(self, pipeline, data):
        self.pipeline, self.data = pipeline, data

    def get_swipe_index(self, atom_types):
        is_same_atom = True
        while is_same_atom:
            i1, i2 = np.random.choice(
                self.natoms, 2, replace=True
            )  # replace=False means can't choose i1 = i2, but use true ow is so slow
            is_same_atom = atom_types[i1] == atom_types[i2]
        return i1, i2

    def get_NN(self, nneigh):
        finder = NearestNeighborFinder(
            nneigh, self.data
        )  # Computes atom neighbor lists.
        neigh_index_list = finder.find_all()[0]  # for each atom its list of neight
        return neigh_index_list

    def get_wc(self, atom_types):
        neigh_atom_types = atom_types[self.neigh_index_list]

        atom_counts = np.bincount(atom_types)
        c = atom_counts / self.natoms
        self.c = c
        self.atom_counts = atom_counts

        pairs = list(product(range(self.ncomponent), repeat=2))
        self.pairs = pairs

        dim = int(np.sqrt(len(pairs)))
        alpha = np.zeros((dim, dim))
        f = np.zeros((dim, dim))

        for pair in pairs:
            a, b = pair
            # fraction of A atoms in NN shell given some B atoms center
            fA = (
                np.count_nonzero(neigh_atom_types[atom_types == b] == a, axis=1)
                / neigh_atom_types.shape[1]
            )

            sum_f = np.sum(fA)
            f[a, b] = sum_f

            alpha[a, b] = 1 - 1 / c[a] * sum_f / atom_counts[b]
        return alpha, f

    def update_wc(
        self,
        i1,
        i2,
        new_atom_types,
        atom_types,
        f,
    ):
        Nb = self.neigh_index_list.shape[1]  # Define Nb as a constant variable

        new_f = np.copy(f)
        # Get the neighborhood indices for i1 and i2
        neigh_index_list_i1 = self.neigh_index_list[i1]
        neigh_index_list_i2 = self.neigh_index_list[i2]

        # Get the atom types for i1 and i2
        old_center_type_i1 = atom_types[i1]
        old_center_type_i2 = atom_types[i2]
        new_center_type_i1 = new_atom_types[i1]
        new_center_type_i2 = new_atom_types[i2]

        # Compute the contributions for i1
        n_center_i1 = new_atom_types[neigh_index_list_i1].astype(int)

        new_f[n_center_i1, old_center_type_i1] -= 1 / Nb
        new_f[n_center_i1, new_center_type_i1] += 1 / Nb
        new_f[old_center_type_i1, n_center_i1] -= 1 / Nb
        new_f[new_center_type_i1, n_center_i1] += 1 / Nb

        # Compute the contributions for i2
        n_center_i2 = new_atom_types[neigh_index_list_i2].astype(int)
        new_f[n_center_i2, old_center_type_i2] -= 1 / Nb
        new_f[n_center_i2, new_center_type_i2] += 1 / Nb
        new_f[old_center_type_i2, n_center_i2] -= 1 / Nb
        new_f[new_center_type_i2, n_center_i2] += 1 / Nb

        # atom_counts = np.bincount(new_atom_types)
        # c = atom_counts / self.natoms
        atom_counts = self.atom_counts
        c = self.c

        new_wc = np.zeros((self.ncomponent, self.ncomponent))
        for pair in self.pairs:
            a, b = pair
            new_wc[a, b] = 1 - 1 / c[a] * new_f[a, b] / atom_counts[b]

        return new_wc, new_f

    def modify(self, frame, data, new_atom_types):
        data.particles_.create_property("Particle Type", data=new_atom_types)

    def save_ovito_snapshot(self, new_atom_types, save_file_name):
        self.pipeline.modifiers.append(
            partial(self.modify, new_atom_types=new_atom_types)
        )
        cols = [
            "Particle Identifier",
            "Particle Type",
            "Position.X",
            "Position.Y",
            "Position.Z",
        ]
        export_file(
            self.pipeline,
            save_file_name,
            "lammps/dump",
            columns=cols,
        )

    def run(self, nneigh, T, tol_percent_diff, save_file_name):
        assert self.data is not None
        assert self.target_wc is not None
        assert self.pipeline is not None

        # Getting some atom types related properties
        atom_types = (
            self.data.particles["Particle Type"] - 1
        )  # reindxing to atom type 0
        self.ncomponent = len(np.unique(atom_types))
        self.natoms = len(atom_types)

        # Getting nearest neighbors
        self.neigh_index_list = self.get_NN(nneigh=nneigh)

        # Getting inital wc parameters
        wc_init, f = self.get_wc(atom_types)
        wc = wc_init
        # Computing WC energies
        wc_energy = np.sum((self.target_wc - wc_init) ** 2)
        percent_diff = np.ones(wc.shape) * 100

        i = 0
        print("---------- Starting MC iteration --------------")
        while np.any(percent_diff > tol_percent_diff):
            i += 1
            count_accept = 0

            # Getting indexes to swap
            i1, i2 = self.get_swipe_index(atom_types)

            new_atom_types = np.copy(atom_types)

            new_atom_types[i1], new_atom_types[i2] = atom_types[i2], atom_types[i1]

            new_wc, new_f = self.update_wc(i1, i2, new_atom_types, atom_types, f)

            new_wc_energy = np.sum((self.target_wc - new_wc) ** 2)

            dE = new_wc_energy - wc_energy

            if dE < 0:
                accept = True
            else:
                r1 = np.random.random()

                wc_cond = min(1, np.exp(-1 / T * dE))
                accept = r1 < wc_cond

            if accept:
                count_accept += 1
                atom_types = new_atom_types
                wc_energy = new_wc_energy
                wc = new_wc
                f = new_f

                percent_diff = np.abs((wc - self.target_wc) / self.target_wc) * 100

            if i % 10000 == 0:
                print("\n")
                print(f"Frac of accepted: {count_accept/i}")
                print(f"WC target is {self.target_wc}")
                print(f"Current WC is {wc}")
                print(f"Energy is {wc_energy}")
                print(f"Percent error {percent_diff}")
                print("\n")
        print("---------- Tolerence criteria reached --------------")
        print("\n")
        print(f"Frac of accepted: {count_accept/i}")
        print(f"WC target is {self.target_wc}")
        print(f"Current WC is {wc}")
        print(f"Energy is {wc_energy}")
        print(f"Percent error {percent_diff}")
        print("\n")

        self.save_ovito_snapshot(
            new_atom_types=atom_types, save_file_name=save_file_name
        )
