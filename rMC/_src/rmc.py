from copy import deepcopy
from itertools import product

import numpy as np
from ase import Atoms
from ase.spacegroup import crystal
from ovito.data import NearestNeighborFinder
from ovito.io import import_file
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource
from tqdm import tqdm


class rMC:
    def __init__(
        self, crystal_structure, dimension, concentrations, elements, target_wc, save_folder
    ):
        self.crystal_structure = crystal_structure
        self.dimension = dimension
        self.concentrations = concentrations
        self.elements = elements
        self.target_wc = target_wc
        self.save_folder = save_folder

        # self.pipeline, self.data = self.ase_2_ovito(self.generate_ase_structure())
        self.pipeline = import_file(
            "/home/ksheriff/PAPERS/first_paper/03_mtp/data/dumps/dumps_rss_mtp_mc/RSS_relaxation_20_1_300K.dump"
        )
        self.data = self.pipeline.compute()

    def generate_ase_structure(self):
        pass

    def ase_2_ovito(self, ase_bulk):
        data = ase_to_ovito(ase_bulk)
        pipeline = Pipeline(source=StaticSource(data=data))

        return pipeline, data

    def get_probability(
        self,
    ):
        avg_bond_concentration = np.zeros((self.ncomponent, self.ncomponent))
        atom_types = np.zeros(self.natoms)

        for iatom in range(self.natoms):
            # center_type =
            pass

    def get_swipe_index(self, atom_types):
        """get i1, i2 so that atom types are different and that the 2 atoms shells are not shared"""
        is_same_atom, is_in_1nn_or_2nn = True, True

        while is_same_atom == True or is_in_1nn_or_2nn == True:
            i1, i2 = np.random.choice(
                self.natoms, 2, replace=False
            )  # replace=False means can't choose i1 = i2

            is_same_atom = atom_types[i1] == atom_types[i2]
            is_in_1nn_or_2nn = bool(set(self.neigh_index_list[i1]) & set(self.neigh_index_list[i2]))

        return i1, i2

    def get_NN(self, nneigh):
        finder = NearestNeighborFinder(nneigh, self.data)  # Computes atom neighbor lists.
        neigh_index_list = finder.find_all()[0]  # for each atom its list of neight
        return neigh_index_list

    def get_wc(self, atom_types):
        neigh_atom_types = atom_types[self.neigh_index_list]

        atom_counts = np.bincount(atom_types)
        c = atom_counts / self.natoms

        combinations = list(product(range(self.ncomponent), repeat=2))

        dim = int(np.sqrt(len(combinations)))
        alpha = np.zeros((dim, dim))

        for pair in combinations:
            a, b = pair
            # fraction of A atoms in NN shell given some B atoms center
            fA = (
                np.count_nonzero(neigh_atom_types[atom_types == b] == a, axis=1)
                / neigh_atom_types.shape[1]
            )

            # Number of b atoms
            Nb = len(atom_types[atom_types == b])

            alpha[a, b] = 1 - 1 / c[a] * np.sum(fA) / Nb
        return alpha

    def run(self, T, n_iter, tol_percent_diff):
        # Getting some atom types related properties
        atom_types = self.data.particles["Particle Type"] - 1  # reindxing to atom type 0
        self.ncomponent = len(np.unique(atom_types))
        self.natoms = len(atom_types)

        # Getting nearest neighbors
        self.neigh_index_list = self.get_NN(nneigh=12)

        # Getting inital wc parameters
        wc_init = self.get_wc(atom_types)
        wc = wc_init
        # Computing WC energies
        wc_energy = np.sum((self.target_wc - wc_init) ** 2)
        percent_diff = np.ones(wc.shape) * 1

        i = 0

        while np.any(percent_diff > tol_percent_diff):
            # for i in tqdm(range(n_iter)):
            i += 1
            count_accept = 0
            # Getting indexes to swap
            i1, i2 = self.get_swipe_index(atom_types)

            new_atom_types = deepcopy(atom_types)
            new_atom_types[i1], new_atom_types[i2] = atom_types[i2], atom_types[i1]
            new_wc = self.get_wc(new_atom_types)  # self.update_wc(i1, i2, new_atom_types)

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

                percent_diff = np.abs((wc - self.target_wc) / self.target_wc)

        print(f"Frac of accepted: {count_accept/i}")
        print(f"WC target is {self.target_wc}")
        print(f"Current WC is {wc}")
        print(f"Energy is {wc_energy}")
        print(f"Percent error {percent_diff}")


if __name__ == "__main__":
    rmc = rMC(
        crystal_structure="fcc",
        dimension=(10, 10, 10),
        concentrations=(1 / 3, 1 / 3, 1 / 3),
        elements=("Ni", "Co", "Cr"),
        target_wc=np.load(
            "/home/ksheriff/PAPERS/first_paper/03_mtp/data/eca_id_temperature/300K/wc_3x3.npy"
        ),  # np.array([[1, -0.5, -0.5], [-1 / 2, 1, -1 / 2], [-1 / 2, -1 / 2, 1]]),
        save_folder="",
    )
    rmc.run(T=1 / 1e6, n_iter=10000000, tol_percent_diff=np.ones((3, 3)) * 0.01)
