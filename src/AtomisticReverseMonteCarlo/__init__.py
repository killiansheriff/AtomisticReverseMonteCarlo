from itertools import product

import numpy as np
from ovito.data import DataCollection, NearestNeighborFinder
from ovito.pipeline import ModifierInterface
from traits.api import Float, Int, List


class AtomisticReverseMonteCarlo(ModifierInterface):
    nneigh = Int(12, label="")
    T = Float(1e-9, label="")
    tol_percent_diff = List(List, value=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    target_wc = List(
        List,
        value=[
            [
                [0.32719603, -0.19925471, -0.12794131],
                [-0.19925471, 0.06350427, 0.13575045],
                [-0.12794131, 0.13575045, -0.00762235],
            ]
        ],
    )

    save_file_name = "fcc_wc.dump"

    def set_target_wc(self, target_wc):
        self.target_wc = target_wc

    def get_swipe_index(self, atom_types, natoms):
        is_same_atom = True
        while is_same_atom:
            i1, i2 = np.random.choice(
                natoms, 2, replace=True
            )  # replace=False means can't choose i1 = i2, but use true ow is so slow
            is_same_atom = atom_types[i1] == atom_types[i2]
        return i1, i2

    def get_NN(self, nneigh, data):
        finder = NearestNeighborFinder(nneigh, data)  # Computes atom neighbor lists.
        neigh_index_list = finder.find_all()[0]  # for each atom its list of neight
        return neigh_index_list

    def get_wc(
        self,
        atom_types,
        neigh_index_list,
        ncomponent,
        natoms,
    ):
        neigh_atom_types = atom_types[neigh_index_list]

        atom_counts = np.bincount(atom_types)
        c = atom_counts / natoms

        pairs = list(product(range(ncomponent), repeat=2))

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
        return alpha, f, pairs

    def update_wc(
        self, i1, i2, new_atom_types, atom_types, f, neigh_index_list, natoms, ncomponent, pairs
    ):
        Nb = neigh_index_list.shape[1]  # Define Nb as a constant variable

        new_f = np.copy(f) * Nb
        # Get the neighborhood indices for i1 and i2
        neigh_index_list_i1 = neigh_index_list[i1]
        neigh_index_list_i2 = neigh_index_list[i2]

        # Get the atom types for i1 and i2
        old_center_type_i1 = atom_types[i1]
        old_center_type_i2 = atom_types[i2]
        new_center_type_i1 = new_atom_types[i1]
        new_center_type_i2 = new_atom_types[i2]

        # Compute the contributions for i1
        n_center_i1 = new_atom_types[neigh_index_list_i1].astype(int)
        # Compute the contributions for i2
        n_center_i2 = new_atom_types[neigh_index_list_i2].astype(int)

        for idx in range(len(n_center_i1)):
            new_f[n_center_i1[idx], old_center_type_i1] -= 1
            new_f[old_center_type_i1, n_center_i1[idx]] -= 1
            new_f[n_center_i1[idx], new_center_type_i1] += 1
            new_f[new_center_type_i1, n_center_i1[idx]] += 1

            new_f[n_center_i2[idx], old_center_type_i2] -= 1
            new_f[old_center_type_i2, n_center_i2[idx]] -= 1
            new_f[n_center_i2[idx], new_center_type_i2] += 1
            new_f[new_center_type_i2, n_center_i2[idx]] += 1

        new_f *= 1 / Nb

        # Need to be recomputed
        atom_counts = np.bincount(new_atom_types)
        c = atom_counts / natoms

        new_wc = np.zeros((ncomponent, ncomponent))

        for pair in pairs:
            a, b = pair
            new_wc[a, b] = 1 - 1 / c[a] * new_f[a, b] / atom_counts[b]

        return new_wc, new_f

    def modify(self, data: DataCollection, frame: int, **kwargs):
        nneigh, T, tol_percent_diff, save_file_name = (
            self.nneigh,
            self.T,
            np.array(self.tol_percent_diff),
            self.save_file_name,
        )

        # Getting some atom types related properties
        atom_types = data.particles["Particle Type"] - 1  # reindxing to atom type 0
        ncomponent = len(np.unique(atom_types))
        natoms = len(atom_types)

        # Getting nearest neighbors
        neigh_index_list = self.get_NN(nneigh=nneigh, data=data)

        # Getting inital wc parameters
        wc_init, f, pairs = self.get_wc(atom_types, neigh_index_list, ncomponent, natoms)

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
            i1, i2 = self.get_swipe_index(atom_types, natoms)

            new_atom_types = np.copy(atom_types)

            new_atom_types[i1], new_atom_types[i2] = atom_types[i2], atom_types[i1]

            new_wc, new_f = self.update_wc(
                i1, i2, new_atom_types, atom_types, f, neigh_index_list, natoms, ncomponent, pairs
            )

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

            if i % 100000 == 0:
                print("\n")
                # print(f"Frac of accepted: {count_accept/i}")
                print(f"WC target: \n {self.target_wc}")
                print(f"Current WC: \n  {wc}")
                # print(f"Energy is {wc_energy}")
                print(f"Percent error:{percent_diff}")
                print("\n")
        print("---------- Tolerence criteria reached --------------")
        print("\n")
        # print(f"Frac of accepted: {count_accept/i}")
        print(f"WC target: \n {self.target_wc}")
        print(f"Current WC: \n {wc}")
        # print(f"Energy: {wc_energy}")
        print(f"Percent error: \n {percent_diff}")
        print("\n")

        data.particles_.create_property(
            "Particle Type",
            data=atom_types + 1,
        )
