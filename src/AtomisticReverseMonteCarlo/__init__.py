from itertools import product, combinations_with_replacement

import numpy as np
from ovito.data import DataCollection, NearestNeighborFinder, DataTable
from ovito.pipeline import ModifierInterface
from traits.api import Int, List, Range, Union


class AtomisticReverseMonteCarlo(ModifierInterface):
    nneigh = Range(low=1, high=None, value=12,
                   label="Max number of neighbors for WC")
    T = Range(low=0.0, high=None, value=1e-9, label="rMC temperature")
    seed = Union(None, Range(low=0, high=2**32-1), label="Seed")
    tol_percent_diff = List(
        List(Range(low=0.0, high=None)),
        value=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        label="Tolerence criteria to stop rMC\n(percent difference between WC params)",
    )
    target_wc = List(
        List(Range(low=None, high=1.0)),
        value=[
            [0.32719603, -0.19925471, -0.12794131],
            [-0.19925471, 0.06350427, 0.13575045],
            [-0.12794131, 0.13575045, -0.00762235],
        ],
        label="Target WC parameters",
    )
    save_rate = Int(100000, label="Save rate")
    max_iter = Union(None, Range(low=0, high=None), label="Maximum iterations")

    def compute_trajectory_length(self,  data_cache: DataCollection, **kwargs):
        try:
            return data_cache.attributes["num_frames"]
        except KeyError:
            return 0

    def validate_input(self, num_species):
        if len(self.target_wc) != len(self.tol_percent_diff):
            raise ValueError(
                f"Tolerance matrix and Target matrix need to be the same size, not {len(self.target_wc)} and { len(self.tol_percent_diff)}"
            )

        if len(self.target_wc) != num_species:
            raise ValueError(
                f"Target matrix dimensions ({len(self.target_wc)}) need to match the number of species ({num_species}) in the system.")

        odim = len(self.tol_percent_diff)
        for i in range(odim):
            if len(self.tol_percent_diff[i]) != odim:
                raise ValueError(
                    f"Tolerance matrix needs to be NxN, not {len(self.tol_percent_diff[i])}x{odim}."
                )

        for i in range(odim):
            if len(self.target_wc[i]) != odim:
                raise ValueError(
                    f"Target matrix needs to be NxN, not {len(self.target_wc[i])}x{odim}."
                )

        for i in range(odim):
            for j in range(i + 1, odim):
                if not np.isclose(self.target_wc[j][i], self.target_wc[i][j]):
                    raise ValueError(f"Target matrix needs to be symmetric.")

    @staticmethod
    def get_type_name(data, id):
        ptype = data.particles["Particle Type"].type_by_id(id)
        name = ptype.name
        if name:
            return name
        return f"{id}"

    def set_target_wc(self, target_wc):
        target_wc = target_wc

    def get_swipe_index(self, atom_types, natoms, nn_index, rng):
        is_same_atom, is_in_1nn_or_2nn = True, True

        while is_same_atom == True or is_in_1nn_or_2nn == True:
            i1, i2 = rng.choice(natoms, 2)

            is_same_atom = atom_types[i1] == atom_types[i2]

            g1, g2 = nn_index[i1], nn_index[i2]
            is_in_1nn_or_2nn = bool(set(g1) & set(g2))

        return i1, i2

    def get_NN(self, nneigh, data):
        # Computes atom neighbor lists.
        finder = NearestNeighborFinder(nneigh, data)
        # for each atom its list of neight
        neigh_index_list = finder.find_all()[0]
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
                np.count_nonzero(
                    neigh_atom_types[atom_types == b] == a, axis=1)
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

    def modify(self, data: DataCollection, frame: int, data_cache: DataCollection, **kwargs):
        # Validate input
        self.validate_input(len(data.particles.particle_types.types))
        if "num_frames" not in data_cache.attributes:
            if frame != 0:
                raise RuntimeError(
                    f"Only static snapshots supported as input trajectory. Please start at frame 0.")

        (nneigh, T, tol_percent_diff, target_wc) = (
            self.nneigh,
            self.T,
            np.array(self.tol_percent_diff),
            np.array(self.target_wc),
        )

        # No cached results available -> run MC
        if "num_frames" not in data_cache.attributes:

            rng = np.random.default_rng(self.seed)

            # Getting some atom types related properties
            # reindxing to atom type 0
            atom_types = data.particles["Particle Type"] - 1
            ncomponent = len(np.unique(atom_types))
            natoms = len(atom_types)

            # Getting nearest neighbors
            neigh_index_list = self.get_NN(nneigh=nneigh, data=data)

            # Getting inital wc parameters
            wc_init, f, pairs = self.get_wc(
                atom_types, neigh_index_list, ncomponent, natoms)

            wc = wc_init
            # Computing WC energies
            wc_energy = np.sum((target_wc - wc_init) ** 2)
            percent_diff = np.ones(wc.shape) * 100

            i = 0

            # Add zeroth frame to cache
            step_trajectory = [i]
            wc_trajectory = [wc]
            wc_error_trajectory = [np.abs((wc - target_wc) / target_wc) * 100]
            pt_trajectory = [atom_types]

            max_iter = self.max_iter if self.max_iter is not None else np.inf
            iteration = 0
            while (iteration < max_iter) and np.any(percent_diff > tol_percent_diff):
                i += 1
                count_accept = 0

                # Getting indexes to swap
                i1, i2 = self.get_swipe_index(
                    atom_types, natoms, neigh_index_list, rng)

                new_atom_types = np.copy(atom_types)

                new_atom_types[i1], new_atom_types[i2] = atom_types[i2], atom_types[i1]

                new_wc, new_f = self.update_wc(
                    i1, i2, new_atom_types, atom_types, f, neigh_index_list, natoms, ncomponent, pairs
                )

                new_wc_energy = np.sum((target_wc - new_wc) ** 2)

                dE = new_wc_energy - wc_energy

                if dE < 0:
                    accept = True
                else:
                    r1 = rng.random()

                    wc_cond = min(1, np.exp(-1 / T * dE))
                    accept = r1 < wc_cond

                if accept:
                    count_accept += 1

                    atom_types = new_atom_types
                    wc_energy = new_wc_energy
                    wc = new_wc
                    f = new_f

                    percent_diff = np.abs((wc - target_wc) / target_wc) * 100

                if i % self.save_rate == 0:
                    step_trajectory.append(i)
                    wc_trajectory.append(wc)
                    wc_error_trajectory.append(percent_diff)
                    pt_trajectory.append(atom_types)

                iteration += 1
                yield

            # Final configuration
            if i % self.save_rate != 0:
                step_trajectory.append(i)
                wc_trajectory.append(wc)
                wc_error_trajectory.append(percent_diff)
                pt_trajectory.append(atom_types)

            # Add results to cache
            data_cache.attributes["step_trajectory"] = np.array(
                step_trajectory)
            data_cache.attributes["wc_trajectory"] = np.array(wc_trajectory)
            data_cache.attributes["wc_error_trajectory"] = np.array(
                wc_error_trajectory)
            data_cache.attributes["pt_trajectory"] = np.array(pt_trajectory)
            data_cache.attributes["num_frames"] = len(pt_trajectory)
            # Refresh frame counter
            self.notify_trajectory_length_changed()

        # MC has run!
        # Populate data collection from cache
        data.particles_[
            "Particle Type_"][...] = data_cache.attributes["pt_trajectory"][frame]+1
        data.attributes["Warren-Cowley parameters"] = data_cache.attributes["wc_trajectory"][frame]
        data.attributes["Target Warren-Cowley parameters"] = target_wc
        data.attributes["Warren-Cowley percent error"] = data_cache.attributes["wc_error_trajectory"][frame]
        data.attributes["Timestep"] = data_cache.attributes["step_trajectory"][frame]

        # Table output
        pairs = list(combinations_with_replacement(
            range(len(data_cache.attributes["wc_trajectory"][0])), 2))
        labels = [
            f'{self.get_type_name(data, i+1)}-{self.get_type_name(data, j+1)}' for i, j in pairs]

        # Warren-Cowley parameters
        table = data.tables.create(
            identifier='wc_parameters', plot_mode=DataTable.PlotMode.Line, title='Warren-Cowley parameters')
        table.x = table.create_property(
            'MC Step', data=data_cache.attributes["step_trajectory"])
        output = np.empty((data_cache.attributes["num_frames"], len(pairs)))
        for i, (j, k) in enumerate(pairs):
            output[:, i] = data_cache.attributes["wc_trajectory"][:, j, k]
        table.y = table.create_property(
            'WC ij', data=output, components=labels)

        # Table output
        # Warren-Cowley percentage error
        table = data.tables.create(
            identifier='wc_error', plot_mode=DataTable.PlotMode.Line, title='Warren-Cowley percentage error')
        table.x = table.create_property(
            'MC Step', data=data_cache.attributes["step_trajectory"])
        pairs = list(combinations_with_replacement(
            range(len(data_cache.attributes["wc_error_trajectory"][0])), 2))
        for i, (j, k) in enumerate(pairs):
            output[:, i] = data_cache.attributes["wc_error_trajectory"][:, j, k]
        table.y = table.create_property(
            'WC ij error', data=output, components=labels)

        # Table output
        # Log Warren-Cowley percentage error
        table = data.tables.create(
            identifier='log_wc_error', plot_mode=DataTable.PlotMode.Line, title='Log Warren-Cowley percentage error')
        table.x = table.create_property(
            'MC Step', data=data_cache.attributes["step_trajectory"])
        pairs = list(combinations_with_replacement(
            range(len(data_cache.attributes["wc_error_trajectory"][0])), 2))
        for i, (j, k) in enumerate(pairs):
            output[:, i] = np.log(
                data_cache.attributes["wc_error_trajectory"][:, j, k])
        table.y = table.create_property(
            'Log10(WC ij error)', data=output, components=labels)
