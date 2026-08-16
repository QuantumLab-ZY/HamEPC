'''
Descripttion: 
version: 
Author: Yang Zhong & Shixu Liu
Date: 2023-04-07 14:24:16
LastEditors: Yang Zhong
LastEditTime: 2024-05-16 23:42:25
'''
import os
import time
import yaml
import numpy as np
from scipy.linalg import eigh
from .utils import *
from tqdm import tqdm
import phonopy
from phonopy.structure.atoms import atom_data
from easydict import EasyDict
from tqdm import tqdm
import opt_einsum as oe
import spglib
from collections import Counter
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.core.periodic_table import Element
from mpi4py import MPI
from numpy_extension import eliashberg_spectrum_cal_helper_sparse

# Upper bound on the H(k)/S(k) arrays a single batched diagonalisation may allocate.
# The batch sizes below are derived from it so that memory stays flat in norbs.
_BATCH_BYTES = 256 * 1024 * 1024

class EPC_calculator(object):

    def __str__(self):
        class_list = []
        function_list = []
        variable_list = []
        for name in dir(self):
            if name.startswith("__"):
                continue
            if type(getattr(self, name)) == type:
                class_list.append(f"  Class {name}")
            elif callable(getattr(self, name)):
                function_list.append(f"  Function {name}")
            else:
                variable_list.append(f"  Variable {name}")
        output_str = "Class EPC_calculator:\n"
        output_str = output_str + '\n'.join(class_list) + '\n'
        output_str = output_str + '\n'.join(function_list) + '\n'
        output_str = output_str + '\n'.join(variable_list)
        return output_str

    def __init__(self, config:dict=None, comm=None):
        if config == None:
            print("This is EPC_calculator")
            return

        # parallelization
        self.comm:MPI.Intracomm = comm
        if self.comm is None:
            self.rank = 0
            self.rank_size = 1
        else:
            self.rank = self.comm.Get_rank()
            self.rank_size = self.comm.Get_size()

        # load default paramters
        for blockname in default_parameters.keys():
            for property_name, value in default_parameters[blockname].items():
                setattr(self, property_name, value)

        # read input
        config = EasyDict(config)
        self._parse_input(config=config)

        # initial running varibles
        self.efermi:float = 0.0 # the fermi energy
        self.carrier_density:float = 0.0    # the carrier density
        self.weight_k:list[float] = None   # the weights of k grid
        self.weight_q:list[float] = None  # the weights of q grid
        self.full2irr:np.ndarray = None # the index that can fold full grid to irr grid

    def _initial_transport(self):
        self.temperature = self.temperature * Hamcts.KELVINtoHARTREE
        self.smeark = self.smeark * Hamcts.MEVtoHARTREE
        self.inv_smeark = 1.0 / self.smeark
        self.smearq = self.smearq * Hamcts.MEVtoHARTREE
        self.inv_smearq = 1.0 / self.smearq
        self.e_thr = self.e_thr * Hamcts.MEVtoHARTREE
        self.phonon_cutoff = self.phonon_cutoff * Hamcts.MEVtoHARTREE
        self.bands_indices = np.array([each - int(1) for each in self.bands_indices], dtype="int64")
        
    def _initial_mobility(self):
        self.over_cbm = self.over_cbm * Hamcts.EVtoHARTREE
        self.over_vbm = self.over_vbm * Hamcts.EVtoHARTREE
        self.MC_sampling = self.MC_sampling.lower()
        if self.MC_sampling == 'cauchy':
            if self.rank == 0: print('Sampling with Cauchy distribution.')
        elif self.MC_sampling == 'uniform':
            if self.rank == 0: print('Sampling with uniform distribution.')
        else:
            self.MC_sampling = 'none'
            if self.rank == 0: print('No important sampling.')
        self.ncarrier = self.ncarrier / (Hamcts.CMtoBOHR ** 3)
        self.mob_level = self.mob_level.lower()
        if self.mob_level in ['erta']:
            if self.rank == 0: print(f"Using {self.mob_level.upper()} method to calculate mobility.")
        else:
            raise NotImplementedError(f"Mobility method {self.mob_level.upper()} is not implemented.", '7001')
        # Matrix of momentum operator
        if self.read_momentum:
            graph_data = np.load(self.graph_data_path_uc, allow_pickle=True)
            graph_data = graph_data['graph'].item()
            graph_data = list(graph_data.values())[0]
            self.graph_data.Mon = graph_data.Mon.numpy().reshape(-1, self.nao_max, self.nao_max, 3)
            self.graph_data.Moff = graph_data.Moff.numpy().reshape(-1, self.nao_max, self.nao_max, 3)
            self.graph_data.M_cell = self._M_cell_prepare() # shape: (ncells, nao_max, nao_max, 3)

    def _initial_superconduct(self):
        if (type(self.mius) != list) or (len(self.mius) != 3) or (type(self.mius[0]) != float) or (type(self.mius[1]) != float)or  (type(self.mius[2]) != float):
            raise RuntimeError("The mius should be set as [miu_min, miu_max, miu_step]", '6604')
        self.mius = np.array(self.mius)
        self.mius = np.arange(self.mius[0], self.mius[1], self.mius[2])
        self.omega_range = np.array(self.omega_range) * Hamcts.MEVtoHARTREE
        self.omega_step = self.omega_step * Hamcts.MEVtoHARTREE

    def _initial_epc(self):
        if os.path.isfile(self.mat_info_rc_path) and os.access(self.mat_info_rc_path, os.R_OK):
            self.mat_info_sc = np.load(self.mat_info_rc_path, allow_pickle=True).item()
        else:
            raise RuntimeError(f"Cannot read mat_info_rc from {self.mat_info_rc_path}.", '4001')
        self.cell_shift_array_reduced = self.mat_info_sc['cell_shift_array'] # shape: (ncells, 3)
        self.cell_index_map_reduced = self.mat_info_sc['cell_index_map'] # len: (ncells,) # index each cell_shift
        self.p2s_indices_reduced = self.mat_info_sc['p2s_indices'] # shape: (natoms_uc,)
        self.s2u_list_reduced = self.mat_info_sc['s2u_list'] # shape: (natoms_sc,)
        self.cell_shift_of_each_atom_in_sc = self.mat_info_sc['cell_shift_of_each_atom'] # shape: (natoms_sc,3)
        self.cell_cut_list = []
        for icell, cell_shift in enumerate(self.cell_shift_array_reduced):
            if (np.abs(cell_shift[0])<self.cell_range_cut[0]) and (np.abs(cell_shift[1])<self.cell_range_cut[1]) and (np.abs(cell_shift[2])<self.cell_range_cut[2]):
                self.cell_cut_list.append(icell)
        self.cell_cut_array = np.array(self.cell_cut_list)
        if self.split_orbits:
            if self.rank == 0:
                grad_mat = np.load(self.grad_mat_path)[self.cell_cut_array[:,None], self.cell_cut_array[None,:]]
                self.grad_mat_split = split_array_along_2axes(grad_mat, axis1=2, axis2=3, sizes1=self.orbital_splits, sizes2=self.orbital_splits)
            else:
                self.grad_mat_split = [[None] * self.split_orbits_num_blocks  for _ in range(self.split_orbits_num_blocks)]
        else:
            # --- MPI shared memory: load grad_mat once per node ---
            # Every rank used to hold its own copy (or re-read it through mmap on every
            # access), so a node running N ranks needed N copies of it.
            # Allocating a shared-memory window instead means one copy per node, which
            # both removes the duplicated RAM and the repeated mmap disk I/O.
            try:
                node_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
                node_rank = node_comm.Get_rank()

                # Read the header through mmap to get shape/dtype without loading data.
                _mmap_tmp = np.load(self.grad_mat_path, mmap_mode='r')
                _full_shape = _mmap_tmp.shape
                _dtype = _mmap_tmp.dtype

                # Work out the cell subset that is actually needed.
                _n_all = len(self.cell_shift_array_reduced)
                _no_cut = (len(self.cell_cut_array) == _n_all and
                           np.all(self.cell_cut_array == np.arange(_n_all)))
                if _no_cut:
                    _target_shape = _full_shape
                else:
                    nc = len(self.cell_cut_array)
                    _target_shape = (nc, nc) + _full_shape[2:]

                _nbytes = int(np.prod(_target_shape)) * np.dtype(_dtype).itemsize

                # Rank 0 of each node owns the allocation; the others map 0 bytes.
                if node_rank == 0:
                    win = MPI.Win.Allocate_shared(_nbytes, np.dtype(_dtype).itemsize, comm=node_comm)
                else:
                    win = MPI.Win.Allocate_shared(0, np.dtype(_dtype).itemsize, comm=node_comm)

                # All ranks obtain a pointer to the same buffer.
                _buf, _itemsize = win.Shared_query(0)
                self.grad_mat = np.ndarray(_target_shape, dtype=_dtype, buffer=_buf)

                # Rank 0 of each node fills the shared buffer from disk.
                if node_rank == 0:
                    _t_load = time.time()
                    if _no_cut:
                        for iA in range(_full_shape[0]):
                            self.grad_mat[iA] = _mmap_tmp[iA]
                    else:
                        _cut = self.cell_cut_array
                        for i, iA in enumerate(_cut):
                            self.grad_mat[i] = _mmap_tmp[iA][_cut]
                    if self.rank == 0:
                        print("grad_mat loaded into shared memory in {:.1f}s. Shape: {}, {:.3f} GB/node".format(
                            time.time() - _t_load, _target_shape, _nbytes / 1e9), flush=True)

                del _mmap_tmp
                node_comm.Barrier()
                self._grad_mat_win = win  # keep a reference so the window is not collected

            except Exception as e:
                # Fall back to mmap when shared memory is unavailable.
                if self.rank == 0:
                    print(f"WARNING: MPI shared memory failed ({e}), falling back to mmap.", flush=True)
                full_grad_mmap = np.load(self.grad_mat_path, mmap_mode='r')
                _n_all = len(self.cell_shift_array_reduced)
                if (len(self.cell_cut_array) == _n_all and
                        np.all(self.cell_cut_array == np.arange(_n_all))):
                    self.grad_mat = full_grad_mmap
                else:
                    self.grad_mat = full_grad_mmap[self.cell_cut_array[:, None],
                                                   self.cell_cut_array[None, :], ...]
            if self.comm is not None:
                self.comm.Barrier()
        self.nbr_shift_of_cell_sc = np.einsum('ni, ij -> nj', self.cell_shift_array_reduced, self.graph_data.latt) # shape: (ncells, 3)
        if self.apply_correction:
            # Rests on the factorisation
            #     g_long(k, q, n->m, nu) = A_nu(q) * <psi_{m,k+q}|psi_{n,k}>
            # valid at LRC_taylor_order == 0 only, where the Ewald G sum holds no
            # electronic quantity and the overlap no G.  First order adds i(q+G).P(k),
            # which makes the electronic factor G-dependent and invalidates both caches;
            # it falls back to _dipole_correction_mat.
            self._lrc_cacheable = (self.LRC_taylor_order == 0)
            if self.LRC_taylor_order == 1:
                # The position-operator matrix elements are optional in the graph data; say so
                # instead of failing with an AttributeError inside _P_cell_prepare.
                if not (hasattr(self.graph_data, 'Pon') and hasattr(self.graph_data, 'Poff')):
                    raise RuntimeError(
                        "LRC_taylor_order == 1 needs the position-operator matrix elements "
                        "Pon and Poff in the graph data, and they are not present.  Set "
                        "LRC_taylor_order to 0, or regenerate the graph data with them.")
                self.graph_data.P_cell = self._P_cell_prepare()
            # 4 * alpha
            self.ewald_param = 4.0 * Hamcts.EWALD_SCALE * np.power(Hamcts.TWOPI / np.linalg.norm(self.graph_data.latt[0]), 2)
        else:
            self._lrc_cacheable = True
        
    def _initial_basic(self):
        if not (os.path.isfile(self.graph_data_path_uc) and os.access(self.graph_data_path_uc, os.R_OK)):
            raise RuntimeError(f"Cannot read graph_data from {self.graph_data_path_uc}.", '2001')
        else:
            graph_data = np.load(self.graph_data_path_uc, allow_pickle=True)
            graph_data = graph_data['graph'].item()
            graph_data = list(graph_data.values())[0]
            self.graph_data = EasyDict()
            self.graph_data.pos = graph_data.pos.numpy()
            self.graph_data.Son = graph_data.Son.numpy().reshape(-1, self.nao_max, self.nao_max)
            self.graph_data.Soff = graph_data.Soff.numpy().reshape(-1, self.nao_max, self.nao_max)
        
        if self.soc_switch:
            self.graph_data.Hon = graph_data.Hon.numpy().reshape(-1, 2*self.nao_max, 2*self.nao_max)
            self.graph_data.Hoff = graph_data.Hoff.numpy().reshape(-1, 2*self.nao_max, 2*self.nao_max)
            self.graph_data.iHon = graph_data.iHon.numpy().reshape(-1, 2*self.nao_max, 2*self.nao_max)
            self.graph_data.iHoff = graph_data.iHoff.numpy().reshape(-1, 2*self.nao_max, 2*self.nao_max)
        else:
            self.graph_data.Hon = graph_data.Hon.numpy().reshape(-1, self.nao_max, self.nao_max)
            self.graph_data.Hoff = graph_data.Hoff.numpy().reshape(-1, self.nao_max, self.nao_max)
        if ('Pon' in graph_data.keys) and ('Poff' in graph_data.keys):
            self.graph_data.Pon = graph_data.Pon.numpy().reshape(-1, self.nao_max, self.nao_max, 3)
            self.graph_data.Poff = graph_data.Poff.numpy().reshape(-1, self.nao_max, self.nao_max, 3)
        self.graph_data.latt = graph_data.cell.numpy().reshape(3,3)
        self.graph_data.lat_per_inv = np.linalg.inv(self.graph_data.latt).T
        self.graph_data.cell_shift = graph_data.cell_shift.numpy()
        self.graph_data.nbr_shift = graph_data.nbr_shift.numpy()
        self.graph_data.edge_index = graph_data.edge_index.numpy()
        self.graph_data.species = graph_data.z.numpy()
        if self.Ham_type == 'openmx':
            self.graph_data.num_electrons = np.sum(num_val_openmx[self.graph_data.species])
        elif self.Ham_type in ['siesta', 'honpas']:
            self.graph_data.num_electrons = np.sum(num_val_siesta[self.graph_data.species])
        if self.soc_switch:
            self.graph_data.num_VMB = self.graph_data.num_electrons-1
        else:
            self.graph_data.num_VMB = math.ceil(self.graph_data.num_electrons / 2) - 1
        # CBM and VBM
        if self.graph_data.num_electrons & 1 == 0:
            # Number of electrons are even
            self.has_unpair_electron = False
            self.VBM_band_index = np.round(self.graph_data.num_electrons / 2.0 - 1.0)
            self.CBM_band_index = self.VBM_band_index + 1
        else:
            # Number of electrons are odd
            self.has_unpair_electron = True
            self.VBM_band_index = np.round((self.graph_data.num_electrons - 1.0) / 2.0)
            self.CBM_band_index = self.VBM_band_index
        # cell volume
        self.volume_uc = np.linalg.det(self.graph_data.latt)
        self.inv_cell = 1.0 / self.volume_uc
        # Real space Hamiltonian matrix
        if self.soc_switch:
            H_cell, cell_shift_array, _, cell_index_map, _ = build_sparse_matrix_soc(self.graph_data.species, self.graph_data.cell_shift, self.nao_max, self.graph_data.Hon, self.graph_data.Hoff, 
                                    self.graph_data.iHon, self.graph_data.iHoff, self.graph_data.edge_index, return_raw_mat=False, Ham_type=self.Ham_type)
        else:
            H_cell, cell_shift_array, _, cell_index_map, _ = build_sparse_matrix(self.graph_data.species, self.graph_data.cell_shift, self.nao_max, 
                                                                                self.graph_data.Hon, self.graph_data.Hoff, self.graph_data.edge_index, return_raw_mat=False, Ham_type=self.Ham_type)
        S_cell, _, _, _, _ = build_sparse_matrix(self.graph_data.species, self.graph_data.cell_shift, self.nao_max, self.graph_data.Son, self.graph_data.Soff, self.graph_data.edge_index, return_raw_mat=False, Ham_type=self.Ham_type)
        nbr_shift_of_cell = np.einsum('ni, ij -> nj', cell_shift_array, self.graph_data.latt) # shape: (ncells, 3)
        self.norbs = H_cell.shape[-1]
        self.graph_data.H_cell = H_cell
        self.graph_data.S_cell = S_cell
        self.graph_data.cell_shift_array = cell_shift_array
        self.graph_data.nbr_shift_of_cell = nbr_shift_of_cell
        self.cell_index_map = cell_index_map
        self.graph_data.Hv_cell = self._Hv_cell_prepare()
        self.graph_data.Sv_cell = self._Sv_cell_prepare()
        self.natoms = len(self.graph_data.species)

    def _initial_advanced(self):
        if self.split_orbits:
            assert self.split_orbits_num_blocks < self.norbs
            orbital_splits = np.zeros(self.split_orbits_num_blocks, dtype=int)
            for i in range(self.norbs):
                orbital_splits[i%self.split_orbits_num_blocks] += 1
            self.orbital_splits = np.cumsum(orbital_splits, axis=0)            
    
    def _initial_phonon(self):
        self.atomic_mass = np.array([atom_data[ia][3] for ia in self.graph_data.species]) * Hamcts.AMU / Hamcts.MASS_E  # in atomic unit, i.e. mass of electron.
        # phonon calculator
        self.phonon = phonopy.load(supercell_matrix = self.supercell_matrix,
                                primitive_matrix = self.primitive_matrix,
                                unitcell_filename = self.unitcell_filename,
                                force_sets_filename = self.force_sets_filename
                                )
        # phonon non-analytical term correction settings
        if self.apply_correction:
            self.q_cut = self.q_cut * np.linalg.norm(self.graph_data.lat_per_inv[0])
            self.phonon.nac_params = {'born': self.BECs,
                                    'factor': Hamcts.BOHR * Hamcts.HARTREE,
                                    'dielectric': self.DL}

    def _initial_dispersion(self):
        self.high_symmetry_auto = False
        if (type(self.high_symmetry_points) != list) or len(self.high_symmetry_points) == 0:
            self.high_symmetry_auto = True
        else:
            for each in self.high_symmetry_points:
                if (type(each) != list) or (len(each) != 3) or (type(each[0]) != float) or (type(each[1]) != float) or (type(each[2]) != float):
                    self.high_symmetry_auto = True
        
        self.high_symmetry_k_vecs, self.high_symmetry_k_dist, self.high_symmetry_k_nodes, self.high_symmetry_labels = \
            self._get_hsk_path(nks_path=self.nks_path, hsk_points=self.high_symmetry_points, hsk_labels=self.high_symmetry_labels)
        self.dispersion_select_index = self.dispersion_select_index.strip()
        if self.cal_mode == 'band':
            if self.dispersion_select_index:
                try:
                    tmp = [np.arange(int(each.split('-')[0])-1, int(each.split('-')[1]), dtype="int64") \
                                                            for each in self.dispersion_select_index.split(',')]
                    self.dispersion_select_index = []
                    for each in tmp:
                        self.dispersion_select_index.extend(each.tolist())
                    self.dispersion_select_index = np.unique(np.array(self.dispersion_select_index, dtype="int64"))
                    if (self.dispersion_select_index > (self.norbs-1)).any():
                        raise RuntimeError(f"The dispersion_select_index is over the total number of bands, which is {self.norbs}")
                except:
                    raise RuntimeError("The setting of dispersion_select_index is wrong.", '1014')
            else:
                self.dispersion_select_index = np.arange(0, self.norbs, dtype="int64")
        elif self.cal_mode == 'phonon':
            nmodes_max = int(3) * self.natoms
            if self.dispersion_select_index:
                try:
                    tmp = [np.arange(int(each.split('-')[0])-1, int(each.split('-')[1]), dtype="int64") \
                                                            for each in self.dispersion_select_index.split(',')]
                    self.dispersion_select_index = []
                    for each in tmp:
                        self.dispersion_select_index.extend(each.tolist())
                    self.dispersion_select_index = np.unique(np.array(self.dispersion_select_index, dtype="int64"))
                    if (self.dispersion_select_index > nmodes_max-1).any():
                        raise RuntimeError(f"The dispersion_select_index is over the total number of branches, which is {nmodes_max}")
                except:
                    raise RuntimeError("The setting of dispersion_select_index is wrong.", '1014')
            else:
                self.dispersion_select_index = np.arange(0, nmodes_max, dtype="int64")
        elif self.cal_mode == 'epc':
            if (type(self.epc_path_fix_k) != list) or (len(self.epc_path_fix_k) != 3) or (type(self.epc_path_fix_k[0]) != float) or \
                (type(self.epc_path_fix_k[1]) != float) or (type(self.epc_path_fix_k[2]) != float):
                raise RuntimeError("The epc_path_fix_k must be a list contains three float elements.", '1015')
            if self.dispersion_select_index:
                try:
                    self.dispersion_select_index = np.array([int(each)-int(1) for each in self.dispersion_select_index.split(',')])
                    if len(self.dispersion_select_index) != 2:
                        raise RuntimeError("The dispersion_select_index must set as \'initial_state_band_index, final_state_band_index\', while using 'epc' calcultion mode.")
                    if (self.dispersion_select_index > (self.norbs-1)).any():
                        raise RuntimeError(f"The dispersion_select_index is over the total number of bands, which is {self.norbs}")
                except:
                    raise RuntimeError("The setting of dispersion_select_index is wrong.", '1014')
            else:
                raise RuntimeError("The dispersion_select_index must be specified, while using 'epc' calcultion mode.", '1016')

        else:
            raise NotImplementedError

    def _parse_input(self, config:EasyDict):
        
        if 'advanced' in config.keys():
            self._parse_input_optional(config=config, block_name='advanced')

        if 'basic' in config.keys():
            self._parse_input_basic(config=config)
            self._initial_basic()
            del config['basic']
        else:
            raise RuntimeError("You must set all parameters in basic part.", '1009')

        if 'advanced' in config.keys():
            self._initial_advanced()
            del config['advanced']

        if self.cal_mode == 'mobility':
            if self.rank == 0: print('#'*50+' Mobility Calculation '+'#'*50)
            self._parse_input_optional(config=config, block_name='phonon')
            self._initial_phonon()
            self._parse_input_optional(config=config, block_name='epc')
            self._initial_epc()
            self._parse_input_optional(config=config, block_name='transport')
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='mobility')
            self._initial_mobility()
            del config['phonon'], config['epc'], config['transport'], config['mobility']
        elif self.cal_mode == 'superconduct':
            if self.rank == 0: print('#'*50+' Superconductivity Calculation '+'#'*50)
            self._parse_input_optional(config=config, block_name='phonon')
            self._initial_phonon()
            self._parse_input_optional(config=config, block_name='epc')
            self._initial_epc()
            self._parse_input_optional(config=config, block_name='transport')
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='superconduct')
            self._initial_superconduct()
            del config['phonon'], config['epc'], config['transport'], config['superconduct']
        elif self.cal_mode == 'band':
            if self.rank == 0: print('#'*50+' Band Calculation '+'#'*50)
            self._parse_input_optional(config=config, block_name='dispersion')
            self._initial_dispersion()
            del config['dispersion']   
        elif self.cal_mode == 'phonon':
            if self.rank == 0: print('#'*50+' Phonon Calculation '+'#'*50)        
            self._parse_input_optional(config=config, block_name='phonon')
            self._initial_phonon()  
            self._parse_input_optional(config=config, block_name='dispersion')
            self._initial_dispersion()    
            del config['phonon'], config['dispersion']   
        elif self.cal_mode == 'epc':
            if self.rank == 0: print('#'*50+' EPC Calculation '+'#'*50)
            self._parse_input_optional(config=config, block_name='phonon')
            self._initial_phonon()
            self._parse_input_optional(config=config, block_name='epc')
            self._initial_epc()
            self._parse_input_optional(config=config, block_name='dispersion')
            self._initial_dispersion()    
            del config['phonon'], config['epc'], config['dispersion']

        if self.rank == 0: 
            for key in config.keys():
                print(f"Ignore block {key}.")
    
    def _parse_input_basic(self, config:EasyDict):
        if 'basic' in config.keys():
            config_basic = config.basic
        else:
            raise RuntimeError('You must give the basic settings in input!', '1001') 
        # read
        self.cal_mode = config_basic.cal_mode.lower()
        self.graph_data_path_uc = config_basic.graph_data_path_uc
        self.nao_max = config_basic.nao_max
        self.Ham_type = config_basic.Ham_type.lower()
        self.outdir = config_basic.outdir
        # check
        if self.cal_mode not in ['mobility', 'superconduct', 'band', 'phonon', 'epc']:
            raise NotImplementedError('The calculation mode is not supported!', '1002')
        if self.Ham_type not in ['openmx', 'honpas', 'siesta']:
            raise NotImplementedError('The Hamitonian type is not supported!', '1003')

    def _parse_input_optional(self, config:EasyDict, block_name:str):
        if block_name in config.keys():
            for property_name, value in config[block_name].items():
                if property_name in default_parameters[block_name].keys():
                    setattr(self, property_name, value)
                else:
                    if self.rank == 0:
                        print(f"Ignore {property_name} in {block_name} block.")

    def run(self):
        if self.cal_mode == 'band':
            self.plot_band()
        elif self.cal_mode == 'phonon':
            self.plot_phonon()
        elif self.cal_mode == 'epc':
            self.plot_epc()
        elif self.cal_mode == 'superconduct':
            self.superconductivity_cal()
        elif self.cal_mode == 'mobility':
            self.mobility_cal()

    def _get_monkhorst_pack(self, mesh, shift=[0,0,0], return_frac: bool=False):
        """
        Construct a uniform sampling of k-space of given size.
        2*pi constant is missed.
        
        Args:
            mesh: list or np.array
            shift: list or np.array
        """     
        struct = Structure(lattice=self.graph_data.latt*Hamcts.BOHRtoANG,
                           species=[Element.from_Z(k).symbol for k in self.graph_data.species],
                           coords=self.graph_data.pos*Hamcts.BOHRtoANG, coords_are_cartesian=True)
        positions = struct.frac_coords
        cell = (self.graph_data.latt*Hamcts.BOHRtoANG, positions, self.graph_data.species)
        mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift)
        # Irreducible k-points
        k_grids = grid / np.array(mesh, dtype=float) # (nk, 3)
        if return_frac:
            return k_grids
        else:
            k_vec = np.tensordot(k_grids, self.graph_data.lat_per_inv, axes=1) # (nk, 3)
            return k_vec

    def _frac2car(self, k_grids):
        """2*pi constant is missed.
        Args:
            k_grids (_type_): _description_

        Returns:
            _type_: _description_
        """
        k_grids = k_grids.reshape(-1, 3)
        k_vec = np.tensordot(k_grids, self.graph_data.lat_per_inv, axes=1)
        return k_vec

    def _car2frac(self, k_grids):
        k_grids = k_grids.reshape(-1, 3)
        k_vec = np.tensordot(k_grids, self.graph_data.latt.T, axes=1)
        return k_vec

    def _get_ir_reciprocal_mesh(self, mesh, shift=[0,0,0], auxiliary_info=False, return_frac:bool=False):
        """Calculate the k-point grid and weights in the irreducible Brillouin zone

        Args:
            mesh: list or np.array
            shift: list or np.array

        Returns:
            k_vec : kpoint coordinates in the irreducible zone, unit: Bohr^-1 
            weight: the weight of each irreducible kpoint
        """        
        struct = Structure(lattice=self.graph_data.latt*Hamcts.BOHRtoANG,
                           species=[Element.from_Z(k).symbol for k in self.graph_data.species],
                           coords=self.graph_data.pos*Hamcts.BOHRtoANG, coords_are_cartesian=True)
        positions = struct.frac_coords
        cell = (self.graph_data.latt*Hamcts.BOHRtoANG, positions, self.graph_data.species)
        mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift)
        # A single np.unique call yields the irreducible ids, the multiplicity of each
        # (the k weights) and the full-grid -> irreducible index map.  Doing this with
        # Counter and Python loops over `mapping` becomes the dominant cost once the
        # mesh is large.
        if auxiliary_info:
            ir_ids, grid2ir_idx, counts = np.unique(mapping, return_inverse=True,
                                                    return_counts=True)
            grid2ir_idx = np.asarray(grid2ir_idx).reshape(-1)
        else:
            ir_ids, counts = np.unique(mapping, return_counts=True)
            grid2ir_idx = None
        # `mapping` is as large as the full mesh; release it as soon as it is consumed.
        del mapping
        # Irreducible k-points
        mesh_arr = np.array(mesh, dtype=float)
        ird_grids = grid[ir_ids] / mesh_arr # (nk, 3)
        if not return_frac:
            k_vec = np.tensordot(ird_grids, self.graph_data.lat_per_inv, axes=1) # (nk, 3)
        else:
            k_vec = ird_grids
        # get k weight
        weight = counts / np.sum(counts)
        if auxiliary_info:
            # `grid` is returned as the raw integer mesh (int32, one third the size of a
            # float64 copy).  Converting the whole mesh here would need two extra float64
            # arrays of shape (prod(mesh), 3) on every rank independently.  The caller only
            # ever uses its own slice, so it divides by `mesh` and rotates that slice
            # instead (see mobility_cal).
            return k_vec, weight, grid, grid2ir_idx
        else:
            # grid is only needed for the irreducible subset, which was already taken above.
            del grid
            return k_vec, weight

    def _phonon_cal(self, q_grid, connect_branches:bool=False):
        """Phonon frequencies and eigenvectors on a set of q points.

        With connect_branches set, the branches are relabelled so that one branch
        index follows one physical branch along the list of q points.  The
        diagonalisation orders modes by frequency at each q independently, so at a
        crossing the two branches involved swap indices and a fixed index no longer
        tracks a single branch -- which is what a dispersion plotted per index shows.

        The relabelling matches each q point against the previous one by solving an
        assignment problem.  Modes that continue the same branch have nearly parallel
        eigenvectors, so their overlap carries most of the cost, and the frequency
        difference breaks near-ties inside a degenerate subspace, where the
        eigenvectors themselves are arbitrary.

        The labelling is relative to the first q point, which has nothing to match
        against and keeps the order the diagonalisation gave it.  Where that point
        is degenerate the eigenvectors within each degenerate subspace are arbitrary,
        so the whole path inherits an arbitrary basis: a path is better started from
        an end with little degeneracy than from one with a lot of it.
        """
        freq_grid = []
        phon_vecs = []
        q_grid = q_grid.reshape(-1, 3)
        prev_freq = None
        prev_eigvecs = None
        for iq, q in enumerate(q_grid):
            dynmat = self.phonon.get_dynamical_matrix_at_q(q)
            eigvals, eigvecs = np.linalg.eigh(dynmat)
            eigvecs = eigvecs.T # shape: (nbranches, nbranches)
            
            freq = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real) # shape: (nbranches,)
            # eigen_vec_phon = eigvecs.reshape(-1, natoms, 3) # shape: (nbranches, natoms, 3)

            if connect_branches:
                if prev_freq is not None:
                    order = self._match_phonon_branches(prev_freq, prev_eigvecs,
                                                        freq, eigvecs)
                    freq = freq[order]
                    eigvecs = eigvecs[order]
                prev_freq, prev_eigvecs = freq, eigvecs

            freq_grid.append(freq)
            phon_vecs.append(eigvecs)

        freq_grid = np.stack(freq_grid, axis=0) * Hamcts.PHONOPYtoHARTREE # shape: (nq, nbranches)
        phon_vecs= np.stack(phon_vecs, axis=0) # shape: (nq, nbranches, nbranches)
        
        return freq_grid, phon_vecs

    def _match_phonon_branches(self, prev_freq, prev_vecs, curr_freq, curr_vecs):
        """Permutation of the current modes that continues the previous branches.

        The eigenvector overlap decides.  A symmetry label would additionally rule
        out pairings between modes of different irreducible representations, but
        the little group along a line is smaller than the one at its end points
        and its operations differ, so labels taken at neighbouring q points are
        not comparable without subducing them to a common subgroup.
        """
        from scipy.optimize import linear_sum_assignment

        # Modes continuing one branch stay nearly parallel.
        overlap = np.abs(np.dot(prev_vecs.conjugate(), curr_vecs.T))

        # Solved exactly, by the Hungarian algorithm: a greedy pass would let two
        # branches claim the same partner where several of them meet at once, and
        # the result would no longer be a permutation.

        # Frequency difference, normalised, as a tie-breaker: within a degenerate
        # subspace the eigenvectors are arbitrary and the overlap cannot separate
        # the modes on its own.
        freq_diff = np.abs(prev_freq[:, None] - curr_freq[None, :])
        max_diff = np.max(freq_diff)
        if max_diff <= Hamcts.TENPM5:
            max_diff = 1.0

        cost = (1.0 - overlap) + (freq_diff / max_diff) * 0.1
        _, order = linear_sum_assignment(cost)
        return order

    def _elec_cal(self, k_grid):
        # Calculate the electron wave function
        k_vec = k_grid.reshape(-1, 3)
        SK = build_reciprocal_from_sparseMat(self.graph_data.S_cell, k_vec, self.graph_data.nbr_shift_of_cell)
        
        if self.soc_switch:
            HK = build_reciprocal_from_sparseMat_soc(self.graph_data.H_cell, k_vec, self.graph_data.nbr_shift_of_cell)
            I = np.identity(2, dtype=SK.dtype)
            SK = np.kron(I,SK)
        else:
            HK = build_reciprocal_from_sparseMat(self.graph_data.H_cell, k_vec, self.graph_data.nbr_shift_of_cell)
        
        # diagonalization
        eigen = []
        eigen_vecs = []
        for ik in range(len(k_vec)):
            w, v = eigh(a=HK[ik], b=SK[ik])
            eigen.append(w)
            eigen_vecs.append(v)

        eigen = np.swapaxes(np.array(eigen), 0, 1) # (norbs, nk)
        eigen_vecs = np.array(eigen_vecs) # (nk, norbs, norbs)
        eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)

        lamda = np.einsum('nai, nij, naj -> na', np.conj(eigen_vecs), SK, eigen_vecs).real
        lamda = 1/np.sqrt(lamda) # shape: (nk, norbs)
        eigen_vecs = eigen_vecs*lamda[:,:,None]

        return eigen, eigen_vecs

    def _elec_cal_partial(self, k_grid, band_indices):
        """
        Partial eigendecomposition: compute only eigenvalues/vectors for bands
        in the range [min(band_indices), max(band_indices)] using scipy.linalg.eigh
        with subset_by_index (LAPACK ?hegvx).  The Cholesky factorisation and the
        tridiagonal reduction are unavoidable, but the eigenvalues of the subset are
        located by bisection and only their eigenvectors are back-transformed, which
        is faster than the full spectrum when only a few bands around the band edge
        are needed.

        Args:
            k_grid (np.ndarray): (nk, 3) k vectors.
            band_indices (list or np.ndarray): the band indices to return.

        Returns:
            eigen      : np.ndarray, shape (len(band_indices), nk)
            eigen_vecs : np.ndarray, shape (nk, len(band_indices), norbs)
        """
        k_vec = k_grid.reshape(-1, 3)
        SK = build_reciprocal_from_sparseMat(
            self.graph_data.S_cell, k_vec, self.graph_data.nbr_shift_of_cell)
        if self.soc_switch:
            HK = build_reciprocal_from_sparseMat_soc(
                self.graph_data.H_cell, k_vec, self.graph_data.nbr_shift_of_cell)
            I = np.identity(2, dtype=SK.dtype)
            SK = np.kron(I, SK)
        else:
            HK = build_reciprocal_from_sparseMat(
                self.graph_data.H_cell, k_vec, self.graph_data.nbr_shift_of_cell)

        i_lo = int(min(band_indices))
        i_hi = int(max(band_indices))
        # positions of band_indices within the returned subset [i_lo .. i_hi]
        local_pos = [int(b) - i_lo for b in band_indices]

        eigen_list = []
        eigen_vecs_list = []
        for ik in range(len(k_vec)):
            w, v = eigh(a=HK[ik], b=SK[ik],
                        subset_by_index=[i_lo, i_hi])
            # w: (i_hi-i_lo+1,)   v: (norbs, i_hi-i_lo+1)
            eigen_list.append(w[local_pos])
            eigen_vecs_list.append(v[:, local_pos])   # (norbs, nbands)

        eigen = np.swapaxes(np.array(eigen_list), 0, 1)  # (nbands, nk)
        eigen_vecs = np.array(eigen_vecs_list)           # (nk, norbs, nbands)
        eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)     # (nk, nbands, norbs)

        # Re-normalise  <psi_n | S_k | psi_n>  (same formula as _elec_cal)
        lamda = np.einsum('nai, nij, naj -> na',
                          np.conj(eigen_vecs), SK, eigen_vecs).real
        lamda = 1.0 / np.sqrt(np.abs(lamda) + 1e-30)
        eigen_vecs = eigen_vecs * lamda[:, :, None]

        return eigen, eigen_vecs

    def EPC_cal_path(self, k_fix, q_paths, band_ini, band_fin, do_symm:bool=True,
                     return_freq:bool=False):
        """
        Args:
            k_fix (list or np.ndarray): (3)
            q_paths (list or np.ndarray): (nqs, 3)
            band_ini (int): The band index of initial state, begin from 0.
            band_fin (int): The band index of final state, begin from 0.
            do_symm (bool): If True, do the average over degenerate state.
            return_freq (bool): If True, also return the phonon frequencies the
                coupling was evaluated at, in the same (nq, nbranches) layout.

        Returns:
            epc_all (np.ndarray): # shape:(nq, nbranches) EPC in Hartree.
            freq_grid (np.ndarray): # shape:(nq, nbranches), only if return_freq.
        """
        # calculate the phonon spectrum
        # This walks a q path, so relabel the branches to follow physical ones:
        # otherwise a fixed branch index swaps branches wherever two of them cross.
        freq_grid, phon_vecs = self._phonon_cal(q_paths, connect_branches=True)
        k_fix = self._frac2car(np.array([k_fix]))[0]
        q_paths = self._frac2car(q_paths)
        epc_all = []
        eig_k, wave_k = self._elec_cal(k_fix)
        phase_k = np.exp(2j*np.pi*np.sum(self.nbr_shift_of_cell_sc*k_fix[None,:], axis=-1)) # shape: (ncells,)
            
        for iq, q in enumerate(tqdm(q_paths)):
            # calculate the electronic info
            _, wave_kpq = self._elec_cal(k_fix+q)
                
            # phonon spectrum
            freq = np.abs(freq_grid[iq])
            eigen_vec_phon = phon_vecs[iq].reshape(-1, self.natoms, 3)
            atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
            phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
            # cal epc
            phase_kpq = np.exp(2j*np.pi*np.sum(self.nbr_shift_of_cell_sc*(k_fix+q)[None,:], axis=-1)) # shape: (ncells,)
            wave_coe1 = wave_k[0, band_ini] # shape: (norbs,)
            wave_coe2 = wave_kpq[0, band_fin] # shape: (norbs,)

            tmp1 = np.einsum('m,n -> mn', np.conj(wave_coe2), wave_coe1)
            # calculate epc
            #
            # The sum over cell pairs carries no branch index, so it is taken once per q
            # rather than once per (q, branch): the two cell phases become an outer
            # product that contracts the gradient tensor over both cell axes, and the
            # electronic overlap then reduces what is left to the displaced-atom and
            # Cartesian axes.  Every branch is finally contracted in one einsum.
            _phase_kpq_cut = np.conj(phase_kpq[self.cell_cut_list]).astype(np.complex128)
            _phase_k_cut = phase_k[self.cell_cut_list].astype(np.complex128)
            phase_mat = np.outer(_phase_kpq_cut, _phase_k_cut) # (ncells_cut, ncells_cut)
            grad_ph = np.einsum('AB, ABmnij -> mnij', phase_mat, self.grad_mat)
            epc_elec = np.einsum('mn, mnij -> ij', tmp1, grad_ph) # (natoms, 3)
            factor_all = 1.0 / np.sqrt(2.0 * self.atomic_mass[None, :] * freq[:, None])
            epc_branch = np.einsum('ij, vij, vi -> v', epc_elec, phvec_wap, factor_all)

            for branch_idx in range(int(3*len(self.atomic_mass))):
                epc = epc_branch[branch_idx]

                # Correction of long-range interactions
                if self.apply_correction and (np.linalg.norm(q) < self.q_cut):
                    epc_corr = self._dipole_correction(tmp1, k_fix, q, factor_all[branch_idx],
                                                       phvec_wap[branch_idx])
                else:
                    epc_corr = 0.0
                epc_all.append(epc + epc_corr)

        # shape:(nq, nbranches)
        epc_all = np.array(epc_all).reshape(len(q_paths), int(3*len(self.atomic_mass)))
        if do_symm:
            self._EPC_symmetrize(epc_all, freq_grid, is_path=True)
        if return_freq:
            return epc_all, freq_grid
        return epc_all

    def _EPC_symmetrize(self, epc_all, freqs, is_path:bool=False):
        if is_path:
            print("Warning: Please make sure that |nk> and |mk+q> states have no degeneracy and also the epc must be rael number!")
            nq, nbranches = epc_all.shape
            for iq in range(nq):
                iw = 0
                epc_sym = epc_all[iq]
                for iw in range(nbranches):
                    if freqs[iq, iw] < 0:
                        epc_sym[iw] = 0.0
                        continue
                    g2 = 0.0 # 简并的epc之和
                    n = 0 # 简并的epc的数目
                    for jw in range(nbranches):
                        if abs(freqs[iq, iw]-freqs[iq, jw]) < self.tenpm5:
                            g2 += epc_all[iq, jw] * epc_all[iq, jw]
                            n += 1
                    epc_sym[iw] = np.sqrt(g2/n)
                epc_all[iq, :] = epc_sym[:]
        else:
            nk, nq, nb_left, nb_right, nbranches = epc_all.shape
            for iq in range(nq):
                iw = 0
                while iw < nbranches:
                    g2 = np.zeros_like(epc_all[:, iq, :, :, iw]) # 简并的epc之和
                    n = 0 # 简并的epc的数目
                    g_list = []
                    for jw in range(max(0, iw-3), min(iw+3, nbranches)):
                        if abs(freqs[iq, iw]-freqs[iq, jw]) < 0.0001:
                            g2 += epc_all[:, iq, :, :, jw] * epc_all[:, iq, :, :, jw]
                            n += 1
                        g_list.append(jw)
                    for ig in g_list:
                        epc_all[:, iq, :, :, ig] = np.sqrt(g2/n)
                    iw += len(g_list)

    def _M_cell_prepare(self,):
        M_cell = []
        for i in range(3):
            M_tmp, _, _, _, _ = build_sparse_matrix(self.graph_data.species, self.graph_data.cell_shift, self.nao_max, self.graph_data.Mon[:,:,:,i], 
                                                    self.graph_data.Moff[:,:,:,i], self.graph_data.edge_index, Ham_type=self.Ham_type)
            M_cell.append(M_tmp)
        M_cell = np.stack(M_cell, axis=-1)
        return M_cell

    def _Hv_cell_prepare(self,):
        Hv_cell = np.einsum('nij,nk->nijk', self.graph_data.H_cell, self.graph_data.nbr_shift_of_cell) # (ncells, norbs, norbs, 3)
        return Hv_cell
    
    def _Sv_cell_prepare(self,):
        Sv_cell = np.einsum('nij,nk->nijk', self.graph_data.S_cell, self.graph_data.nbr_shift_of_cell) # (ncells, norbs, norbs, 3)
        return Sv_cell

    def vel_nk_cal_from_M(self, band_indices, k_grid):
        """Calculate the band velocity of the electron

        Args:
            band_indices (list, tuple, np.array): The index of the energy band to be calculated
            k_grid (np.array): shape: (nk, 3)

        Returns:
            elec_vel: # shape: (nk, nbands, 3)
        """
        k_vec = k_grid.reshape(-1, 3)
        
        eigen, eigen_vecs = self._elec_cal(k_vec)
        # shape: (nk, norbs, norbs, 3)
        MK = build_reciprocal_from_sparseMat3(self.graph_data.M_cell, k_vec, self.graph_data.nbr_shift_of_cell)
        wfn = eigen_vecs[:,band_indices,:]
        elec_vel = oe.contract('nij, nik, njkm->nim', np.conj(wfn), wfn, MK).real # shape: (nk, nbands, 3)
        return elec_vel

    def vel_nk_cal_from_HS(self, band_indices, k_grid):
        """Calculate the band velocity of the electron

        Args:
            band_indices (list, tuple, np.array): The index of the energy band to be calculated
            k_grid (np.array): shape: (nk, 3)

        Returns:
            elec_vel: # shape: (nk, nbands, 3)
        """
        k_vec = k_grid.reshape(-1, 3)
        
        eigen, eigen_vecs = self._elec_cal(k_vec)
        # shape: (nk, norbs, norbs, 3)
        HK_v = build_reciprocal_from_sparseMat3(self.graph_data.Hv_cell, k_vec, self.graph_data.nbr_shift_of_cell)
        wfn = eigen_vecs[:,band_indices,:]
        eig = eigen[band_indices,:] # shape: (nbands, nk)
        elec_vel_1 = oe.contract('nij, nik, njkm->nim', np.conj(wfn), wfn, HK_v).real # shape: (nk, nbands, 3)
        del HK_v
        # shape: (nk, norbs, norbs, 3)
        SK_v = build_reciprocal_from_sparseMat3(self.graph_data.Sv_cell, k_vec, self.graph_data.nbr_shift_of_cell)
        elec_vel_coor = oe.contract('nij, nik, in, njkm->nim', np.conj(wfn), wfn, eig, SK_v).real # shape: (nk, nbands, 3)
        elec_vel = elec_vel_1 - elec_vel_coor
        del SK_v
        return elec_vel

    def _P_cell_prepare(self,):
        P_cell = []
        for i in range(3):
            P_tmp, _, _, _, _ = build_sparse_matrix(self.graph_data.species, self.graph_data.cell_shift, self.nao_max, self.graph_data.Pon[:,:,:,i], 
                                                    self.graph_data.Poff[:,:,:,i], self.graph_data.edge_index, Ham_type=self.Ham_type)
            P_cell.append(P_tmp)
        P_cell = np.stack(P_cell, axis=-1)
        return P_cell

    def _get_reciprocal_lattice_vectors(self, n1:int=3, n2:int=3, n3:int=3, exclude_gamma: bool=False):
        X, Y, Z = np.mgrid[-n1:n1+1, -n2:n2+1, -n3:n3+1]
        g_grid = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=-1)
        if exclude_gamma:
            g_grid = np.delete(g_grid, [int((g_grid.shape[0]+1)/2)-1], axis=0)
        g_vec = np.tensordot(g_grid, self.graph_data.lat_per_inv, axes=1) # (ng, 3)
        return g_vec
    
    def _dipole_correction(self, wave_coe_tp:np.ndarray, k_vec:np.ndarray, q_vec:np.ndarray, ph_prefac:np.ndarray, phon_vec:np.ndarray):
        """
        Args:
            wave_coe_tp (np.array): (norbs, norbs)
            k_vec (np.array): (3,)
            q_vec (np.array): (3)
            freq (np.array): (nbranches,)
            phon_vec (np.array): (natoms, 3)

        Returns:
            ret: scalar
        """
        if self.LRC_taylor_order == 0:
            # qG_vec_cart in bohr^{-1}
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q_vec)
            phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k_vec[None,:], axis=-1)) # shape: (ncells, )
            sum_r = np.einsum('n, ij, nij->', phase, wave_coe_tp, self.graph_data.S_cell)
            atomic_phase = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos))  # shape: (natoms, ngs)
            temp1 = np.einsum('gi, kij, kj->kg', qG_vec_cart, self.BECs, phon_vec) # shape: (natoms, ngs)
            temp2 = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term
            temp3 = temp1 * temp2[None,:] * atomic_phase # shape: (natoms, ngs)
            ret = Hamcts.JFOURPI * np.einsum('kg, k->', temp3, ph_prefac) * sum_r / self.volume_uc
        elif self.LRC_taylor_order == 1:
            # qG_vec_cart in bohr^{-1}
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q_vec)
            phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k_vec[None,:], axis=-1)) # shape: (ncells, )
            mat_r = 1.0j*np.einsum('nija, ga->gnij', self.graph_data.P_cell, qG_vec_cart)   # shape: (ngs, ncells, norbs, norbs)
            # mat_exp: shape (ngs, ncells, norbs, norbs)
            mat_exp_taylor = self.graph_data.S_cell[None, :, :] + mat_r
            sum_r = np.einsum('n, ij, gnij->g', phase, wave_coe_tp, mat_exp_taylor) # shape: (ngs, )
            atomic_phase = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos))  # shape: (natoms, ngs)
            temp1 = np.einsum('gi, kij, kj->kg', qG_vec_cart, self.BECs, phon_vec) # shape: (natoms, ngs)
            temp2 = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term
            temp3 = temp1 * temp2[None,:] * atomic_phase # shape: (natoms, ngs)
            temp4 = (temp3*sum_r[None,:]).sum(-1) # shape: (natoms, )
            ret = Hamcts.JFOURPI * (temp4 * ph_prefac).sum() / self.volume_uc
        return ret

    def _dipole_correction_mat(self, ibnd, wave_k, wave_kpq, k_vec, q_vec, factor_all,
                               phvec_wap, freq):
        """
        The dipole correction of one (k, q, ibnd) for every (jbnd, branch), as a
        (nbands, nmodes) matrix.

        This is the fallback used at LRC_taylor_order == 1, where the correction does not
        factorise into a (q, branch)-dependent Ewald sum times an electronic overlap, so the
        cached form used at zeroth order is invalid and every band pair and branch has to go
        through _dipole_correction individually, as before.

        Args:
            ibnd (int): index of the initial band within band_indice.
            wave_k (np.ndarray): (nbands, norbs) eigenvectors at k.
            wave_kpq (np.ndarray): (nbands, norbs) eigenvectors at k+q.
            k_vec (np.ndarray): (3,) k vector.
            q_vec (np.ndarray): (3,) q vector.
            factor_all (np.ndarray): (nmodes, natoms) phonon prefactors.
            phvec_wap (np.ndarray): (nmodes, natoms, 3) phonon eigenvectors with the
                atomic phase already applied.
            freq (np.ndarray): (nmodes,) phonon frequencies.

        Returns:
            np.ndarray: (nbands, nmodes) complex.
        """
        nbands = len(wave_kpq)
        nmodes = len(phvec_wap)
        out = np.zeros((nbands, nmodes), dtype=np.complex128)
        for jbnd in range(nbands):
            tmp1 = np.einsum('m,n -> mn', np.conj(wave_kpq[jbnd]), wave_k[ibnd])
            for imode in range(nmodes):
                out[jbnd, imode] = self._dipole_correction(
                    tmp1, k_vec, q_vec, factor_all[imode], phvec_wap[imode])
        return out

    def _get_LRC_ewald_G(self, q_vec_cart:np.ndarray):
        # move q_vec to 1BZ
        q_vec = self._car2frac(q_vec_cart)
        q_vec = q_vec - np.floor(q_vec)
        # Transform q_vec to cartesian coordinates
        q_vec_cart = self._frac2car(q_vec)[0]
        if np.linalg.norm(q_vec_cart) < Hamcts.TENPM10:
            G_vec_cart = self._get_reciprocal_lattice_vectors(5,5,5, True) # (ngs, 3)
        else:
            G_vec_cart = self._get_reciprocal_lattice_vectors(5,5,5, False) # (ngs, 3)
        qG_vec_cart = q_vec_cart[None, :] + G_vec_cart  # (ngs, 3)
        qG_vec_cart = qG_vec_cart * Hamcts.TWOPI
        exp_inner_term = np.einsum('gi, ij, gj->g', qG_vec_cart, self.DL, qG_vec_cart)   # (ngs, )
        tmp_mask = exp_inner_term < (Hamcts.EWALD_LN_CUTOFF * self.ewald_param)
        exp_inner_term = exp_inner_term[tmp_mask]  # (ngs_in, )
        qG_vec_cart = qG_vec_cart[tmp_mask] # (ngs_in, 3)
        return qG_vec_cart, exp_inner_term

    def _get_ecbm(self, enks, icbm):
        """
        Get the energy of CBM.

        Args:
            enks (np.ndarray): (nbnd, nks) electron energys 
            icbm (int): the band index of CBM
        
        Returns:
            ecbm: The energy of CBM in Hartree.
        """
        return np.min(enks[icbm,:])

    def _get_evbm(self, enks, ivbm):
        """
        Get the energy of VBM.

        Args:
            enks (np.ndarray): (nbnd, nks) electron energys 
            icbm (int): the band index of VBM
        
        Returns:
            evbm: The energy of VBM in Hartree.
        """
        return np.max(enks[ivbm,:])

    # ref from PW sumkg.f90
    def _sumkg(self, enks, degauss, ngauss, ene):
        # Vectorized: (nbnd, nks) broadcast replaces loop over nks
        return np.sum(wgauss((ene - enks) / degauss, ngauss) * self.weight_k[None, :])

    # ref from EPW utilities.f90
    def _get_fermi_level_insulator(self, enks, iband_edge):
        """
        Calculate the fermi energy for a given carrier density, and then recalculate the carrier density through fermi level.

        Args:
            enks (np.ndarray): (nbnd, nks) electron energys 
            iband_edge (int): the first index of band edge in enks
        
        Returns:
            efermi: The fermi energy in Hartree
            carrier_density: The carrier density in a.u.
        """
        efermi = 0.0
        nbnd = enks.shape[0]
        nks = enks.shape[1]
        carrier_density = 0.0
        carrier_small_judge = Hamcts.TENPM80 / (Hamcts.CMtoBOHR ** 3)
        if self.ishole:
            evbm = self._get_evbm(enks, iband_edge)
            if self.rank == 0:
                print("VBM = {} eV".format(evbm * Hamcts.HARTREEtoEV))
            # Vectorized: the exponential is evaluated for the whole (nbnd, nks) array at
            # once.  np.where evaluates both arms, but the discarded one only underflows,
            # so the values kept are the same as those the element-wise loop produced.
            args_hole = (enks - evbm) / self.temperature
            ks_exp = np.where(args_hole < -self.maxarg, 0.0, np.exp(args_hole))
            # ef sets the Fermi level through efermi = evbm - log(ef) * T, so ef below
            # one puts it above the VBM (dilute holes, Fermi level in the gap) and ef
            # above one puts it below (degenerate holes, Fermi level inside the valence
            # bands).  Bracketing at an upper bound of one therefore excludes the
            # degenerate case entirely; bracket symmetrically instead.
            eup = Hamcts.TENPM80
            elw = Hamcts.TENPP80
            for i in range(self.fermi_maxiter):
                ef = np.sqrt(eup) * np.sqrt(elw)
                _kse = ks_exp[:iband_edge+1] * ef
                fnk = np.where(_kse > Hamcts.TENPP60, 0.0, 1.0 / (_kse + 1.0))
                hole_density = np.sum((1.0 - fnk) * self.weight_k[None, :])
                hole_density *= self.inv_cell
                if np.abs(hole_density) < carrier_small_judge:
                    rel_err = -Hamcts.TENPP3
                else:
                    rel_err = (hole_density - np.abs(self.ncarrier)) / hole_density
                # Tested on the magnitude, as the electron branch below does.  The
                # underflow case sets rel_err negative to mean "too few holes, raise ef",
                # and a bare rel_err < TENPM5 would read that as convergence and break on
                # the first iteration, leaving ef at its starting value.
                if np.abs(rel_err) < Hamcts.TENPM5:
                    efermi = evbm - (np.log(ef) * self.temperature)
                    break
                elif rel_err > Hamcts.TENPM5:
                    elw = ef
                else:
                    eup = ef
            fnk = fermi_weight(enks[:iband_edge+1] - efermi, self.temperature)
            carrier_density += np.sum((1.0 - fnk) * self.weight_k[None, :])
        else:
            ecbm = self._get_ecbm(enks, iband_edge)
            if self.rank == 0:
                print("CBM = {} eV".format(ecbm * Hamcts.HARTREEtoEV))
            # Vectorized, as in the hole branch above.
            args_elec = (enks - ecbm) / self.temperature
            ks_expcb = np.where(args_elec > self.maxarg, Hamcts.TENPP200,
                                np.exp(np.minimum(args_elec, self.maxarg)))
            eup = 1.0
            elw = Hamcts.TENPP80
            for i in range(self.fermi_maxiter):
                ef = np.sqrt(eup) * np.sqrt(elw)
                _kse = ks_expcb[iband_edge:] * ef
                fnk = np.where(_kse > Hamcts.TENPP60, 0.0, 1.0 / (_kse + 1.0))
                electron_density = np.sum(fnk * self.weight_k[None, :])
                electron_density *= self.inv_cell
                if np.abs(electron_density) < carrier_small_judge:
                    rel_err = Hamcts.TENPP3
                else:
                    rel_err = (electron_density - np.abs(self.ncarrier)) / electron_density
                if np.abs(rel_err) < Hamcts.TENPM5:
                    efermi = ecbm - (np.log(ef) * self.temperature)
                    break
                elif rel_err > Hamcts.TENPM5:
                    eup = ef
                else:
                    elw = ef
            fnk = fermi_weight(enks[iband_edge:] - efermi, self.temperature)
            carrier_density += np.sum(fnk * self.weight_k[None, :])
        if i == (self.fermi_maxiter - 1):
            raise RuntimeError("The insulator fermi level cannot converge.", '6002')
        return efermi, carrier_density

    # ref from PW efermig.f90
    def _get_fermi_level_metal(self, enks, degauss, ngauss, nelec):
        elw = min(1.0E8, np.min(enks[0, :]))
        eup = max(-1.0E8, np.max(enks[-1, :]))
        eup = eup + 2 * degauss
        elw = elw - 2 * degauss
        sumkup = self._sumkg(enks, degauss, ngauss, eup)
        sumklw = self._sumkg(enks, degauss, ngauss, elw)
        if (sumkup - nelec) < -Hamcts.TENPM10 or (sumklw - nelec) > Hamcts.TENPM10:
            raise RuntimeError("Cannot bracket Ef.", '6002')
        ef = 0.0
        for i in range(self.fermi_maxiter):
            ef = (eup + elw) / 2.0
            sumkmid = self._sumkg(enks, degauss, ngauss, ef)
            if np.abs(sumkmid - nelec) < Hamcts.TENPM10:
                break
            elif (sumkmid - nelec) < -Hamcts.TENPM10:
                elw = ef
            else:
                eup = ef
        if i == (self.fermi_maxiter - 1):
            raise RuntimeError("The metal fermi level cannot converge.", '6002')
        return ef

    def _get_ef_dos(self, enks):
        # Vectorized: broadcast over (nbnd, nks) replaces nested loops
        delta_f3 = w0gauss((enks - self.efermi) * self.inv_smeark, ngauss=1) * self.inv_smeark
        return np.sum(delta_f3 * self.weight_k[None, :])

    def eliashberg_spectrum_cal(self):
        """
        Calculate eliashberg spectral function \alpha^{2} F(\omega).

        Args:
            k_grid (list or np.ndarray): (3,)
            q_grid (list or np.ndarray): (3,)
            bands_indices (np.ndarray): (list or np.ndarray)

        Returns:
            a2f: # shape: ()
        """

        # get the number of included electrons
        if self.has_unpair_electron:
            num_electron_include = len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index]) * 2.0 + 1.0
        else:
            num_electron_include = len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index]) * 2.0
        
        k_grid, self.weight_k = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=False)
        q_grid = self._get_monkhorst_pack(self.q_size, self.graph_data.latt, return_frac=True)

        nmodes = int(3) * self.natoms
        nbands = len(self.bands_indices)

        # initial
        if self.omega_range[0] < Hamcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        a2f = np.zeros(nomegas)

        # consider the spin factor
        self.weight_k *= 2.0
        self.weight_q = 1.0 / len(q_grid)
        
        # q points are parallelized and q grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(len(q_grid)):
            split_sections[i%self.rank_size] += 1
        
        split_sections = np.cumsum(split_sections, axis=0)
        q_grid = np.split(q_grid, indices_or_sections=split_sections, axis=0)
        
        if q_grid[self.rank].size>0:
            # calculate the phonon spectrum in parallel
            q_grid = q_grid[self.rank]
            freq_grid, phon_vecs = self._phonon_cal(q_grid)
            nqs_local = len(q_grid)
            phon_vecs = phon_vecs.reshape(nqs_local, nmodes, self.natoms, 3)
            # change fractional coordinates to cartesian coordinates
            q_grid = self._frac2car(q_grid)
        else:
            nqs_local = 0
            q_grid = np.empty((0, 3))
            freq_grid = np.empty((0, nmodes))
            phon_vecs = np.empty(0, nmodes, self.natoms, 3)
        
        # k grid is split
        if self.rank == 0:
            print('k grid parallel is also switched on!')
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(len(k_grid)):
            split_sections[i%self.rank_size] += 1
        
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_all = np.split(k_grid, indices_or_sections=split_sections, axis=0)
        
        if k_grid_all[self.rank].size > 0:            
            # eigen: (norbs, nk_local) eigen_vecs: (nk_local, norbs, norbs)
            eigen, eigen_vecs = self._elec_cal(k_grid_all[self.rank])
            eigen = eigen[self.bands_indices,:]
            eigen_vecs = eigen_vecs[:, self.bands_indices, :]
        else:
            eigen, eigen_vecs = np.empty((self.norbs, 0)), np.empty((0, self.norbs, self.norbs))
        # gather eigen & eigen_vecs
        eigen_all = self.comm.allgather(eigen)
        eigen_all = np.concatenate(eigen_all, axis=-1) # (norbs, nk)
        eigen_vecs_all = self.comm.allgather(eigen_vecs)
        eigen_vecs_all = np.concatenate(eigen_vecs_all, axis=0) # (nk, norbs, norbs)
        del eigen, eigen_vecs
        
        # get the fermi energy and dos of fermi level
        self.efermi = self._get_fermi_level_metal(eigen_all, self.smeark, self.gauss_type, num_electron_include)
        # N(Ef) in lambda formula is the DOS per spin
        dos_ef = self._get_ef_dos(eigen_all) / 2.0
        if self.rank == 0:
            print("Fermi energy = {} eV\nDOS of Fermi level (spin factor not included) = {}".format(self.efermi * Hamcts.HARTREEtoEV, dos_ef / Hamcts.HARTREEtoEV))
        
        nqs = len(freq_grid)
        epc_strengths = np.zeros((nqs, nmodes))
        if self.rank == 0:
            logger = time_logger(len(q_grid), 'eliashberg_spectrum_cal')
        # calculate electron-phonon coupling strength
        for iq, q in enumerate(q_grid):
            # phonon spectrum
            freq = freq_grid[iq]
            eigen_vec_phon = phon_vecs[iq]
            atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
            phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
            for ik, k in enumerate(k_grid):
                eig_k, wave_k = eigen_all[:,ik], eigen_vecs_all[ik]
                kpq = k+q
                eig_kpq, eigen_vecs_kpq = self._elec_cal(kpq) # (norbs, 1), (1, norbs, norbs)
                eig_kpq = eig_kpq[self.bands_indices, 0]
                wave_kpq = eigen_vecs_kpq[0, self.bands_indices, :] # (norbs, norbs)
                enk_match_table = np.abs(eig_k - self.efermi) < self.e_thr
                emkq_match_table = np.abs(eig_kpq - self.efermi) < self.e_thr
                # cal epc
                phase_k = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                phase_kpq = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)

                for ibnd in range(nbands):
                    if not enk_match_table[ibnd]:
                        continue
                    for jbnd in range(nbands):
                        if not emkq_match_table[jbnd]:
                            continue
                        wave_coe1 = wave_k[ibnd] # shape: (norbs,)
                        wave_coe2 = wave_kpq[jbnd] # shape: (norbs,)
                        tmp1 = np.einsum('m,n -> mn', np.conj(wave_coe2), wave_coe1)
                        for branch_idx in range(nmodes):
                            if freq[branch_idx] < self.phonon_cutoff:
                                continue
                            factor = 1/np.sqrt(2 * self.atomic_mass * abs(freq[branch_idx])) # shape:(natoms,)
                            tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*phvec_wap[branch_idx], tmp1)
                            # calculate epc
                            epc = 0.0
                            for i_m, m in enumerate(self.cell_cut_list): # ncells
                                for i_n, n in enumerate(self.cell_cut_list): # ncells 
                                    epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                            g2_tmp = np.abs(epc) * np.abs(epc)
                            delta_nk = w0gauss((eig_k[ibnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                            delta_mkq = w0gauss((eig_kpq[jbnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                            epc_strengths[iq, branch_idx] = epc_strengths[iq, branch_idx] + g2_tmp * delta_nk * delta_mkq * self.weight_k[ik]
            if self.rank==0:
                logger.step(iq+1)
        # The formula divide by an extra w_qv, but we do not do it here, because it will be eliminated in the a2f calculation.
        epc_strengths = epc_strengths / dos_ef # shape: (nqs[local], nmodes)
        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        a2f = np.einsum('qv, wqv->w', epc_strengths, delta_omegas) / 2.0 * self.weight_q
        # The a2f of the whole q is obtained by allreducing the a2f of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
        return omegas_list, a2f

    def _load_match_table(self, filename:str):
        match_tables = np.load(filename)
        if len(match_tables) > 0:
            unique_ik, unique_inv_ik = np.unique(match_tables[:, 0], return_inverse=True)
            # self._generate_k_uniquek_mapping(unique_ik)
            match_tables[:, 0] = unique_inv_ik
            unique_iq, unique_inv_iq = np.unique(match_tables[:, 1], return_inverse=True)
            match_tables[:, 1] = unique_inv_iq
            unique_ibnd, unique_inv_ibnd = np.unique(match_tables[:, 2], return_inverse=True)
            match_tables[:, 2] = unique_inv_ibnd
            unique_jbnd, unique_inv_jbnd = np.unique(match_tables[:, 3], return_inverse=True)
            match_tables[:, 3] = unique_inv_jbnd
        else:
            match_tables, unique_ik, unique_iq, unique_ibnd, unique_jbnd = None, None, None, None, None
        return match_tables, unique_ik, unique_iq, unique_ibnd, unique_jbnd

    def eliashberg_spectrum_cal_sparse(self):
        """
        Calculate eliashberg spectral function \alpha^{2} F(\omega).

        Args:
            k_grid (list or np.ndarray): (3,)
            q_grid (list or np.ndarray): (3,)
            bands_indices (np.ndarray): (list or np.ndarray)

        Returns:
            a2f: # shape: ()
        """

        # 将grad_mat转为稀疏矩阵

        grad_mat_dict = dict()
        from scipy.sparse import csr_matrix
        for i_m, m in enumerate(self.cell_cut_list):
            for i_n, n in enumerate(self.cell_cut_list):
                for ia in range(self.natoms):
                    for k in range(3):
                        # use numpy.where to filter elements that are smaller than a threshold and then convert to a sparse matrix  
                        threshold = 0.0001
                        grad_mat_small = self.grad_mat[i_m, i_n, :, :, ia, k]
                        filtered_array = np.where(np.abs(grad_mat_small) > threshold, grad_mat_small, 0.0)
                        sparse_matrix = csr_matrix(filtered_array)
                        if sparse_matrix.nnz > 0:
                            grad_mat_dict[(m,n,ia,k)] = (sparse_matrix.data.copy(), sparse_matrix.indices.copy(), sparse_matrix.indptr.copy())
        del self.grad_mat

        # get the number of included electrons
        if self.has_unpair_electron:
            num_electron_include = len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index]) * 2.0 + 1.0
        else:
            num_electron_include = len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index]) * 2.0

        k_grid, self.weight_k = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=False)
        q_grid = self._get_monkhorst_pack(self.q_size, self.graph_data.latt, return_frac=True)

        nmodes = 3 * self.natoms

        # initial
        if self.omega_range[0] < Hamcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        a2f = np.zeros(nomegas)

        # consider the spin factor
        self.weight_k *= 2.0
        self.weight_q = 1.0 / len(q_grid)

        # q points are parallelized and q grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(len(q_grid)):
            split_sections[i%self.rank_size] += 1

        split_sections = np.cumsum(split_sections, axis=0)
        q_grid = np.split(q_grid, indices_or_sections=split_sections, axis=0)

        if q_grid[self.rank].size>0:
            # calculate the phonon spectrum in parallel
            freq_grid, phon_vecs = self._phonon_cal(q_grid[self.rank])
            # change fractional coordinates to cartesian coordinates
            q_grid[self.rank] = self._frac2car(q_grid[self.rank])
        else:
            q_grid[self.rank] = np.empty((0, 3))

        eigen_all, eigen_vecs_all = self._elec_cal(k_grid)

        # get the fermi energy and dos of fermi level
        self.efermi = self._get_fermi_level_metal(eigen_all[self.bands_indices,:], self.smeark, self.gauss_type, num_electron_include)
        # N(Ef) in lambda formula is the DOS per spin
        dos_ef = self._get_ef_dos(eigen_all[self.bands_indices,:]) / 2.0
        if self.rank == 0:
            print("Fermi energy = {} eV\nDOS of Fermi level (spin factor not included) = {}".format(self.efermi * Hamcts.HARTREEtoEV, dos_ef / Hamcts.HARTREEtoEV))

        nqs = len(freq_grid)
        epc_strengths = np.zeros((nqs, nmodes))

        bands_indices = np.array(self.bands_indices, dtype="intc")

        # calculate electron-phonon coupling strength
        for iq, q in enumerate(q_grid[self.rank]):
            # phonon spectrum
            eigen_vec_phon = phon_vecs[iq].reshape(-1, self.natoms, 3)
            atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
            phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
            epc_strengths_q = np.zeros((self.natoms*3,), dtype=float, order='C')

            for ik, k in enumerate(k_grid):

                eig_k, wave_k = eigen_all[:,ik], eigen_vecs_all[ik]
                kpq = k+q
                eig_kpq, eigen_vecs_kpq = self._elec_cal(kpq) # (1, 3), (1, norbs, norbs)
                eig_kpq = eig_kpq[:, 0]
                wave_kpq = eigen_vecs_kpq[0] # (norbs, norbs)
                enk_match_table = np.abs(eig_k[bands_indices] - self.efermi) < self.e_thr
                emkq_match_table = np.abs(eig_kpq[bands_indices] - self.efermi) < self.e_thr

                # cal epc
                phase_k = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                phase_kpq = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                eliashberg_spectrum_cal_helper_sparse(freq_grid[iq].copy(), self.atomic_mass.copy(),phvec_wap.copy(), 
                                                      self.cell_cut_list, phase_k.copy(),phase_kpq.copy(),grad_mat_dict,
                                                      wave_k.copy(),wave_kpq.copy(),
                                                      bands_indices.copy(),bands_indices.copy(),
                                                      enk_match_table.copy(), emkq_match_table.copy(),
                                                      eig_k.copy(),eig_kpq.copy(),epc_strengths_q,
                                                      self.phonon_cutoff, self.inv_smeark, self.efermi, self.weight_k[ik])
            epc_strengths[iq] = epc_strengths_q
        # The formula divide by an extra w_qv, but we do not do it here, because it will be eliminated in the a2f calculation.
        epc_strengths = epc_strengths / dos_ef # shape: (nqs[local], nmodes)

        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        a2f = np.einsum('qv, wqv->w', epc_strengths, delta_omegas) / 2.0 * self.weight_q

        # The a2f of the whole q is obtained by allreducing the a2f of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
        return omegas_list, a2f

    def epc_strength_cal(self, a2f, omegas_list):
        """
        Calculate the electron-phonon coupling strength.

        args:
            a2f: The eliashberg_spectrum.
            omegas_list: A list of omegas corresponding to a2f.

        returns:
            epc_strength: The electron-phonon coupling strength.
        """
        epc_strength = 0.0
        for iomega, omega in enumerate(omegas_list):
            epc_strength = epc_strength + a2f[iomega] / omega
        epc_strength = epc_strength * self.omega_step * 2.0
        return epc_strength

    def logave_freq_cal(self, epc_strength, a2f, omegas_list):
        """
        Calculate the logarithmic average of the phonon frequencies.

        args:
            epc_strength: The lambda.
            a2f: The eliashberg_spectrum.
            omegas_list: A list of omegas corresponding to a2f.

        returns:
            logave_freq: The logarithmic average of the phonon frequencies in Hartree. 
        """
        inner_term = 0.0
        for iomega, omega in enumerate(omegas_list):
            inner_term = inner_term + a2f[iomega] / omega * np.log(omega)
        inner_term = inner_term * 2.0 / epc_strength * self.omega_step

        return np.exp(inner_term)

    def Allen_Dynes_Tc_cal(self, epc_strength:float, miu:np.ndarray, omega_logave:float):
        """
        Calculate the Tc in Allen-Dynes theory.

        args:
            epc_strength: The lambda.
            miu: A list of effective Coulomb potential.
            omega_logave: The logarithmic average of the phonon frequencies

        return:
            Tc: The Tc in Hartree.
        """
        exp_term = np.exp(-1.04 * (1.0 + epc_strength) / (epc_strength - miu * (1.0 + 0.62 * epc_strength)))
        Tc = omega_logave / 1.2 * exp_term 
        return Tc

    def superconductivity_cal(self):
        """
        Calculate for superconductivity.

        Args:

        Returns:
        """

        omegas, a2f = self.eliashberg_spectrum_cal_sparse()
        if self.rank == 0:
            fout = open(os.path.join(self.outdir, "a2f.dat"), 'w')
            for iomega, omega in enumerate(omegas):
                fout.write(f"{str(round(omega * Hamcts.HARTREEtoMEV, 10))}    {str(round(a2f[iomega], 10))}\n")
            fout.close()
        epc_strength = self.epc_strength_cal(a2f, omegas)
        logave_freq = self.logave_freq_cal(epc_strength, a2f, omegas)
        Tc = self.Allen_Dynes_Tc_cal(epc_strength, self.mius, logave_freq)
        if self.rank == 0:
            print(f"Lambda = {epc_strength}")
            print(f"Omega log = {logave_freq * Hamcts.HARTREEtoMEV} meV")
            print(f"  Miu        Tc (K)")
            miu_Tc_str = '\n'.join([f"{np.round(self.mius[imiu], 4)}    {Tc[imiu] * Hamcts.HARTREEtoKELVIN}" for imiu in range(len(self.mius))])
            print(miu_Tc_str)

    # ref from Perturbo pert_utils.f90
    def _get_match_table(self, eigs_k, eigs_kpq, freq):
        """
        Get the match table to speed up function "rate_cal" as Perturbo.

        Args:
            eigs_k (np.ndarray): (nbnd)
            eigs_kpq (np.ndarray): (mbnd)
            freq (np.ndarray): (nmodes)

        Returns:
            match_table: # shape: (nbnd, nbnd, nmodes)
        """
        match_table = np.zeros((len(eigs_k), len(eigs_kpq), len(freq)), dtype=bool)
        for imode, wmode in enumerate(freq):
            if (wmode < self.phonon_cutoff):
                continue
            for ibnd, enk in enumerate(eigs_k):
                for jbnd, emkq in enumerate(eigs_kpq):
                    if np.abs(np.abs(enk - emkq) - wmode) < self.e_thr:
                        match_table[ibnd, jbnd, imode] = True
        return match_table

    def rate_cal(self, k_grid, q_grid, band_indice, ecbm):
        """
        Calculate the scattering rate and the energy loss rate with delta function cutoff.

        Args:
            k_grid (list or np.ndarray): (3,)
            q_grid (list or np.ndarray): (3,)
            bands_indice (int): The only band indice.
            ecbm (float): The energy of CBM.

        Returns:
            rate_all: # shape: (nbands, nk) if eloss == False
        """

        nmodes = int(3) * self.natoms
        ncells = len(self.cell_shift_array_reduced)
        nbands = len(band_indice)
        nks = len(k_grid)
        nqs = len(q_grid)

        # Energy window around the band edge: above the CBM for electrons, below the VBM
        # for holes (ecbm holds the VBM energy when ishole is set).
        if self.ishole:
            efocus_min = ecbm - self.over_vbm
            efocus_max = ecbm
        else:
            efocus_min = -np.inf
            efocus_max = ecbm + self.over_cbm

        rate_all = np.zeros((nbands, nks))

        # ------------------------------------------------------------------
        # Ranks are arranged as n_kgroups k-groups x rank_size / n_kgroups q-groups.  The
        # per-k quantities built below are paid once per k point, so splitting k as well as
        # q divides that part of the work by n_kgroups.  n_kgroups = 1 (the default) leaves
        # the k points unsplit, i.e. the original pure q-parallel layout.
        # ------------------------------------------------------------------
        nk_groups = int(getattr(self, 'n_kgroups', 1))
        if nk_groups < 1 or nk_groups > self.rank_size or (self.rank_size % nk_groups) != 0:
            if self.rank == 0 and nk_groups != 1:
                print('  WARNING: n_kgroups={} does not divide rank_size={}; '
                      'falling back to 1 (pure q-parallel).'.format(nk_groups, self.rank_size), flush=True)
            nk_groups = 1
        nq_groups = self.rank_size // nk_groups
        # Ranks of one k-group are consecutive, so the row collectives below stay on-node.
        k_group_id = self.rank // nq_groups
        q_group_id = self.rank % nq_groups
        comm_k = self.comm.Split(color=q_group_id, key=k_group_id)      # same q, different k
        k_row_comm = self.comm.Split(color=k_group_id, key=q_group_id)  # same k, different q

        # q points are distributed over the q-groups; all ranks of a group share them.
        _q_split = np.zeros(nq_groups, dtype=int)
        for i in range(nqs):
            _q_split[i % nq_groups] += 1
        _q_cumsum = np.cumsum(_q_split)
        _q_start = int(_q_cumsum[q_group_id] - _q_split[q_group_id])
        _q_end = int(_q_cumsum[q_group_id])
        nqs_group = _q_end - _q_start
        q_grid_group = q_grid[_q_start:_q_end]
        weights_q_local = self.weight_q[_q_start:_q_end]
        self.weight_q = None

        # Phonons for this group's q subset: the k-ranks of a q-group need the same
        # frequencies and eigenvectors, so they take a slice each and allgather rather than
        # every rank repeating the whole subset.
        if self.rank == 0:
            print('  2D decomposition: {} k-groups x {} q-groups; {} q per group'.format(
                nk_groups, nq_groups, nqs_group), flush=True)
        if nqs_group > 0:
            _ph_counts = np.zeros(nk_groups, dtype=int)
            for i in range(nqs_group):
                _ph_counts[i % nk_groups] += 1
            _ph_cum = np.cumsum(_ph_counts)
            _ph_lo = int(_ph_cum[k_group_id] - _ph_counts[k_group_id])
            _ph_hi = int(_ph_cum[k_group_id])
            if _ph_hi > _ph_lo:
                freq_local, phon_local = self._phonon_cal(q_grid_group[_ph_lo:_ph_hi])
                phon_local = phon_local.reshape(-1, nmodes, self.natoms, 3)
            else:
                freq_local = np.empty((0, nmodes))
                phon_local = np.empty((0, nmodes, self.natoms, 3))
            freq_grid = np.concatenate(comm_k.allgather(freq_local), axis=0)
            phon_vecs = np.concatenate(comm_k.allgather(phon_local), axis=0)
            # change fractional coordinates to cartesian coordinates
            q_grid_group = self._frac2car(q_grid_group)
        else:
            freq_grid = np.empty((0, nmodes))
            phon_vecs = np.empty((0, nmodes, self.natoms, 3))
            q_grid_group = np.empty((0, 3))

        # Only the band energies are needed globally: every rank must agree on which k
        # points fall outside the energy window.  The wave functions are indexed only
        # at the k points a rank owns, and gathering them everywhere would cost
        # nks * nbands * norbs * 16 B per rank, so they stay local.
        # The k grid is split over the k-ranks; within a column all q-groups need the
        # same k data, so only q_group_id == 0 diagonalises and broadcasts along the row.
        _k_split = np.zeros(nk_groups, dtype=int)
        for i in range(nks):
            _k_split[i % nk_groups] += 1
        _k_split_cum = np.cumsum(_k_split)
        k_start_idx = int(_k_split_cum[k_group_id] - _k_split[k_group_id])
        k_end_idx = int(_k_split_cum[k_group_id])
        k_grid_local = k_grid[k_start_idx:k_end_idx]
        if nk_groups == 1:
            # One k-group means k is not split: k_grid_local is already the whole grid and
            # the gather below would be a no-op, but the q_group_id == 0 guard would still
            # leave a single rank diagonalising while every other rank waits on the
            # broadcast.  Each rank therefore diagonalises for itself, which is the
            # redundant-but-parallel layout the pure q-parallel path expects.
            eigen_local, eigen_vecs_local = self._elec_cal_partial(k_grid_local, band_indice)
            all_eigens = np.ascontiguousarray(eigen_local, dtype=np.float64)
            col0_comm = MPI.COMM_NULL
        else:
            if q_group_id == 0:
                if k_grid_local.size > 0:
                    eigen_local, eigen_vecs_local = self._elec_cal_partial(k_grid_local, band_indice)
                else:
                    eigen_local = np.empty((nbands, 0))
                    eigen_vecs_local = np.empty((0, nbands, self.norbs), dtype=np.complex128)
            else:
                eigen_local, eigen_vecs_local = None, None

            # The band energies are gathered globally: every rank must agree on which k points
            # fall inside the energy window, and eig_k is needed in the main loop.  One
            # representative per k-rank contributes and the assembled array goes back along the
            # row; the payload is only nbands * nks * 8 B.
            col0_color = 0 if q_group_id == 0 else MPI.UNDEFINED
            col0_comm = self.comm.Split(color=col0_color, key=k_group_id)
            if q_group_id == 0:
                all_eigens = np.concatenate(
                    col0_comm.allgather(np.ascontiguousarray(eigen_local, dtype=np.float64)), axis=1)
            else:
                all_eigens = None
            all_eigens = k_row_comm.bcast(all_eigens, root=0)

        # Mark the k points outside the energy window and keep only the active ones.
        rate_all[(all_eigens > efocus_max) | (all_eigens < efocus_min)] = np.inf
        active_k_mask = np.any((all_eigens <= efocus_max) & (all_eigens >= efocus_min), axis=0)
        active_k_indices = np.where(active_k_mask)[0]
        n_active = len(active_k_indices)
        if self.rank == 0:
            print('  active k points: {} / {} ({:.2f}%)'.format(n_active, nks, 100.0*n_active/max(nks,1)), flush=True)

        # Only the active k points enter the main loop, so replicating their wave functions
        # is cheap and lets the *active* points be distributed below.  That matters for load
        # balance: the energy window usually selects a clustered region, so slicing the raw
        # mesh would leave most ranks idle.
        _mine_in_slice = active_k_indices[(active_k_indices >= k_start_idx) & (active_k_indices < k_end_idx)]
        if nk_groups == 1:
            # Every rank already holds the eigenvectors of the whole grid, so the active
            # subset is a plain slice; no gather or broadcast is involved.
            active_waves = np.ascontiguousarray(
                eigen_vecs_local[_mine_in_slice - k_start_idx], dtype=np.complex128)
        else:
            if q_group_id == 0:
                active_waves = np.concatenate(col0_comm.allgather(np.ascontiguousarray(
                    eigen_vecs_local[_mine_in_slice - k_start_idx], dtype=np.complex128)), axis=0)
            else:
                active_waves = None
            active_waves = k_row_comm.bcast(active_waves, root=0)
        if col0_comm != MPI.COMM_NULL:
            col0_comm.Free()

        # Distribute the active k points over the k-groups so every rank gets the same
        # number of them no matter how they are spread over the mesh.  active_waves is
        # assembled in ascending k order, so active_waves[_k_lo + i] is the wave function
        # of my_active_indices[i].
        _k_counts = np.zeros(nk_groups, dtype=int)
        for i in range(n_active):
            _k_counts[i % nk_groups] += 1
        _k_cum = np.cumsum(_k_counts)
        _k_lo = int(_k_cum[k_group_id] - _k_counts[k_group_id])
        _k_hi = int(_k_cum[k_group_id])
        my_active_indices = active_k_indices[_k_lo:_k_hi]

        if self.rank == 0:
            logger = time_logger(total_cycles=100, routine_name='rate_cal', line_per_step=True)
        # Progress is reported against the (k, q) pairs this rank owns, so it advances
        # evenly no matter whether the k or the q dimension dominates.
        _work_total = max(len(my_active_indices) * max(nqs_group, 1), 1)
        _work_done = 0
        _decile_done = 0

        # Pre-compute the q-dependent prefactor of the dipole correction; see the comment in
        # rate_cal_polar.  A_qm only depends on (q, imode), so it is hoisted out of the k loop.
        # Unlike the polar/rmp branches, which require the correction, this one also has to
        # honour apply_correction: without it the coupling is short-range only.
        corr_mask = (self.apply_correction & (np.linalg.norm(q_grid_group, axis=-1) < self.q_cut)) if nqs_group else np.empty(0, dtype=bool)
        # A_qm depends only on (q, imode) and every rank of a q-group would otherwise build
        # the same table, so the q are shared out over the group and the result allgathered
        # -- the same pattern as the phonon spectrum above.
        _aq_counts = np.zeros(nk_groups, dtype=int)
        for i in range(nqs_group):
            _aq_counts[i % nk_groups] += 1
        _aq_cum = np.cumsum(_aq_counts)
        _aq_lo = int(_aq_cum[k_group_id] - _aq_counts[k_group_id])
        _aq_hi = int(_aq_cum[k_group_id])
        _A_local = np.zeros((_aq_hi - _aq_lo, nmodes), dtype=complex)
        for _aj in range(_aq_hi - _aq_lo):
            iq = _aq_lo + _aj
            if not corr_mask[iq]:
                continue
            q = q_grid_group[iq]
            freq = freq_grid[iq]
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q)
            # Quantities shared by every branch.
            atomic_phase_G = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos))
            temp2 = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term
            # Branches below the phonon cutoff are gated out by _get_match_table, so their
            # prefactor stays zero; the rest are contracted in one einsum over the branch axis.
            valid_modes = freq >= self.phonon_cutoff
            A_qm = np.zeros(nmodes, dtype=complex)
            if np.any(valid_modes):
                atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1))
                phvec_wap_iq = atomic_phase[None,:,None] * phon_vecs[iq]
                phvec_valid = phvec_wap_iq[valid_modes]
                freq_valid = freq[valid_modes]
                temp1_all = np.einsum('gi, kij, mkj->mkg', qG_vec_cart, self.BECs, phvec_valid)
                temp3_all = temp1_all * temp2[None,None,:] * atomic_phase_G[None,:,:]
                factor_all = 1.0 / np.sqrt(2.0 * self.atomic_mass[None,:] * np.abs(freq_valid[:,None]))
                A_qm[valid_modes] = Hamcts.JFOURPI * np.einsum('mkg, mk->m', temp3_all, factor_all) / self.volume_uc
            _A_local[_aj] = A_qm
        A_qm_cache = (np.concatenate(comm_k.allgather(_A_local), axis=0)
                      if nqs_group else np.zeros((0, nmodes), dtype=complex))

        # Only the cells that survive the cell cut ever enter the phase factors.
        nbr_shift_cut = self.nbr_shift_of_cell_sc[self.cell_cut_list]   # (ncells_cut, 3)
        # _elec_cal_partial materialises H(k) and S(k) for the whole batch, i.e.
        # nk * norbs^2 * 32 B.  Cap the batch so that stays bounded whatever the
        # basis size: a small basis then runs unbatched in practice, while a large
        # one is split rather than allocating the whole batch at once.
        _q_batch = max(1, int(_BATCH_BYTES / (32.0 * self.norbs ** 2)))
        # Main loop: this rank's own active k points outer, its q subset inner.
        for _i_loc, ik_all in enumerate(my_active_indices):
                k = k_grid[ik_all]
                eig_k = all_eigens[:, ik_all]
                wave_k = active_waves[_k_lo + _i_loc]
                skip_mask = (eig_k > efocus_max) | (eig_k < efocus_min)
                if skip_mask.all():
                    continue
                phase_k = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                # grad_mat is indexed [A, B, m, n, i, j] with A <- conj(phase_kpq),
                # B <- phase_k, m <- conj(wave_kpq), n <- wave_k.  Folding phase_k and
                # wave_k in once per k point replaces the loop over cell pairs for every
                # (q, band pair, branch), and drops the second orbital axis.
                # B is contracted before n: B leads grad_mat[A], so (B, m*n*i*j) is a
                # contiguous view and the product never transposes the slice.  Taking n
                # first would, since axis 2 is not trailing.
                _pk_cut = phase_k[self.cell_cut_list].astype(np.complex128)
                _nB = len(self.cell_cut_list)
                grad_contracted_k = np.empty((_nB, self.norbs,
                                              self.natoms, 3, nbands), dtype=np.complex128)
                _gm_is_complex = np.iscomplexobj(self.grad_mat)
                # Real grad_mat (no SOC): split phase_k into its real and imaginary rows so a
                # single GEMM covers both and grad_mat is read only once.
                _pk_ri = None if _gm_is_complex else np.stack([_pk_cut.real, _pk_cut.imag])
                _wkT = np.ascontiguousarray(wave_k.T)             # (norbs, nbands)
                for _iA in range(_nB):
                    _gA = self.grad_mat[_iA].reshape(_nB, -1)     # (B, m*n*i*j) view
                    if _gm_is_complex:
                        _tB = _pk_cut @ _gA
                    else:
                        _ri = _pk_ri @ _gA
                        _tB = _ri[0] + 1j*_ri[1]
                    _tB = _tB.reshape(self.norbs, self.norbs, self.natoms*3) # (m, n, i*j)
                    # (m, i*j, n) @ (n, nbands) -> (m, i*j, nbands), all bands in one call
                    grad_contracted_k[_iA] = np.matmul(
                        _tB.transpose(0, 2, 1), _wkT).reshape(
                            self.norbs, self.natoms, 3, nbands)
                    del _tB
                # S(k) contracted over cells, for the dipole correction.
                phase_k_uc = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k[None,:], axis=-1))
                SK_k = np.einsum('n, nij->ij', phase_k_uc, self.graph_data.S_cell)
                # S(k) enters only through conj(wave_kpq) . S(k) . wave_k^T and both depend
                # on k alone, so fold wave_k in here rather than per band pair.
                _Sw_k = SK_k @ wave_k.T # (norbs, nbands)
                kpq_all = k + q_grid_group                            # (nqs_group, 3)
                # k+q states and cell phases are evaluated in batches instead of one q at a time:
                # the diagonalisation is batched over the chunk, only the bands in band_indice are
                # requested, and the cell phase is built in one call over the cells that survive the
                # cell cut.  A chunk rather than the whole q subset keeps the batch arrays bounded.
                for _q_start in range(0, nqs_group, _q_batch):
                    _q_end = min(_q_start + _q_batch, nqs_group)
                    eig_kpq_b, wave_kpq_b = self._elec_cal_partial(kpq_all[_q_start:_q_end], band_indice)
                    phase_kpq_cut_b = np.exp(Hamcts.JTWOPI*np.einsum('qd, nd->qn',
                                             kpq_all[_q_start:_q_end], nbr_shift_cut)).astype(np.complex128)
                    for _iq_b in range(_q_end - _q_start):
                        iq = _q_start + _iq_b
                        q = q_grid_group[iq]
                        apply_correction_for_this_q = corr_mask[iq]
                        # phonon spectrum
                        freq = freq_grid[iq]
                        eigen_vec_phon = phon_vecs[iq]
                        atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                        phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                        bose_qvs = bose_weight(freq, self.temperature)
                        # calculate the electronic info for k+q
                        eig_kpq = eig_kpq_b[:, _iq_b]
                        wave_kpq = wave_kpq_b[_iq_b]
                        match_table = self._get_match_table(eig_k, eig_kpq, freq)
                        # Every contribution below is gated by the match table, so when it
                        # selects nothing for this (k, q) the cell contraction and the
                        # band/branch contractions are skipped altogether.  The energy
                        # window makes this the common case.
                        if not np.any(match_table[~skip_mask]):
                            continue
                        fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                        # cal epc
                        grad_ph = np.tensordot(np.conj(phase_kpq_cut_b[_iq_b]),
                                              grad_contracted_k, axes=([0], [0])) # (m,i,j,nbands)
                        # Built for every mode at once, so the branches the match table
                        # discards are formed too; leaving 1/sqrt(0) in them would put an
                        # inf where the per-element code never evaluates anything.
                        factor_all = np.zeros((len(freq), len(self.atomic_mass)))
                        _fv = freq >= self.phonon_cutoff
                        factor_all[_fv] = 1.0 / np.sqrt(
                            2.0 * self.atomic_mass[None, :] * np.abs(freq[_fv, None]))
                        if apply_correction_for_this_q and self._lrc_cacheable:
                            sum_r_all = np.conj(wave_kpq) @ _Sw_k # (nbands, nbands)
                            A_qm = A_qm_cache[iq]
                        for ibnd in range(nbands):
                            if skip_mask[ibnd]:
                                continue
                            # Tested before the contractions below rather than after: the
                            # energy window makes an empty row the common case.
                            mt = match_table[ibnd] # (nbands, nmodes)
                            if not np.any(mt):
                                continue
                            grad_partial = grad_ph[..., ibnd] # (norbs, natoms, 3); wave_k already folded in
                            # grad_elec for ALL jbnd at once: (nbands, natoms, 3)
                            grad_elec_all = np.einsum('jm, mab -> jab', np.conj(wave_kpq), grad_partial)
                            # epc for all (jbnd, imode): (nbands, nmodes)
                            epc_mat = np.einsum('jax, max, ma -> jm', grad_elec_all, phvec_wap, factor_all)
                            if apply_correction_for_this_q:
                                # The long-range dipole term is added in, so g2 is the full
                                # |g_short + g_corr|^2 and can never come out negative.
                                if self._lrc_cacheable:
                                    epc_mat = epc_mat + A_qm[None, :] * sum_r_all[:, ibnd, None]
                                else:
                                    epc_mat = epc_mat + self._dipole_correction_mat(
                                        ibnd, wave_k, wave_kpq, k, q, factor_all, phvec_wap, freq)
                            g2_mat = np.abs(epc_mat) ** 2 # (nbands, nmodes)
                            de = eig_k[ibnd] - eig_kpq # (nbands,)
                            d1 = w0gauss((de[:, None] + freq[None, :]) * self.inv_smearq) * self.inv_smearq
                            d2 = w0gauss((de[:, None] - freq[None, :]) * self.inv_smearq) * self.inv_smearq
                            bose = bose_qvs[None, :]    # (1, nmodes)
                            fermi = fermi_kpqs[:, None] # (nbands, 1)
                            if self.ishole:
                                # For holes the roles of the emission and absorption
                                # occupations swap.
                                contrib = g2_mat * mt * ((bose + 1.0 - fermi) * d1 +
                                                         (bose + fermi) * d2)
                            else:
                                contrib = g2_mat * mt * ((bose + fermi) * d1 +
                                                         (bose + 1.0 - fermi) * d2)
                            rate_all[ibnd, ik_all] += weights_q_local[iq] * contrib.sum()
                    # Counted per q batch rather than per k point: a rank owns only a few
                    # active k, so a per-k update could not resolve single percent steps.
                    _work_done += _q_end - _q_start
                    _decile = int(100 * _work_done / _work_total)
                    if self.rank == 0 and _decile > _decile_done:
                        _decile_done = _decile
                        logger.step(_decile)

        comm_k.Free()
        k_row_comm.Free()
        rate_all *= Hamcts.TWOPI
        # Each rank accumulated the contribution of its own q subset; the Allreduce(SUM)
        # completes the q integral.  It moves only a few
        # hundred kB but cannot finish until every rank has arrived, so an imbalanced
        # workload appears as a long silence after the progress bar reaches 100% -- that bar
        # tracks rank 0 alone.  The first message without the second therefore means the run
        # is still waiting for the slowest rank, not that it has deadlocked.
        if self.comm is not None:
            if self.rank == 0:
                print('  waiting for all ranks at the rate_all reduction...', flush=True)
            _t_wait = time.time()
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
            if self.rank == 0:
                print('  reduction done after waiting {:.0f}s for the slowest rank.'.format(
                    time.time() - _t_wait), flush=True)
        return rate_all

    def rate_cal_MRTA(self, k_grid, q_grid, bands_indices, ecbm, is_mrta=False):
        """
        Calculate the scattering rate and the energy loss rate with delta function cutoff.

        Args:
            k_grid (list or np.ndarray): (3,)
            q_grid (list or np.ndarray): (3,)
            bands_indice (list[int]): The only band indice.
            ecbm (float): The energy of CBM.
            is_mrta (bool): If True, using MRTA instead of SERTA.
 
        Returns:
            rate_all: # shape: (nbands, nk) if eloss == False
        """
        
        nmodes = int(3) * self.natoms
        ncells = len(self.cell_shift_array_reduced)
        nbands = len(bands_indices)
        nks = len(k_grid)
        nqs = len(q_grid)

        # get cbm plus over_cbm to obtain the energy range we focus on
        efocus_max = ecbm + self.over_cbm

        rate_all = np.zeros((nbands, nks))

        # q points are parallelized and q grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nqs):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        q_grid = np.split(q_grid, indices_or_sections=split_sections, axis=0)
        weights_q = np.split(self.weight_q, indices_or_sections=split_sections, axis=0)
        self.weight_q = None
        if q_grid[self.rank].size>0:
            # calculate the phonon spectrum in parallel
            freq_grid, phon_vecs = self._phonon_cal(q_grid[self.rank])
            nqs_local = len(freq_grid)
            phon_vecs = phon_vecs.reshape(nqs_local, nmodes, self.natoms, 3)
            weights_q_local = weights_q[self.rank]
            del weights_q
            # change fractional coordinates to cartesian coordinates
            q_grid[self.rank] = self._frac2car(q_grid[self.rank])
        else:
            q_grid[self.rank] = np.empty((0, 3))
        
        # k grid is split
        if self.rank == 0:
            print('k grid parallel is also switched on!')
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nks):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_all = np.split(k_grid, indices_or_sections=split_sections, axis=0)
        if k_grid_all[self.rank].size > 0:            
            # eigen: (norbs, nk_local) eigen_vecs: (nk_local, norbs, norbs)
            eigen, eigen_vecs = self._elec_cal(k_grid_all[self.rank])
            eigen = eigen[bands_indices, :]
            eigen_vecs = eigen_vecs[:, bands_indices, :]
        else:
            eigen, eigen_vecs = np.empty((nbands, 0)), np.empty((0, nbands, self.norbs))

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='rate_cal_test')

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
        
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik, :]
                if is_mrta:
                    v_nk = self.vel_nk_cal_from_HS(bands_indices, [k])
                for ibnd in range(nbands):
                    if eig_k[ibnd] > efocus_max:
                        rate_all[ibnd, ik_all] = np.inf
                        continue
                    phase_k = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)

                    for iq, q in enumerate(q_grid[self.rank]):
                        apply_correction_for_this_q = self.apply_correction and (np.linalg.norm(q) < self.q_cut)
                        # phonon spectrum
                        freq = freq_grid[iq]
                        eigen_vec_phon = phon_vecs[iq]
                        atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                        phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                        bose_qvs = bose_weight(freq, self.temperature)
                        kpq = k + q
                        if is_mrta:
                            v_mkq = self.vel_nk_cal_from_HS(bands_indices, [kpq])
                        # calculate the electronic info for k+q
                        eig_kpq, wave_kpq = self._elec_cal(kpq)
                        eig_kpq = eig_kpq[bands_indices, 0]
                        wave_kpq = wave_kpq[0, bands_indices, :]
                        match_table = self._get_match_table(eig_k, eig_kpq, freq)
                        fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                        # cal epc
                        phase_kpq = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                        for jbnd in range(nbands):
                            tmp1 = np.einsum('m,n -> mn', np.conj(wave_kpq[jbnd]), wave_k[ibnd])
                            # calculate epc
                            for branch_idx in range(nmodes):
                                if match_table[ibnd, jbnd, branch_idx]:
                                    factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * abs(freq[branch_idx])) # shape:(natoms,)
                                    tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*phvec_wap[branch_idx], tmp1)
                                    epc = 0.0
                                    for i_m, m in enumerate(self.cell_cut_list): # ncells
                                        for i_n, n in enumerate(self.cell_cut_list): # ncells  
                                            epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                                    
                                    # Correction of long-range interactions
                                    if apply_correction_for_this_q:
                                        epc_corr = self._dipole_correction(tmp1, k, q, factor, phvec_wap[branch_idx])
                                    else:
                                        epc_corr = 0.0
                                    epc = epc + epc_corr
                                    delta_f1 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] + freq[branch_idx]) * self.inv_smearq) * self.inv_smearq
                                    delta_f2 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] - freq[branch_idx]) * self.inv_smearq) * self.inv_smearq
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                    if is_mrta:
                                        rate_all[ibnd,ik_all] += weights_q_local[iq] * g2_tmp * \
                                                                 ((bose_qvs[branch_idx] + fermi_kpqs[jbnd]) * delta_f1 + \
                                                                  (bose_qvs[branch_idx] + 1.0 - fermi_kpqs[jbnd]) * delta_f2) * \
                                                                 (1.0 - np.dot(v_nk[ibnd, ik], v_mkq[jbnd, 0]) / np.dot(v_nk[ibnd, ik], v_nk[ibnd, ik]))
                                    else:
                                        rate_all[ibnd,ik_all] += weights_q_local[iq] * g2_tmp * \
                                                                ((bose_qvs[branch_idx] + fermi_kpqs[jbnd]) * delta_f1 + \
                                                                (bose_qvs[branch_idx] + 1.0 - fermi_kpqs[jbnd]) * delta_f2)
            if self.rank == 0:
                logger.step(send_rank+1)
        rate_all *= Hamcts.TWOPI
        # The rate_all of the whole q is obtained by allreducing the rate_all of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
        return rate_all

    def rate_cal_save_memory(self, k_grid, q_grid, band_indice, ecbm):
        """
        Calculate the scattering rate and the energy loss rate with delta function cutoff.
        Crazy mode with only one band.

        Args:
            k_grid (list or np.ndarray): (3,)
            q_grid (list or np.ndarray): (3,)
            bands_indice (int): The only band indice.
            ecbm (float): The energy of CBM.

        Returns:
            rate_all: # shape: (nbands, nk) if eloss == False
        """

        nmodes = int(3) * self.natoms
        ncells = len(self.cell_shift_array_reduced)
        nbands = len(band_indice)
        nks = len(k_grid)
        nqs = len(q_grid)

        # get cbm plus over_cbm to obtain the energy range we focus on
        efocus_max = ecbm + self.over_cbm

        rate_all = np.zeros((nbands, nks))

        # q points are parallelized and q grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nqs):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        q_grid = np.split(q_grid, indices_or_sections=split_sections, axis=0)
        weights_q = np.split(self.weight_q, indices_or_sections=split_sections, axis=0)
        self.weight_q = None
        if q_grid[self.rank].size>0:
            # calculate the phonon spectrum in parallel
            freq_grid, phon_vecs = self._phonon_cal(q_grid[self.rank])
            nqs_local = len(freq_grid)
            phon_vecs = phon_vecs.reshape(nqs_local, nmodes, self.natoms, 3)
            weights_q_local = weights_q[self.rank]
            del weights_q
            # change fractional coordinates to cartesian coordinates
            q_grid[self.rank] = self._frac2car(q_grid[self.rank])
        else:
            q_grid[self.rank] = np.empty((0, 3))

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='rate_cal')

        for ik, k in enumerate(k_grid):
            eig_k, wave_k = self._elec_cal(k)
            eig_k, wave_k = eig_k[:, 0], wave_k[0]
            for ibnd in range(nbands):
                if eig_k[ibnd] > efocus_max:
                    rate_all[ibnd, ik] = np.inf
                    continue
                phase_k = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)

                for iq, q in enumerate(q_grid[self.rank]):
                    apply_correction_for_this_q = self.apply_correction and (np.linalg.norm(q) < self.q_cut)
                    # phonon spectrum
                    freq = freq_grid[iq]
                    eigen_vec_phon = phon_vecs[iq]
                    atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                    phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                    bose_qvs = bose_weight(freq, self.temperature)

                    # calculate the electronic info for k+q
                    kpq = k + q
                    eig_kpq, wave_kpq = self._elec_cal(kpq)
                    eig_kpq = eig_kpq[band_indice, 0]
                    wave_kpq = wave_kpq[0, band_indice, :]
                    match_table = self._get_match_table(eig_k, eig_kpq, freq)
                    fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                    # cal epc
                    phase_kpq = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    for jbnd in range(nbands):
                        tmp1 = np.einsum('m,n -> mn', np.conj(wave_kpq[jbnd]), wave_k[ibnd])
                        # calculate epc
                        for branch_idx in range(nmodes):
                            if match_table[ibnd, jbnd, branch_idx]:
                                factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * abs(freq_grid[iq, branch_idx])) # shape:(natoms,)
                                tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*phvec_wap[branch_idx], tmp1)
                                
                                epc = 0.0
                                for i_m, m in enumerate(self.cell_cut_list): # ncells
                                    for i_n, n in enumerate(self.cell_cut_list): # ncells  
                                        epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                                
                                # Correction of long-range interactions
                                if apply_correction_for_this_q:
                                    epc_corr = self._dipole_correction(tmp1, k, q, factor, phvec_wap[branch_idx])
                                else:
                                    epc_corr = 0.0
                                epc = epc + epc_corr
                                
                                delta_f1 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] + freq[branch_idx]) * self.inv_smearq) * self.inv_smearq
                                delta_f2 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] - freq[branch_idx]) * self.inv_smearq) * self.inv_smearq
                                g2_tmp = np.abs(epc) * np.abs(epc)
                                rate_all[ibnd,ik] += g2_tmp * ((bose_qvs[branch_idx] + fermi_kpqs[jbnd]) * delta_f1 + 
                                                                (bose_qvs[branch_idx] + 1.0 - fermi_kpqs[jbnd]) * delta_f2) * weights_q_local[iq] 
            if self.rank == 0:
                logger.step(k+1)
        rate_all *= Hamcts.TWOPI
        # The rate_all of the whole q is obtained by allreducing the rate_all of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
        return rate_all
    
    def rate_cal_polar(self, k_grid, q_grid, band_indice, ecbm):
        """
        Calculate the scattering rate for LRC part only, |nk> within ecut.
        The apply_correction must be True.

        Args:
            k_grid (np.ndarray): The k vectors. # shape: (nks, 3)
            q_grid (np.ndarray): The q vectors. # shape: (nqs, 3)
            bands_indice (int): The only band indice.
            ecbm (float): The energy of CBM.

        Returns:
            rate_all: # shape: (nbands, nk) if eloss == False
        """
        nmodes = int(3) * self.natoms
        ncells = len(self.cell_shift_array_reduced)
        nbands = len(band_indice)
        nks = len(k_grid)
        nqs = len(q_grid)

        # Energy window around the band edge: above the CBM for electrons, below the VBM
        # for holes (ecbm holds the VBM energy when ishole is set).
        if self.ishole:
            efocus_min = ecbm - self.over_vbm
            efocus_max = ecbm
        else:
            efocus_min = -np.inf
            efocus_max = ecbm + self.over_cbm

        rate_all = np.zeros((nbands, nks))

        # The q points are shared out over the ranks and every rank keeps all the active k
        # points, so q is the only dimension that is split and the reduction at the end is a
        # single sum.  With q on the outer loop the per-q work -- the phonon spectrum and the
        # dipole prefactor A_qm -- is amortised over every active k point at once, so it is
        # paid as rarely as possible.
        _q_split = np.zeros(self.rank_size, dtype=int)
        for i in range(nqs):
            _q_split[i % self.rank_size] += 1
        _q_cumsum = np.cumsum(_q_split)
        _q_start = int(_q_cumsum[self.rank] - _q_split[self.rank])
        _q_end = int(_q_cumsum[self.rank])
        nqs_group = _q_end - _q_start
        q_grid_group = q_grid[_q_start:_q_end]
        weights_q_local = self.weight_q[_q_start:_q_end]
        self.weight_q = None

        # Phonon frequencies and eigenvectors for this rank's own q subset.
        if nqs_group > 0:
            freq_grid, phon_vecs = self._phonon_cal(q_grid_group)
            phon_vecs = phon_vecs.reshape(-1, nmodes, self.natoms, 3)
            # change fractional coordinates to cartesian coordinates
            q_grid_group = self._frac2car(q_grid_group)
        else:
            freq_grid = np.empty((0, nmodes))
            phon_vecs = np.empty((0, nmodes, self.natoms, 3))
            q_grid_group = np.empty((0, 3))

        # Band energies and wave functions on the k grid.  The mesh is split over the ranks
        # so that no rank diagonalises more than its share.  The band energies are then
        # gathered globally, because every rank has to agree on which k points fall inside
        # the energy window and needs eig_k in the main loop; the payload is only
        # nbands * nks * 8 B.  The wave functions are gathered for the *active* k points
        # alone: the full set would waste nks * nbands * norbs * 16 B on every rank -- tens
        # of GB per node at high rank counts, and large enough to overflow mpi4py's pickle
        # protocol -- whereas the active subset costs n_active * nbands * norbs * 16 B.
        _k_split = np.zeros(self.rank_size, dtype=int)
        for i in range(nks):
            _k_split[i % self.rank_size] += 1
        _k_split_cum = np.cumsum(_k_split)
        k_start_idx = int(_k_split_cum[self.rank] - _k_split[self.rank])
        k_end_idx = int(_k_split_cum[self.rank])
        k_grid_local = k_grid[k_start_idx:k_end_idx]
        if k_grid_local.size > 0:
            eigen_local, eigen_vecs_local = self._elec_cal_partial(k_grid_local, band_indice)
        else:
            eigen_local = np.empty((nbands, 0))
            eigen_vecs_local = np.empty((0, nbands, self.norbs), dtype=np.complex128)
        all_eigens = np.concatenate(
            self.comm.allgather(np.ascontiguousarray(eigen_local, dtype=np.float64)), axis=1) # (nbands, nks)

        # Mark the k points outside the energy window and keep only the active ones.
        rate_all[(all_eigens > efocus_max) | (all_eigens < efocus_min)] = np.inf
        active_k_mask = np.any((all_eigens <= efocus_max) & (all_eigens >= efocus_min), axis=0)
        active_k_indices = np.where(active_k_mask)[0]
        n_active = len(active_k_indices)
        if self.rank == 0:
            print('  active k points: {} / {} ({:.2f}%)'.format(n_active, nks, 100.0*n_active/max(nks,1)), flush=True)

        # Assembled in ascending k order, so active_waves[i] belongs to active_k_indices[i].
        _mine_in_slice = active_k_indices[(active_k_indices >= k_start_idx) & (active_k_indices < k_end_idx)]
        active_waves = np.concatenate(self.comm.allgather(np.ascontiguousarray(
            eigen_vecs_local[_mine_in_slice - k_start_idx], dtype=np.complex128)), axis=0) # (n_active, nbands, norbs)

        # Every rank walks all the active k points; only the q integral is split, so
        # active_waves[i] is the wave function of my_active_indices[i].
        my_active_indices = active_k_indices

        # Pre-compute the q-dependent prefactor of the dipole correction.  For
        # LRC_taylor_order == 0 the correction factorises exactly as
        #     epc(k, ibnd, jbnd, q, imode) = A_qm(q, imode) * sum_r(k, ibnd, jbnd)
        # because _dipole_correction multiplies a purely (q, imode)-dependent Ewald sum by
        # sum_r, which only involves the overlap matrix and the two wave functions.  A_qm
        # therefore only needs to be evaluated once per (q, imode) instead of once per
        # (k, ibnd, jbnd, q, imode).
        q_active = np.flatnonzero(np.linalg.norm(q_grid_group, axis=-1) < self.q_cut) if nqs_group else np.empty(0, dtype=int)
        A_qm_cache = {}
        for iq in q_active:
            q = q_grid_group[iq]
            freq = freq_grid[iq]
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q)
            # Quantities shared by every branch.
            atomic_phase_G = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos)) # (natoms, ngs)
            temp2 = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term # (ngs,)
            # Branches below the phonon cutoff are switched off by _get_match_table, which
            # gates every contribution below, so their prefactor is left at zero instead of
            # being evaluated.  The surviving branches are contracted in a single einsum over
            # the branch axis rather than one Python iteration each.
            valid_modes = freq >= self.phonon_cutoff # (nmodes,)
            A_qm = np.zeros(nmodes, dtype=complex)
            if np.any(valid_modes):
                atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # (natoms,)
                phvec_wap = atomic_phase[None,:,None] * phon_vecs[iq] # (nmodes, natoms, 3)
                phvec_valid = phvec_wap[valid_modes] # (nv, natoms, 3)
                freq_valid = freq[valid_modes] # (nv,)
                temp1_all = np.einsum('gi, kij, mkj->mkg', qG_vec_cart, self.BECs, phvec_valid) # (nv, natoms, ngs)
                temp3_all = temp1_all * temp2[None,None,:] * atomic_phase_G[None,:,:]
                factor_all = 1.0 / np.sqrt(2.0 * self.atomic_mass[None,:] * np.abs(freq_valid[:,None])) # (nv, natoms)
                A_qm[valid_modes] = Hamcts.JFOURPI * np.einsum('mkg, mk->m', temp3_all, factor_all) / self.volume_uc
            A_qm_cache[int(iq)] = A_qm

        # Report progress in 1% steps: the loop runs over q points, whose number can be
        # very large, so one line per iteration would flood the log.
        if self.rank == 0:
            logger = time_logger(total_cycles=100, routine_name='rate_cal_polar', line_per_step=True)
        # Progress is reported against the (k, q) pairs this rank owns, so it advances
        # evenly no matter whether the k or the q dimension dominates.
        _work_total = max(len(q_active) * len(my_active_indices), 1)
        _work_done = 0
        _decile_done = 0

        # Main loop: q outer, this rank's own k points inner.  Every rank only
        # diagonalises k+q for the k points it owns, so nothing has to be exchanged.
        _k_batch = max(1, int(_BATCH_BYTES / (32.0 * self.norbs ** 2)))
        k_grid_my = k_grid[my_active_indices]
        n_my = len(my_active_indices)
        # S(k) enters only through  sum_r[jbnd, ibnd] = conj(wave_kpq[jbnd]) . S(k) . wave_k[ibnd],
        # and S(k) depends on k alone.  With q on the outer loop a naive placement rebuilds it
        # for every (k, q) pair, which streams the whole of S_cell (ncells x norbs^2) once per
        # pair, which for a large basis is by far the dominant cost of this routine.  Matrix multiplication is associative, so cache
        #     Sw_k = S(k) . wave_k^T          shape (norbs, nbands)
        # once per k instead.  That is nbands/norbs of the memory and leaves a
        # (nbands, norbs) @ (norbs, nbands) product per pair.
        _Sw_k = np.empty((n_my, self.norbs, nbands), dtype=np.complex128)
        for _i_k in range(n_my):
            _ph_uc = np.exp(Hamcts.JTWOPI*np.sum(
                self.graph_data.nbr_shift_of_cell*k_grid_my[_i_k][None,:], axis=-1))
            _SK = np.einsum('n, nij->ij', _ph_uc, self.graph_data.S_cell) # (norbs, norbs)
            _Sw_k[_i_k] = _SK @ active_waves[_i_k].T
            del _SK
        for _iq_count, iq in enumerate(q_active):
            q = q_grid_group[iq]
            freq = freq_grid[iq]
            bose_qvs = bose_weight(freq, self.temperature)
            if self._lrc_cacheable:
                A_qm2 = np.abs(A_qm_cache[iq]) ** 2
            else:
                # The per-(band pair, branch) fallback needs the phonon quantities that the
                # cached path folds into A_qm once per q.
                _at_ph = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1))
                phvec_wap_q = _at_ph[None,:,None] * phon_vecs[iq] # (nmodes, natoms, 3)
                factor_all_q = 1.0 / np.sqrt(
                    2.0 * self.atomic_mass[None, :] * np.abs(freq[:, None])) # (nmodes, natoms)
            if n_my == 0:
                continue
            # Same batching as in the other branches: bound the batched H/S arrays.
            for _k_start in range(0, n_my, _k_batch):
                _k_end = min(_k_start + _k_batch, n_my)
                eig_kpq_b, wave_kpq_b = self._elec_cal_partial(
                    k_grid_my[_k_start:_k_end] + q[None, :], band_indice)
                for _i_b in range(_k_end - _k_start):
                    i_loc = _k_start + _i_b
                    ik_all = my_active_indices[i_loc]
                    eig_k = all_eigens[:, ik_all]
                    skip_mask = (eig_k > efocus_max) | (eig_k < efocus_min)
                    if skip_mask.all():
                        continue
                    eig_kpq = eig_kpq_b[:, _i_b]
                    wave_kpq = wave_kpq_b[_i_b]
                    match_table = self._get_match_table(eig_k, eig_kpq, freq)
                    # The contribution below is gated by the match table, so when it selects
                    # nothing for this (k, q) the overlap and coupling contractions are skipped.
                    if not np.any(match_table[~skip_mask]):
                        continue
                    fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                    # cal epc; sum_r_all[jbnd, ibnd] replaces the per-band-pair einsum.
                    # S(k) . wave_k^T was cached per k above, so nothing k-only is rebuilt here.
                    if self._lrc_cacheable:
                        sum_r_all = np.conj(wave_kpq) @ _Sw_k[i_loc] # (nbands, nbands)
                        # |g|^2 for every (ibnd, jbnd, imode) at once
                        g2_all = (np.abs(sum_r_all.T) ** 2)[:, :, None] * A_qm2[None, None, :]
                    else:
                        # First order does not factorise, so neither cache applies: evaluate
                        # the correction per (band pair, branch) as before.
                        g2_all = np.empty((nbands, nbands, nmodes))
                        for ibnd in range(nbands):
                            g2_all[ibnd] = np.abs(self._dipole_correction_mat(
                                ibnd, active_waves[i_loc], wave_kpq, k_grid_my[i_loc], q,
                                factor_all_q, phvec_wap_q, freq)) ** 2
                    de = eig_k[:, None, None] - eig_kpq[None, :, None]
                    fw = freq[None, None, :]
                    delta_f1 = w0gauss((de + fw) * self.inv_smearq) * self.inv_smearq
                    delta_f2 = w0gauss((de - fw) * self.inv_smearq) * self.inv_smearq
                    bose = bose_qvs[None, None, :]
                    fermi = fermi_kpqs[None, :, None]
                    if self.ishole:
                        # For holes the roles of the emission and absorption occupations swap.
                        contrib = weights_q_local[iq] * g2_all * match_table * (
                            (bose + 1.0 - fermi) * delta_f1 + (bose + fermi) * delta_f2)
                    else:
                        contrib = weights_q_local[iq] * g2_all * match_table * (
                            (bose + fermi) * delta_f1 + (bose + 1.0 - fermi) * delta_f2)
                    valid_ibnd = ~skip_mask
                    rate_all[valid_ibnd, ik_all] += contrib.sum(axis=(1, 2))[valid_ibnd]
            _work_done += n_my
            _decile = int(100 * _work_done / _work_total)
            if self.rank == 0 and _decile > _decile_done:
                _decile_done = _decile
                logger.step(_decile)

        rate_all *= Hamcts.TWOPI
        # Each rank accumulated the contribution of its own q subset, so the Allreduce(SUM)
        # completes the q integral.  It moves only a few hundred kB but cannot finish until
        # every rank has arrived, so an imbalanced workload appears as a long silence after the
        # progress bar reaches 100% -- that bar tracks rank 0 alone.  The first message without
        # the second therefore means the run is still waiting for the slowest rank rather than
        # that it has deadlocked.
        if self.comm is not None:
            if self.rank == 0:
                print('  waiting for all ranks at the rate_all reduction...', flush=True)
            _t_wait = time.time()
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
            if self.rank == 0:
                print('  reduction done after waiting {:.0f}s for the slowest rank.'.format(
                    time.time() - _t_wait), flush=True)
        return rate_all

    def rate_cal_rmp(self, k_grid, q_grid, band_indice, ecbm):
        """
        Calculate the scattering rate for remainder part only, |nk> within ecut.
        The apply_correction must be True.

        Args:
            k_grid (np.ndarray): The k vectors. # shape: (nks, 3)
            q_grid (np.ndarray): The q vectors. # shape: (nqs, 3)
            bands_indice (int): The only band indice.
            ecbm (float): The energy of CBM.

        Returns:
            rate_all: # shape: (nbands, nk) if eloss == False
        """

        nmodes = int(3) * self.natoms
        ncells = len(self.cell_shift_array_reduced)
        nbands = len(band_indice)
        nks = len(k_grid)
        nqs = len(q_grid)

        # Energy window around the band edge: above the CBM for electrons, below the VBM
        # for holes (ecbm holds the VBM energy when ishole is set).
        if self.ishole:
            efocus_min = ecbm - self.over_vbm
            efocus_max = ecbm
        else:
            efocus_min = -np.inf
            efocus_max = ecbm + self.over_cbm

        rate_all = np.zeros((nbands, nks))

        # ------------------------------------------------------------------
        # Ranks are arranged as n_kgroups k-groups x rank_size / n_kgroups q-groups.  The
        # per-k quantities built below are paid once per k point, so splitting k as well as
        # q divides that part of the work by n_kgroups.  n_kgroups = 1 (the default) leaves
        # the k points unsplit, i.e. the original pure q-parallel layout.
        # ------------------------------------------------------------------
        nk_groups = int(getattr(self, 'n_kgroups', 1))
        if nk_groups < 1 or nk_groups > self.rank_size or (self.rank_size % nk_groups) != 0:
            if self.rank == 0 and nk_groups != 1:
                print('  WARNING: n_kgroups={} does not divide rank_size={}; '
                      'falling back to 1 (pure q-parallel).'.format(nk_groups, self.rank_size), flush=True)
            nk_groups = 1
        nq_groups = self.rank_size // nk_groups
        # Ranks of one k-group are consecutive, so the row collectives below stay on-node.
        k_group_id = self.rank // nq_groups
        q_group_id = self.rank % nq_groups
        comm_k = self.comm.Split(color=q_group_id, key=k_group_id)      # same q, different k
        k_row_comm = self.comm.Split(color=k_group_id, key=q_group_id)  # same k, different q

        # q points are distributed over the q-groups; all ranks of a group share them.
        _q_split = np.zeros(nq_groups, dtype=int)
        for i in range(nqs):
            _q_split[i % nq_groups] += 1
        _q_cumsum = np.cumsum(_q_split)
        _q_start = int(_q_cumsum[q_group_id] - _q_split[q_group_id])
        _q_end = int(_q_cumsum[q_group_id])
        nqs_group = _q_end - _q_start
        q_grid_group = q_grid[_q_start:_q_end]
        weights_q_local = self.weight_q[_q_start:_q_end]
        self.weight_q = None

        # Phonons for this group's q subset: the k-ranks of a q-group need the same
        # frequencies and eigenvectors, so they take a slice each and allgather rather than
        # every rank repeating the whole subset.
        if self.rank == 0:
            print('  2D decomposition: {} k-groups x {} q-groups; {} q per group'.format(
                nk_groups, nq_groups, nqs_group), flush=True)
        if nqs_group > 0:
            _ph_counts = np.zeros(nk_groups, dtype=int)
            for i in range(nqs_group):
                _ph_counts[i % nk_groups] += 1
            _ph_cum = np.cumsum(_ph_counts)
            _ph_lo = int(_ph_cum[k_group_id] - _ph_counts[k_group_id])
            _ph_hi = int(_ph_cum[k_group_id])
            if _ph_hi > _ph_lo:
                freq_local, phon_local = self._phonon_cal(q_grid_group[_ph_lo:_ph_hi])
                phon_local = phon_local.reshape(-1, nmodes, self.natoms, 3)
            else:
                freq_local = np.empty((0, nmodes))
                phon_local = np.empty((0, nmodes, self.natoms, 3))
            freq_grid = np.concatenate(comm_k.allgather(freq_local), axis=0)
            phon_vecs = np.concatenate(comm_k.allgather(phon_local), axis=0)
            # change fractional coordinates to cartesian coordinates
            q_grid_group = self._frac2car(q_grid_group)
        else:
            freq_grid = np.empty((0, nmodes))
            phon_vecs = np.empty((0, nmodes, self.natoms, 3))
            q_grid_group = np.empty((0, 3))

        # Only the band energies are needed globally: every rank must agree on which k
        # points fall outside the energy window.  The wave functions are indexed only
        # at the k points a rank owns, and gathering them everywhere would cost
        # nks * nbands * norbs * 16 B per rank, so they stay local.
        # The k grid is split over the k-ranks; within a column all q-groups need the
        # same k data, so only q_group_id == 0 diagonalises and broadcasts along the row.
        _k_split = np.zeros(nk_groups, dtype=int)
        for i in range(nks):
            _k_split[i % nk_groups] += 1
        _k_split_cum = np.cumsum(_k_split)
        k_start_idx = int(_k_split_cum[k_group_id] - _k_split[k_group_id])
        k_end_idx = int(_k_split_cum[k_group_id])
        k_grid_local = k_grid[k_start_idx:k_end_idx]
        if nk_groups == 1:
            # One k-group means k is not split: k_grid_local is already the whole grid and
            # the gather below would be a no-op, but the q_group_id == 0 guard would still
            # leave a single rank diagonalising while every other rank waits on the
            # broadcast.  Each rank therefore diagonalises for itself, which is the
            # redundant-but-parallel layout the pure q-parallel path expects.
            eigen_local, eigen_vecs_local = self._elec_cal_partial(k_grid_local, band_indice)
            all_eigens = np.ascontiguousarray(eigen_local, dtype=np.float64)
            col0_comm = MPI.COMM_NULL
        else:
            if q_group_id == 0:
                if k_grid_local.size > 0:
                    eigen_local, eigen_vecs_local = self._elec_cal_partial(k_grid_local, band_indice)
                else:
                    eigen_local = np.empty((nbands, 0))
                    eigen_vecs_local = np.empty((0, nbands, self.norbs), dtype=np.complex128)
            else:
                eigen_local, eigen_vecs_local = None, None

            # The band energies are gathered globally: every rank must agree on which k points
            # fall inside the energy window, and eig_k is needed in the main loop.  One
            # representative per k-rank contributes and the assembled array goes back along the
            # row; the payload is only nbands * nks * 8 B.
            col0_color = 0 if q_group_id == 0 else MPI.UNDEFINED
            col0_comm = self.comm.Split(color=col0_color, key=k_group_id)
            if q_group_id == 0:
                all_eigens = np.concatenate(
                    col0_comm.allgather(np.ascontiguousarray(eigen_local, dtype=np.float64)), axis=1)
            else:
                all_eigens = None
            all_eigens = k_row_comm.bcast(all_eigens, root=0)

        # Mark the k points outside the energy window and keep only the active ones.
        rate_all[(all_eigens > efocus_max) | (all_eigens < efocus_min)] = np.inf
        active_k_mask = np.any((all_eigens <= efocus_max) & (all_eigens >= efocus_min), axis=0)
        active_k_indices = np.where(active_k_mask)[0]
        n_active = len(active_k_indices)
        if self.rank == 0:
            print('  active k points: {} / {} ({:.2f}%)'.format(n_active, nks, 100.0*n_active/max(nks,1)), flush=True)

        # Only the active k points enter the main loop, so replicating their wave functions
        # is cheap and lets the *active* points be distributed below.  That matters for load
        # balance: the energy window usually selects a clustered region, so slicing the raw
        # mesh would leave most ranks idle.
        _mine_in_slice = active_k_indices[(active_k_indices >= k_start_idx) & (active_k_indices < k_end_idx)]
        if nk_groups == 1:
            # Every rank already holds the eigenvectors of the whole grid, so the active
            # subset is a plain slice; no gather or broadcast is involved.
            active_waves = np.ascontiguousarray(
                eigen_vecs_local[_mine_in_slice - k_start_idx], dtype=np.complex128)
        else:
            if q_group_id == 0:
                active_waves = np.concatenate(col0_comm.allgather(np.ascontiguousarray(
                    eigen_vecs_local[_mine_in_slice - k_start_idx], dtype=np.complex128)), axis=0)
            else:
                active_waves = None
            active_waves = k_row_comm.bcast(active_waves, root=0)
        if col0_comm != MPI.COMM_NULL:
            col0_comm.Free()

        # Distribute the active k points over the k-groups so every rank gets the same
        # number of them no matter how they are spread over the mesh.  active_waves is
        # assembled in ascending k order, so active_waves[_k_lo + i] is the wave function
        # of my_active_indices[i].
        _k_counts = np.zeros(nk_groups, dtype=int)
        for i in range(n_active):
            _k_counts[i % nk_groups] += 1
        _k_cum = np.cumsum(_k_counts)
        _k_lo = int(_k_cum[k_group_id] - _k_counts[k_group_id])
        _k_hi = int(_k_cum[k_group_id])
        my_active_indices = active_k_indices[_k_lo:_k_hi]

        if self.rank == 0:
            logger = time_logger(total_cycles=100, routine_name='rate_cal_rmp', line_per_step=True)
        # Progress is reported against the (k, q) pairs this rank owns, so it advances
        # evenly no matter whether the k or the q dimension dominates.
        _work_total = max(len(my_active_indices) * max(nqs_group, 1), 1)
        _work_done = 0
        _decile_done = 0

        # Pre-compute the q-dependent prefactor of the dipole correction; see the comment in
        # rate_cal_polar.  A_qm only depends on (q, imode), so it is hoisted out of the k loop.
        corr_mask = (np.linalg.norm(q_grid_group, axis=-1) < self.q_cut) if nqs_group else np.empty(0, dtype=bool)
        # A_qm depends only on (q, imode) and every rank of a q-group would otherwise build
        # the same table, so the q are shared out over the group and the result allgathered
        # -- the same pattern as the phonon spectrum above.
        _aq_counts = np.zeros(nk_groups, dtype=int)
        for i in range(nqs_group):
            _aq_counts[i % nk_groups] += 1
        _aq_cum = np.cumsum(_aq_counts)
        _aq_lo = int(_aq_cum[k_group_id] - _aq_counts[k_group_id])
        _aq_hi = int(_aq_cum[k_group_id])
        _A_local = np.zeros((_aq_hi - _aq_lo, nmodes), dtype=complex)
        for _aj in range(_aq_hi - _aq_lo):
            iq = _aq_lo + _aj
            if not corr_mask[iq]:
                continue
            q = q_grid_group[iq]
            freq = freq_grid[iq]
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q)
            # Quantities shared by every branch.
            atomic_phase_G = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos))
            temp2 = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term
            # Branches below the phonon cutoff are gated out by _get_match_table, so their
            # prefactor stays zero; the rest are contracted in one einsum over the branch axis.
            valid_modes = freq >= self.phonon_cutoff
            A_qm = np.zeros(nmodes, dtype=complex)
            if np.any(valid_modes):
                atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1))
                phvec_wap_iq = atomic_phase[None,:,None] * phon_vecs[iq]
                phvec_valid = phvec_wap_iq[valid_modes]
                freq_valid = freq[valid_modes]
                temp1_all = np.einsum('gi, kij, mkj->mkg', qG_vec_cart, self.BECs, phvec_valid)
                temp3_all = temp1_all * temp2[None,None,:] * atomic_phase_G[None,:,:]
                factor_all = 1.0 / np.sqrt(2.0 * self.atomic_mass[None,:] * np.abs(freq_valid[:,None]))
                A_qm[valid_modes] = Hamcts.JFOURPI * np.einsum('mkg, mk->m', temp3_all, factor_all) / self.volume_uc
            _A_local[_aj] = A_qm
        A_qm_cache = (np.concatenate(comm_k.allgather(_A_local), axis=0)
                      if nqs_group else np.zeros((0, nmodes), dtype=complex))

        # Only the cells that survive the cell cut ever enter the phase factors.
        nbr_shift_cut = self.nbr_shift_of_cell_sc[self.cell_cut_list]   # (ncells_cut, 3)
        # _elec_cal_partial materialises H(k) and S(k) for the whole batch, i.e.
        # nk * norbs^2 * 32 B.  Cap the batch so that stays bounded whatever the
        # basis size: a small basis then runs unbatched in practice, while a large
        # one is split rather than allocating the whole batch at once.
        _q_batch = max(1, int(_BATCH_BYTES / (32.0 * self.norbs ** 2)))
        # Main loop: this rank's own active k points outer, its q subset inner.
        for _i_loc, ik_all in enumerate(my_active_indices):
                k = k_grid[ik_all]
                eig_k = all_eigens[:, ik_all]
                wave_k = active_waves[_k_lo + _i_loc]
                skip_mask = (eig_k > efocus_max) | (eig_k < efocus_min)
                if skip_mask.all():
                    continue
                phase_k = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                # grad_mat is indexed [A, B, m, n, i, j] with A <- conj(phase_kpq),
                # B <- phase_k, m <- conj(wave_kpq), n <- wave_k.  Folding phase_k and
                # wave_k in once per k point replaces the loop over cell pairs for every
                # (q, band pair, branch), and drops the second orbital axis.
                # B is contracted before n: B leads grad_mat[A], so (B, m*n*i*j) is a
                # contiguous view and the product never transposes the slice.  Taking n
                # first would, since axis 2 is not trailing.
                _pk_cut = phase_k[self.cell_cut_list].astype(np.complex128)
                _nB = len(self.cell_cut_list)
                grad_contracted_k = np.empty((_nB, self.norbs,
                                              self.natoms, 3, nbands), dtype=np.complex128)
                _gm_is_complex = np.iscomplexobj(self.grad_mat)
                # Real grad_mat (no SOC): split phase_k into its real and imaginary rows so a
                # single GEMM covers both and grad_mat is read only once.
                _pk_ri = None if _gm_is_complex else np.stack([_pk_cut.real, _pk_cut.imag])
                _wkT = np.ascontiguousarray(wave_k.T)             # (norbs, nbands)
                for _iA in range(_nB):
                    _gA = self.grad_mat[_iA].reshape(_nB, -1)     # (B, m*n*i*j) view
                    if _gm_is_complex:
                        _tB = _pk_cut @ _gA
                    else:
                        _ri = _pk_ri @ _gA
                        _tB = _ri[0] + 1j*_ri[1]
                    _tB = _tB.reshape(self.norbs, self.norbs, self.natoms*3) # (m, n, i*j)
                    # (m, i*j, n) @ (n, nbands) -> (m, i*j, nbands), all bands in one call
                    grad_contracted_k[_iA] = np.matmul(
                        _tB.transpose(0, 2, 1), _wkT).reshape(
                            self.norbs, self.natoms, 3, nbands)
                    del _tB
                # S(k) contracted over cells, for the dipole correction.
                phase_k_uc = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k[None,:], axis=-1))
                SK_k = np.einsum('n, nij->ij', phase_k_uc, self.graph_data.S_cell)
                # S(k) enters only through conj(wave_kpq) . S(k) . wave_k^T and both depend
                # on k alone, so fold wave_k in here rather than per band pair.
                _Sw_k = SK_k @ wave_k.T # (norbs, nbands)
                kpq_all = k + q_grid_group                            # (nqs_group, 3)
                # k+q states and cell phases are evaluated in batches instead of one q at a time:
                # the diagonalisation is batched over the chunk, only the bands in band_indice are
                # requested, and the cell phase is built in one call over the cells that survive the
                # cell cut.  A chunk rather than the whole q subset keeps the batch arrays bounded.
                for _q_start in range(0, nqs_group, _q_batch):
                    _q_end = min(_q_start + _q_batch, nqs_group)
                    eig_kpq_b, wave_kpq_b = self._elec_cal_partial(kpq_all[_q_start:_q_end], band_indice)
                    phase_kpq_cut_b = np.exp(Hamcts.JTWOPI*np.einsum('qd, nd->qn',
                                             kpq_all[_q_start:_q_end], nbr_shift_cut)).astype(np.complex128)
                    for _iq_b in range(_q_end - _q_start):
                        iq = _q_start + _iq_b
                        q = q_grid_group[iq]
                        apply_correction_for_this_q = corr_mask[iq]
                        # phonon spectrum
                        freq = freq_grid[iq]
                        eigen_vec_phon = phon_vecs[iq]
                        atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                        phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                        bose_qvs = bose_weight(freq, self.temperature)
                        # calculate the electronic info for k+q
                        eig_kpq = eig_kpq_b[:, _iq_b]
                        wave_kpq = wave_kpq_b[_iq_b]
                        match_table = self._get_match_table(eig_k, eig_kpq, freq)
                        # Every contribution below is gated by the match table, so when it
                        # selects nothing for this (k, q) the cell contraction and the
                        # band/branch contractions are skipped altogether.  The energy
                        # window makes this the common case.
                        if not np.any(match_table[~skip_mask]):
                            continue
                        fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                        # cal epc
                        grad_ph = np.tensordot(np.conj(phase_kpq_cut_b[_iq_b]),
                                              grad_contracted_k, axes=([0], [0])) # (m,i,j,nbands)
                        # Built for every mode at once, so the branches the match table
                        # discards are formed too; leaving 1/sqrt(0) in them would put an
                        # inf where the per-element code never evaluates anything.
                        factor_all = np.zeros((len(freq), len(self.atomic_mass)))
                        _fv = freq >= self.phonon_cutoff
                        factor_all[_fv] = 1.0 / np.sqrt(
                            2.0 * self.atomic_mass[None, :] * np.abs(freq[_fv, None]))
                        if apply_correction_for_this_q and self._lrc_cacheable:
                            sum_r_all = np.conj(wave_kpq) @ _Sw_k # (nbands, nbands)
                            A_qm = A_qm_cache[iq]
                        for ibnd in range(nbands):
                            if skip_mask[ibnd]:
                                continue
                            # Tested before the contractions below rather than after: the
                            # energy window makes an empty row the common case.
                            mt = match_table[ibnd] # (nbands, nmodes)
                            if not np.any(mt):
                                continue
                            grad_partial = grad_ph[..., ibnd] # (norbs, natoms, 3); wave_k already folded in
                            # grad_elec for ALL jbnd at once: (nbands, natoms, 3)
                            grad_elec_all = np.einsum('jm, mab -> jab', np.conj(wave_kpq), grad_partial)
                            # epc for all (jbnd, imode): (nbands, nmodes)
                            epc_mat = np.einsum('jax, max, ma -> jm', grad_elec_all, phvec_wap, factor_all)
                            if apply_correction_for_this_q:
                                if self._lrc_cacheable:
                                    # epc_corr[j,m] = A_qm[m] * sum_r_all[j, ibnd]
                                    epc_corr_mat = A_qm[None, :] * sum_r_all[:, ibnd, None] # (nbands, nmodes)
                                else:
                                    epc_corr_mat = self._dipole_correction_mat(
                                        ibnd, wave_k, wave_kpq, k, q, factor_all, phvec_wap, freq)
                                epc_full = epc_mat + epc_corr_mat
                                g2_mat = np.abs(epc_full) ** 2 - np.abs(epc_corr_mat) ** 2
                            else:
                                g2_mat = np.abs(epc_mat) ** 2 # (nbands, nmodes)

                            de = eig_k[ibnd] - eig_kpq # (nbands,)
                            d1 = w0gauss((de[:, None] + freq[None, :]) * self.inv_smearq) * self.inv_smearq
                            d2 = w0gauss((de[:, None] - freq[None, :]) * self.inv_smearq) * self.inv_smearq
                            bose = bose_qvs[None, :]    # (1, nmodes)
                            fermi = fermi_kpqs[:, None] # (nbands, 1)
                            if self.ishole:
                                # For holes the roles of the emission and absorption
                                # occupations swap.
                                contrib = g2_mat * mt * ((bose + 1.0 - fermi) * d1 +
                                                         (bose + fermi) * d2)
                            else:
                                contrib = g2_mat * mt * ((bose + fermi) * d1 +
                                                         (bose + 1.0 - fermi) * d2)
                            rate_all[ibnd, ik_all] += weights_q_local[iq] * contrib.sum()
                    # Counted per q batch rather than per k point: a rank owns only a few
                    # active k, so a per-k update could not resolve single percent steps.
                    _work_done += _q_end - _q_start
                    _decile = int(100 * _work_done / _work_total)
                    if self.rank == 0 and _decile > _decile_done:
                        _decile_done = _decile
                        logger.step(_decile)

        comm_k.Free()
        k_row_comm.Free()
        rate_all *= Hamcts.TWOPI
        # Each rank accumulated its own (k subset) x (q subset); the global Allreduce(SUM)
        # completes both the q integral and the k gathering at once.  It moves only a few
        # hundred kB but cannot finish until every rank has arrived, so an imbalanced
        # workload appears as a long silence after the progress bar reaches 100% -- that bar
        # tracks rank 0 alone.  The first message without the second therefore means the run
        # is still waiting for the slowest rank, not that it has deadlocked.
        if self.comm is not None:
            if self.rank == 0:
                print('  waiting for all ranks at the rate_all reduction...', flush=True)
            _t_wait = time.time()
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
            if self.rank == 0:
                print('  reduction done after waiting {:.0f}s for the slowest rank.'.format(
                    time.time() - _t_wait), flush=True)
        return rate_all

    def mobility_cal(self):
        """
        Calculate the mobility in RTA.

        Args:

        Returns:
        """

        k_grid, self.weight_k, grid_all, grid2ir_idx = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=True)
        # consider the spin factor
        self.weight_k *= 2.0
        if self.MC_sampling == 'cauchy':
            q_grid, self.weight_q = random_cauchy(self.nsamples, cauchy_scale=self.cauchy_scale, random_seed=self.sampling_seed)
        elif self.MC_sampling == 'uniform':
            q_grid, self.weight_q = random_uniform(self.nsamples, random_seed=self.sampling_seed)
        else:
            q_grid = self._get_monkhorst_pack(self.q_size, self.graph_data.latt, return_frac=True)
            self.weight_q = np.ones(len(q_grid)) / len(q_grid)
        if self.ishole:
            band_edge_index = self.VBM_band_index
        else:
            band_edge_index = self.CBM_band_index
        iband_edge = np.where(np.array(self.bands_indices)==band_edge_index)[0][0]
        
        # The band energies on the irreducible k grid are needed by every rank, but the
        # diagonalisation is embarrassingly parallel: split the k points, diagonalise the
        # local chunk and allgather the (small) eigenvalue array.
        if self.comm is not None and self.rank_size > 1:
            nk = len(k_grid)
            counts = [nk // self.rank_size + (1 if i < nk % self.rank_size else 0)
                      for i in range(self.rank_size)]
            offsets = [sum(counts[:i]) for i in range(self.rank_size)]
            k_grid_local = k_grid[offsets[self.rank]:offsets[self.rank] + counts[self.rank]]
            if len(k_grid_local) > 0:
                enks_local, _ = self._elec_cal(k_grid_local)
                enks_local = enks_local[self.bands_indices, :]
            else:
                enks_local = np.empty((len(self.bands_indices), 0))
            enks = np.concatenate(self.comm.allgather(enks_local), axis=1) # (nbnd, nks)
        else:
            enks, _ = self._elec_cal(k_grid) # (nbandtots, nk)
            enks = enks[self.bands_indices, :] # (nbnd, nk)

        # carrier_density has been multiplied by unit cell volume
        self.efermi, self.carrier_density = self._get_fermi_level_insulator(enks, iband_edge)
        if self.rank == 0:
            print("Fermi energy = {} eV, Carrier density = {} cm^(-3).".format(
                self.efermi * Hamcts.HARTREEtoEV, self.carrier_density * self.inv_cell / (Hamcts.BOHRtoCM ** 3)
                ))
            
        # The band edge the window is measured from: the CBM for electrons, the VBM for
        # holes.  The scattering routines take it as `ecbm` either way.
        if self.ishole:
            ecbm = self._get_evbm(enks, iband_edge)
        else:
            ecbm = self._get_ecbm(enks, iband_edge)

        # k points are parallelized and k grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(len(grid_all)):
            split_sections[i%self.rank_size] += 1
        
        split_sections = np.cumsum(split_sections, axis=0)
        grid_all = np.split(grid_all, indices_or_sections=split_sections, axis=0)
        
        # _get_ir_reciprocal_mesh hands back the raw integer mesh; scale and rotate only
        # this rank's slice so no full-mesh float64 copy is ever materialised.
        _grid_mine = grid_all[self.rank]
        _n_grid_mine = len(_grid_mine)
        if _n_grid_mine > 0:
            _grid_mine = np.tensordot(_grid_mine / np.array(self.k_size, dtype=float),
                                      self.graph_data.lat_per_inv, axes=1)
            # calculate the electron velocity in parallel
            elec_velocities = self.vel_nk_cal_from_HS(self.bands_indices, _grid_mine) # (nk_split, nbands, 3)
        else:
            elec_velocities = np.empty((0, len(self.bands_indices), 3))

        rate_ir = np.full_like(enks, np.inf)
        if self.polar_rate_path and (not self.rmp_rate_path):
            if self.rank == 0:
                print(f"Reading polar scattering rate from {self.polar_rate_path}")
                fin = open(self.polar_rate_path, 'r')
                lines = fin.readlines()
                fin.close()
                for line in lines[2:]:
                    words = line.split()
                    ik = int(words[0])
                    ibnd = int(words[1])
                    rate_ir[ibnd, ik] = float(words[4])
            rate_ir = self.comm.bcast(rate_ir, root=0)
        elif self.rmp_rate_path and (not self.polar_rate_path):
            if self.rank == 0:
                print(f"Reading rmp scattering rate from {self.rmp_rate_path}")
                fin = open(self.rmp_rate_path, 'r')
                lines = fin.readlines()
                fin.close()
                for line in lines[2:]:
                    words = line.split()
                    ik = int(words[0])
                    ibnd = int(words[1])
                    rate_ir[ibnd, ik] = float(words[4])
            rate_ir = self.comm.bcast(rate_ir, root=0)
        elif self.polar_rate_path and self.rmp_rate_path:
            if self.rank == 0:
                print(f"Reading polar scattering rate from {self.polar_rate_path}")
                print(f"Reading rmp scattering rate from {self.rmp_rate_path}")
                fin = open(self.polar_rate_path, 'r')
                lines = fin.readlines()
                fin.close()
                for line in lines[2:]:
                    words = line.split()
                    ik = int(words[0])
                    ibnd = int(words[1])
                    rate_ir[ibnd, ik] = float(words[4])
                fin = open(self.rmp_rate_path, 'r')
                lines = fin.readlines()
                fin.close()
                for line in lines[2:]:
                    words = line.split()
                    ik = int(words[0])
                    ibnd = int(words[1])
                    rate_ir[ibnd, ik] += float(words[4])
            rate_ir = self.comm.bcast(rate_ir, root=0)
        else:
            if self.polar_split == 'polar':
                fout_name = 'rate_nk_polar.dat'
                rate_ir = self.rate_cal_polar(k_grid, q_grid, self.bands_indices, ecbm)
            elif self.polar_split == 'rmp':
                fout_name = 'rate_nk_rmp.dat'
                rate_ir = self.rate_cal_rmp(k_grid, q_grid, self.bands_indices, ecbm)
            else:
                fout_name = 'rate_nk.dat'
                rate_ir = self.rate_cal(k_grid, q_grid, self.bands_indices, ecbm)

            rate_ir[0, 0] = np.inf
            if self.rank == 0:
                fout = open(os.path.join(self.outdir, fout_name), 'w')
                fout.write('ef = {} a.u.    nc = {} a.u.\n'.format(self.efermi, self.carrier_density))
                fout.write('ik    ibnd    weight_k    enk(a.u.)    scattering_rate(a.u.)\n')
                for ik in range(len(enks[0])):
                    for ibnd in range(len(self.bands_indices)):
                        if not np.isinf(rate_ir[ibnd, ik]):
                            fout.write('{}  {}  {}  {}  {}\n'.format(ik, ibnd, self.weight_k[ik], enks[ibnd, ik], rate_ir[ibnd, ik]))
                fout.close()
        
        rate_all = np.split(rate_ir[:, grid2ir_idx], indices_or_sections=split_sections, axis=-1)
        rate_rank = rate_all[self.rank]
        enks_all = np.split(enks[:, grid2ir_idx], indices_or_sections=split_sections, axis=-1)
        enks_rank = enks_all[self.rank]
        
        if _n_grid_mine > 0:
            mdf = minus_dfermi(enks_rank - self.efermi, self.temperature)
            sigma_mat = oe.contract('nk, kni, knj, nk->ij', mdf, elec_velocities, elec_velocities, 1.0 / rate_rank) # shape: (3, 3)
        else:
            sigma_mat = np.zeros((3,3))

        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, sigma_mat, op=MPI.SUM)
        sigma_mat = sigma_mat / float(len(grid2ir_idx)) * 2.0
        mobility = (sigma_mat * Hamcts.EV * self.inv_cell) / (Hamcts.HBAR_EV * Hamcts.BOHRtoCM)
        mobility = mobility / (Hamcts.EV * self.carrier_density * self.inv_cell) * (Hamcts.BOHRtoCM ** 3)
        if self.rank == 0:
            print("Mobility(cm^2/V/s)      x                y                z")
            print('       x       {:12.4e}      {:12.4e}      {:12.4e}'.format(*mobility[0]))
            print('       y       {:12.4e}      {:12.4e}      {:12.4e}'.format(*mobility[1]))
            print('       z       {:12.4e}      {:12.4e}      {:12.4e}'.format(*mobility[2]))
        return mobility

    def _get_hsk_path(self, nks_path, hsk_points:list[list[float]]=None, hsk_labels:list[str]=None):
        """
        Get the q points path.

        Args:
            hsk_points (list or np.ndarray): List of special q points in fractional coordinate.
            nks_path (list or int): Number of q points between. 
            hsk_labels (list): The name of special q points.
            automatic (bool): If true, automatically generate the high symmetry points.
        Returns:
            q_path (np.ndarray): List of q points path.
        """
        kpts=kpoints_generator(dim_k=3, lat=self.graph_data.latt)
        if self.high_symmetry_auto:
            struct = Structure(lattice=self.graph_data.latt*Hamcts.BOHRtoANG,
                species=[Element.from_Z(k).symbol for k in self.graph_data.species],
                coords=self.graph_data.pos*Hamcts.BOHRtoANG, coords_are_cartesian=True)
            try:
                kpath_seek = KPathSeek(structure=struct)
            except:
                raise RuntimeError("Cannot automatically generate k path for this structure.", '1011')
            hsk_labels = []
            for lbs in kpath_seek.kpath['path']:
                hsk_labels += lbs
            print(kpath_seek.kpath['path'])
            print(kpath_seek.kpath['kpoints'])

            # remove adjacent duplicates   
            res = [hsk_labels[0]]
            [res.append(x) for x in hsk_labels[1:] if x != res[-1]]
            hsk_labels = res

            hsk_points = [kpath_seek.kpath['kpoints'][k] for k in hsk_labels]
            hsk_labels_plot = [rf'${lb}$' for lb in hsk_labels]

        try:
            hsk_labels_plot = [rf'${lb}$' for lb in hsk_labels]
            k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(hsk_points, nks_path)
            k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
        except:
            raise RuntimeError("kpoints_generator failed.", '1012')
        return k_vec, k_dist, k_node, hsk_labels_plot

    def plot_band(self):
        """ 
        Calculate the electron energy for a special k path.

        Args:

        Returns:
        """
        hsk_path_vecs = self._frac2car(self.high_symmetry_k_vecs)
        enks, _ = self._elec_cal(hsk_path_vecs)
        enks = enks[self.dispersion_select_index, :] * Hamcts.HARTREEtoEV
        nbands, nks = enks.shape
        fout = open(os.path.join(self.outdir, "bands.dispersion"), 'w')
        fout.write(f"# k_lable: {' '.join(self.high_symmetry_labels)}\n")
        fout.write(f"# k_node: { '  '.join([str(round(each, 10)) for each in self.high_symmetry_k_nodes]) } \n")
        for ibnd in range(nbands):
            for ik in range(nks):
                fout.write(f"{str(round(self.high_symmetry_k_dist[ik], 10))}    {str(round(enks[ibnd, ik], 10))}\n")
            fout.write('\n')
        fout.close()

    def plot_dos(self, k_grid, bands_indices, emin:float=0.0, emax:float=0.0, estep:float=0.01, cbm_band_index:int=0):
        """ 
        Calculate the DOS for a special energy list that referenced to CBM.

        Args:
            k_grid (list or np.ndarray): (3,)
            bands_indices (np.ndarray): Band included.
            emin (float): Minimum energy of the energy range in eV, referenced to CBM.
            emax (float): Maximum energy of the energy range in eV, referenced to CBM.
            estep (float): Energy step in eV.
            cbm_band_index (int): Band index of CBM.

        Returns:
            ene_list (np.ndarray): # shape (nenes) Energy list in eV, referenced to CBM.
            dos_list (np.ndarray): # shape (nenes) Dos list in 1/eV.
        """
        k_grid, self.weight_k = self._get_ir_reciprocal_mesh(k_grid)
        enks, _ = self._elec_cal(k_grid) # (nbnds, nks)
        enks = enks[bands_indices, :]
        iband_edge = np.where(np.array(bands_indices)==cbm_band_index)[0][0]
        ecbm = self._get_ecbm(enks, iband_edge)
        enks = enks - ecbm
        self.weight_k = self.weight_k * 2.0

        ene_list = np.arange(emin, emax+estep, estep) * Hamcts.EVtoHARTREE
        dos_list = np.zeros(ene_list.shape[0])

        for ie, ene in enumerate(ene_list):
            for _, ekks in enumerate(enks):
                for ik, enk in enumerate(ekks):
                    delta_f3 = w0gauss((enk - ene) * self.inv_smeark) * self.inv_smeark
                    dos_list[ie] += delta_f3 * self.weight_k[ik]

        dos_list = dos_list / Hamcts.HARTREEtoEV
        ene_list = ene_list * Hamcts.HARTREEtoEV
        return ene_list, dos_list

    def plot_phonon(self):
        """ 
        Calculate the phonon dispersion for a special q path.

        Args:

        Returns:
        """
        wqvs, _ = self._phonon_cal(self.high_symmetry_k_vecs)
        wqvs = wqvs[:, self.dispersion_select_index] * Hamcts.HARTREEtoMEV
        nqs, nmodes = wqvs.shape
        fout = open(os.path.join(self.outdir, "phbands.dispersion"), 'w')
        fout.write(f"# k_lable: {' '.join(self.high_symmetry_labels)}\n")
        fout.write(f"# k_node: { '  '.join([str(round(each, 10)) for each in self.high_symmetry_k_nodes]) }\n")
        for imode in range(nmodes):
            for iq in range(nqs):
                fout.write(f"{str(round(self.high_symmetry_k_dist[iq], 10))}    {str(round(wqvs[iq, imode], 10))}\n")
            fout.write('\n')
        fout.close()

    def plot_epc(self):
        """ 
        Calculate the EPC for a special q path.

        Args:

        Returns:
        """
        epcs, freqs = self.EPC_cal_path(self.epc_path_fix_k, self.high_symmetry_k_vecs,
                                        self.dispersion_select_index[0], self.dispersion_select_index[1],
                                        do_symm=False, return_freq=True) # shape: (nqs, nmodes)
        epcs = np.abs(epcs) * Hamcts.HARTREEtoMEV
        # Written beside the coupling: a branch index alone does not name a mode to a
        # reader, and the frequency also confirms two runs walked the same q path.
        freqs = freqs * Hamcts.HARTREEtoMEV
        nqs, nmodes = epcs.shape
        fout = open(os.path.join(self.outdir, "epc.dispersion"), 'w')
        fout.write(f"# k_lable: {' '.join(self.high_symmetry_labels)}\n")
        fout.write(f"# k_node: { '  '.join([str(round(each, 10)) for each in self.high_symmetry_k_nodes]) }\n")
        fout.write("# Format: q_distance(Bohr^-1)  Frequency(meV)  |g|(meV)\n")
        for imode in range(nmodes):
            for iq in range(nqs):
                fout.write(f"{str(round(self.high_symmetry_k_dist[iq], 10))}  "
                           f"{str(round(freqs[iq, imode], 10))}  "
                           f"{str(round(epcs[iq, imode], 10))}\n")
            fout.write('\n')
        fout.close()

    def plot_phdos(self, q_dim, emin:float=0.0, emax:float=0.0, estep:float=0.01):
        q_grid = self._get_monkhorst_pack(q_dim, return_frac=True)
        nqs = len(q_grid)
        wqvs, _ = self._phonon_cal(q_grid)
        wqvs = wqvs
        omegas = np.arange(emin, emax+Hamcts.TENPM80, estep) * Hamcts.MEVtoHARTREE
        phdos = np.zeros_like(omegas)
        for iomega, omega in enumerate(omegas):
            for iq, wqq in enumerate(wqvs):
                for imode, wqv in enumerate(wqq):
                    tmp = w0gauss((wqv - omega) * self.inv_smearq) * self.inv_smearq
                    phdos[iomega] += tmp
        omegas = omegas * Hamcts.HARTREEtoMEV
        phdos = phdos / nqs / Hamcts.HARTREEtoMEV
        return omegas, phdos

    def epc_grid_cal(self, k_size, q_size, bands_indices):
        k_grid = self._get_monkhorst_pack(k_size, self.graph_data.latt, return_frac=False)
        q_grid = self._get_monkhorst_pack(q_size, self.graph_data.latt, return_frac=True)
        
        nmodes = int(3) * self.natoms
        nbands = len(bands_indices)
        nks = len(k_grid)
        nqs = len(q_grid)

        # q points are parallelized and q grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nqs):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        q_grid = np.split(q_grid, indices_or_sections=split_sections, axis=0)
        if q_grid[self.rank].size>0:
            q_grid = q_grid[self.rank]
            nqs_local = len(q_grid)
            q_grid = self._frac2car(q_grid)
        else:
            q_grid = np.empty((0, 3))
            nqs_local = int(0)

        gnorm_save = np.zeros((nks, nqs_local, nbands, nbands, nmodes))
        if self.rank == 0:
            logger = time_logger(total_cycles=nks, routine_name='epc_grid_cal')
        for ik, k in enumerate(k_grid):
            _, wave_k = self._elec_cal(k)
            wave_k = wave_k[0, bands_indices, :]
            phase_k = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
            for iq, q in enumerate(q_grid):
                freqs, phon_vecs = self._phonon_cal(q)
                freqs = freqs[0]
                phon_vecs = phon_vecs[0].reshape(nmodes, self.natoms, 3)
                atomic_phase = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                phvec_wap = atomic_phase[None,:,None] * phon_vecs # shape (nbranches, natoms, 3)
                kpq = k + q
                _, wave_kpq = self._elec_cal(kpq)
                wave_kpq = wave_kpq[0, bands_indices, :]
                phase_kpq = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                for ibnd in range(nbands):
                    for jbnd in range(nbands):
                        tmp1 = np.einsum('m,n -> mn', np.conj(wave_kpq[jbnd]), wave_k[ibnd])
                        for imode in range(nmodes):
                            factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * abs(freqs[imode])) # shape:(natoms,)
                            tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*phvec_wap[imode], tmp1)
                            epc = 0.0
                            for i_m, m in enumerate(self.cell_cut_list): # ncells
                                for i_n, n in enumerate(self.cell_cut_list): # ncells  
                                    epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                            gnorm_save[ik, iq, ibnd, jbnd, imode] = np.abs(epc)
            if self.rank == 0:
                logger.step(ik+1)

            gnorm_save_all = self.comm.allgather(gnorm_save)
            gnorm_save_all = np.concatenate(gnorm_save_all, axis=1) # (norbs, nk)
            np.save(os.path.join(self.outdir, "gnorm_save.npy"), arr=gnorm_save_all)

