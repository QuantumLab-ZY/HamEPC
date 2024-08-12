'''
Descripttion: 
version: 
Author: Yang Zhong & Shixu Liu
Date: 2023-04-07 14:24:16
LastEditors: Yang Zhong
LastEditTime: 2024-05-16 23:42:25
'''
import os
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
            if self.read_large_grad_mat:
                original_list = list(range(self.rank_size))
                result_list = [original_list[i:i+3] for i in range(0, len(original_list), 3)] # 每次三个进程同时读大的grad_mat文件,防止内存不够
                for tmp_list in result_list:
                    if self.rank in tmp_list:
                        self.grad_mat = np.load(self.grad_mat_path)[self.cell_cut_array[:,None], self.cell_cut_array[None,:]]
                    self.comm.Barrier()
            else:
                self.grad_mat = np.load(self.grad_mat_path)[self.cell_cut_array[:,None], self.cell_cut_array[None,:]]
        self.nbr_shift_of_cell_sc = np.einsum('ni, ij -> nj', self.cell_shift_array_reduced, self.graph_data.latt) # shape: (ncells, 3)
        if self.apply_correction:
            # The mapping from orbital index to atomic index
            self.orb2atomidx = get_orb2atomidx(self.nao_max, self.graph_data.species, Ham_type=self.Ham_type)
            # Looking for the cell_shift after expanding.        
            cell_shift_array_expand = []
            nbr_range = 9
            ncells = len(self.graph_data.cell_shift_array)
            for i in range(-nbr_range, nbr_range+1):
                for j in range(-nbr_range, nbr_range+1):
                    for k in range(-nbr_range, nbr_range+1):
                        if (i,j,k) not in self.cell_index_map:         
                            cell_shift_array_expand.append([i,j,k])
    
            self.cell_shift_array_expand = np.array(cell_shift_array_expand) # shape: (ncells_expand, 3)
            self.ncells_expand = ncells + len(self.cell_shift_array_expand)
            self.nbr_shift_of_cell_expand = np.einsum('ni, ij -> nj', self.cell_shift_array_expand, self.graph_data.latt) # shape: (ncells_expand, 3)
            
            self.n_list_1 = [] # usage: for i, n in enumerate(n_list_1[m])
            self.relative_cell_mn_list_1 = [] # usage: relative_cell_mn = relative_cell_mn_list_1[m][i]
            for m, Cm in enumerate(self.graph_data.cell_shift_array): # ncells
                tmp_list = []
                tmp_relative_list = []
                for n, Cn in enumerate(self.graph_data.cell_shift_array): # ncells
                    relative_cell_shift = tuple((Cn - Cm).tolist())
                    if relative_cell_shift in self.cell_index_map:
                        relative_cell_mn = self.cell_index_map[relative_cell_shift] 
                        tmp_list.append(n)
                        tmp_relative_list.append(relative_cell_mn)
                self.n_list_1.append(tmp_list)
                self.relative_cell_mn_list_1.append(tmp_relative_list)
                
            self.n_list_2 = [] # usage: for i, n in enumerate(n_list_2[m])
            self.relative_cell_mn_list_2 = [] # usage: relative_cell_mn = relative_cell_mn_list_2[m][i]
            for m, Cm in enumerate(self.cell_shift_array_expand): # ncells_expand
                tmp_list = []
                tmp_relative_list = []
                for n, Cn in enumerate(self.cell_shift_array_expand): # ncells_expand
                    relative_cell_shift = tuple((Cn - Cm).tolist())
                    if relative_cell_shift in self.cell_index_map:
                        relative_cell_mn = self.cell_index_map[relative_cell_shift] 
                        tmp_list.append(n)
                        tmp_relative_list.append(relative_cell_mn)
                self.n_list_2.append(tmp_list)
                self.relative_cell_mn_list_2.append(tmp_relative_list)
            
            self.n_list_3 = [] # usage: for i, n in enumerate(n_list_3[m])
            self.relative_cell_mn_list_3 = [] # usage: relative_cell_mn = relative_cell_mn_list_3[m][i]
            for m, Cm in enumerate(self.graph_data.cell_shift_array): # ncells
                tmp_list = []
                tmp_relative_list = []
                for n, Cn in enumerate(self.cell_shift_array_expand): # ncells_expand
                    relative_cell_shift = tuple((Cn - Cm).tolist())
                    if relative_cell_shift in self.cell_index_map:
                        relative_cell_mn = self.cell_index_map[relative_cell_shift] 
                        tmp_list.append(n)
                        tmp_relative_list.append(relative_cell_mn)
                self.n_list_3.append(tmp_list)
                self.relative_cell_mn_list_3.append(tmp_relative_list)

            self.n_list_4 = [] # usage: for i, n in enumerate(n_list_4[m])
            self.relative_cell_mn_list_4 = [] # usage: relative_cell_mn = relative_cell_mn_list_4[m][i]
            for m, Cm in enumerate(self.cell_shift_array_expand): # ncells_expand
                tmp_list = []
                tmp_relative_list = []
                for n, Cn in enumerate(self.graph_data.cell_shift_array): # ncells
                    relative_cell_shift = tuple((Cn - Cm).tolist())
                    if relative_cell_shift in self.cell_index_map:
                        relative_cell_mn = self.cell_index_map[relative_cell_shift] 
                        tmp_list.append(n)
                        tmp_relative_list.append(relative_cell_mn)
                self.n_list_4.append(tmp_list)
                self.relative_cell_mn_list_4.append(tmp_relative_list)
        
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
        k_vec = np.tensordot(k_grids, self.graph_data.lat_per_inv, axes=1)
        return k_vec

    def _car2frac(self, k_grids):
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
        # Irreducible k-points
        ird_grids = grid[np.unique(mapping)] / np.array(mesh, dtype=float) # (nk, 3)
        if not return_frac:
            k_vec = np.tensordot(ird_grids, self.graph_data.lat_per_inv, axes=1) # (nk, 3)
        else:
            k_vec = ird_grids
        # get k weight
        res = Counter(mapping) # this is a dict
        weight = []
        for i in np.unique(mapping):
            weight.append(res[i])
        weight = np.array(weight)
        weight = weight/np.sum(weight)
        if auxiliary_info:
            ir_ids = np.unique(mapping)
            ir_idx_dict = {}
            for i, id in enumerate(ir_ids):
                ir_idx_dict[id] = i
            grid2ir_idx = []
            for ir_gp_id in mapping:
                grid2ir_idx.append(ir_idx_dict[ir_gp_id])
            grid2ir_idx = np.array(grid2ir_idx)
            
            grid = grid / np.array(mesh, dtype=float)
            if not return_frac:
                grid = np.tensordot(grid, self.graph_data.lat_per_inv, axes=1) # (nk, 3)
            return k_vec, weight, grid, grid2ir_idx
        else:
            return k_vec, weight

    def _phonon_cal(self, q_grid):
        freq_grid = []
        phon_vecs = []
        q_grid = q_grid.reshape(-1, 3)
        for q in q_grid:
            dynmat = self.phonon.get_dynamical_matrix_at_q(q)
            eigvals, eigvecs = np.linalg.eigh(dynmat)
            eigvecs = eigvecs.T # shape: (nbranches, nbranches)
            
            freq = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real) # shape: (nbranches,)
            # eigen_vec_phon = eigvecs.reshape(-1, natoms, 3) # shape: (nbranches, natoms, 3)
            freq_grid.append(freq)
            phon_vecs.append(eigvecs)

        freq_grid = np.stack(freq_grid, axis=0) * Hamcts.PHONOPYtoHARTREE # shape: (nq, nbranches)
        phon_vecs= np.stack(phon_vecs, axis=0) # shape: (nq, nbranches, nbranches)
        return freq_grid, phon_vecs
    
    def _get_longitude_phonon_indice(self):
        return [2, 5]

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
        
    def EPC_cal_path(self, k_fix, q_paths, band_ini, band_fin, do_symm:bool=True):
        """
        Args:
            k_fix (list or np.ndarray): (3)
            q_paths (list or np.ndarray): (nqs, 3)
            band_ini (int): The band index of initial state, begin from 0.
            band_fin (int): The band index of final state, begin from 0.
            do_symm (bool): If True, do the average over degenerate state.

        Returns:
            epc_all (np.ndarray): # shape:(nq, nbranches) EPC in Hartree.
        """
        # calculate the phonon spectrum
        freq_grid, phon_vecs = self._phonon_cal(q_paths)
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
                
            # cal epc
            phase_kpq = np.exp(2j*np.pi*np.sum(self.nbr_shift_of_cell_sc*(k_fix+q)[None,:], axis=-1)) # shape: (ncells,)
            wave_coe1 = wave_k[0, band_ini] # shape: (norbs,)
            wave_coe2 = wave_kpq[0, band_fin] # shape: (norbs,)

            tmp1 = np.einsum('m,n -> mn', np.conj(wave_coe2), wave_coe1)
            # calculate epc
            for branch_idx in range(int(3*len(self.atomic_mass))):
                factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * freq[branch_idx]) # shape:(natoms,)
                tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*eigen_vec_phon[branch_idx], tmp1)
                
                epc = 0.0
                for i_m, m in enumerate(self.cell_cut_list): # ncells
                    for i_n, n in enumerate(self.cell_cut_list): # ncells 
                        epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])

                # Correction of long-range interactions
                if self.apply_correction and (np.linalg.norm(q) < self.q_cut):
                    epc_corr = self._dipole_correction(tmp1, k_fix, q, freq[branch_idx], eigen_vec_phon[branch_idx])
                else:
                    epc_corr = 0.0
                epc_all.append(epc + epc_corr)

        # shape:(nq, nbranches)
        epc_all = np.array(epc_all).reshape(len(q_paths), int(3*len(self.atomic_mass)))
        if do_symm:
            self._EPC_symmetrize(epc_all, freq_grid, is_path=True)
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
    
    def _dipole_correction_fast(self, wave_coe_tp, k_vec, q_vec, freq, phon_vec):
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
        atomic_mass = self.atomic_mass
        
        if np.allclose(q_vec, np.array([0.0,0.0,0.0])):
            G_vec = self._get_reciprocal_lattice_vectors(3,3,3, True) # (ng, 3)
        else:
            G_vec = self._get_reciprocal_lattice_vectors(3,3,3, False) # (ng, 3)
        tao_k = self.graph_data.pos # (natoms, 3)
        tao_k = tao_k[self.orb2atomidx] # (norbs, 3)

        phase1 = np.exp(-2j*np.pi*np.sum(self.graph_data.nbr_shift_of_cell*k_vec[None,:], axis=-1)) # shape: (ncells,)
        phase2 = np.exp(2j*np.pi*np.sum((tao_k[None,:,:]-self.graph_data.nbr_shift_of_cell[:,None,:])*q_vec[None,None,:], axis=-1)) # shape: (ncells,norbs)

        S_cell = self.graph_data.S_cell
        
        # P_cell = self.graph_data.P_cell
        # S_cell = S_cell-2j*np.pi*np.einsum('nijk, k->nij', P_cell, q_vec)
        inner_product = oe.contract('ij, m, mi, mij', wave_coe_tp, phase1, phase2, S_cell)
        
        temp1 = Hamcts.TWOPI*np.einsum('gi,kij,kj -> kg', q_vec[None,:]+G_vec, self.BECs, phon_vec) # shape: (natoms, ng)
        temp2 = Hamcts.TWOPI_SQUARE*np.einsum('gi,ij,gj -> g', q_vec[None,:]+G_vec, self.DL, q_vec[None,:]+G_vec) # shape: (ng,)
        temp3 = temp1/temp2[None,:] # shape: (natoms, ng)
        
        temp4 = (temp3*inner_product).sum(-1) # shape: (natoms,)
        temp5 = np.sqrt(1/(2.0*atomic_mass*freq)) # shape: (natoms,)
        
        ret = Hamcts.JFOURPI*(temp4*temp5).sum()/self.volume_uc
        
        return ret

    def _dipole_correction(self, wave_coe_tp, k_vec, q_vec, freq, phon_vec):
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
        atomic_mass = self.atomic_mass
        
        if np.allclose(q_vec, np.array([0.0,0.0,0.0])):
            G_vec = self._get_reciprocal_lattice_vectors(3,3,3, True) # (ng, 3)
        else:
            G_vec = self._get_reciprocal_lattice_vectors(3,3,3, False) # (ng, 3)
        # tao_k = self.graph_data.pos # (natoms, 3)
        
        # # Looking for the cell_shift after expanding.        
        # cell_shift_array_expand = []
        # nbr_range = 3
        # ncells = len(self.graph_data.cell_shift_array)
        # for i in range(-nbr_range, nbr_range+1):
        #     for j in range(-nbr_range, nbr_range+1):
        #         for k in range(-nbr_range, nbr_range+1):
        #             if (i,j,k) not in self.cell_index_map:         
        #                 cell_shift_array_expand.append([i,j,k])

        # cell_shift_array_expand = np.array(cell_shift_array_expand) # shape: (ncells_expand, 3)
        # ncells_expand = ncells + len(self.cell_shift_array_expand)

        phase1 = np.exp(-Hamcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*(k_vec+q_vec)[None,:], axis=-1)) # shape: (ncells,)
        phase2 = np.exp(Hamcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k_vec[None,:], axis=-1)) # shape: (ncells,)
        
        # nbr_shift_of_cell_expand = np.einsum('ni, ij -> nj', self.cell_shift_array_expand, self.graph_data.latt) # shape: (ncells_expand, 3)
        phase1_expand = np.exp(-Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_expand*(k_vec+q_vec)[None,:], axis=-1)) # shape: (ncells_expand,)
        phase2_expand = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_expand*k_vec[None,:], axis=-1)) # shape: (ncells_expand,)

        sum_s = 0.0 # shape:(norbs, norbs)
        # sum_r = 0.0 # shape:(norbs, norbs, natoms)
        
        S_cell = self.graph_data.S_cell
        
        for m in range(len(self.graph_data.cell_shift_array)): # ncells
            for i, n in enumerate(self.n_list_1[m]): # ncells
                relative_cell_mn = self.relative_cell_mn_list_1[m][i]
                sum_s += phase1[m]*phase2[n]*S_cell[relative_cell_mn]
        
        for m in range(len(self.cell_shift_array_expand)): # ncells_expand
            for i, n in enumerate(self.n_list_2[m]): # ncells_expand
                relative_cell_mn = self.relative_cell_mn_list_2[m][i]
                sum_s += phase1_expand[m]*phase2_expand[n]*S_cell[relative_cell_mn]

        for m in range(len(self.graph_data.cell_shift_array)): # ncells
            for i, n in enumerate(self.n_list_3[m]): # ncells_expand
                relative_cell_mn = self.relative_cell_mn_list_3[m][i]
                sum_s += phase1[m]*phase2_expand[n]*S_cell[relative_cell_mn]

        for m in range(len(self.cell_shift_array_expand)): # ncells_expand
            for i, n in enumerate(self.n_list_4[m]): # ncells
                relative_cell_mn = self.relative_cell_mn_list_4[m][i]
                sum_s += phase1_expand[m]*phase2[n]*S_cell[relative_cell_mn]
        
        # g->ng, i,j->(x,y,z), k->natoms, m/n -> norbs
        # for m, Cm in enumerate(self.graph_data.cell_shift_array): # ncells
        #     for n, Cn in enumerate(self.graph_data.cell_shift_array): # ncells
        #         relative_cell_shift = tuple((Cn - Cm).tolist())
        #         if relative_cell_shift in self.cell_index_map:
        #             relative_cell_mn = self.cell_index_map[relative_cell_shift] 
        #             # tmp = np.expand_dims(self.graph_data.P_cell[relative_cell_mn], axis=0) - self.graph_data.S_cell[relative_cell_mn][None,:,:,None]*tao_k[:,None,None,:] # (natoms, norbs, norbs, 3)
        #             # sum_r += phase1[m]*phase2[n]*2j*np.pi*np.einsum('gi, kmni -> kgmn', q_vec[None,:]+G_vec, tmp)
        #             sum_s += phase1[m]*phase2[n]*self.graph_data.S_cell[relative_cell_mn]
                    
        # for m, Cm in enumerate(cell_shift_array_expand): # ncells_expand
        #     for n, Cn in enumerate(cell_shift_array_expand): # ncells_expand
        #         relative_cell_shift = tuple((Cn - Cm).tolist())
        #         if relative_cell_shift in self.cell_index_map:
        #             relative_cell_mn = self.cell_index_map[relative_cell_shift]
        #             # tmp = np.expand_dims(self.graph_data.P_cell[relative_cell_mn], axis=0) - self.graph_data.S_cell[relative_cell_mn][None,:,:,None]*tao_k[:,None,None,:] # (natoms, norbs, norbs, 3)
        #             # sum_r += phase1_expand[m]*phase2_expand[n]*2j*np.pi*np.einsum('gi, kmni -> kgmn', q_vec[None,:]+G_vec, tmp)
        #             sum_s += phase1_expand[m]*phase2_expand[n]*self.graph_data.S_cell[relative_cell_mn]

        # for m, Cm in enumerate(self.graph_data.cell_shift_array): # ncells
        #     for n, Cn in enumerate(cell_shift_array_expand): # ncells_expand
        #         relative_cell_shift = tuple((Cn - Cm).tolist())
        #         if relative_cell_shift in self.cell_index_map:
        #             relative_cell_mn = self.cell_index_map[relative_cell_shift]
        #             # tmp = np.expand_dims(self.graph_data.P_cell[relative_cell_mn], axis=0) - self.graph_data.S_cell[relative_cell_mn][None,:,:,None]*tao_k[:,None,None,:] # (natoms, norbs, norbs, 3)
        #             # sum_r += phase1[m]*phase2_expand[n]*2j*np.pi*np.einsum('gi, kmni -> kgmn', q_vec[None,:]+G_vec, tmp)
        #             sum_s += phase1[m]*phase2_expand[n]*self.graph_data.S_cell[relative_cell_mn]
                    
        # for m, Cm in enumerate(cell_shift_array_expand): # ncells_expand
        #     for n, Cn in enumerate(self.graph_data.cell_shift_array): # ncells
        #         relative_cell_shift = tuple((Cn - Cm).tolist())
        #         if relative_cell_shift in self.cell_index_map:
        #             relative_cell_mn = self.cell_index_map[relative_cell_shift]
        #             # tmp = np.expand_dims(self.graph_data.P_cell[relative_cell_mn], axis=0) - self.graph_data.S_cell[relative_cell_mn][None,:,:,None]*tao_k[:,None,None,:] # (natoms, norbs, norbs, 3)
        #             # sum_r += phase1_expand[m]*phase2[n]*2j*np.pi*np.einsum('gi, kmni -> kgmn', q_vec[None,:]+G_vec, tmp)
        #             sum_s += phase1_expand[m]*phase2[n]*self.graph_data.S_cell[relative_cell_mn]
    
        # inner_product = np.einsum('mn, kgmn->kg', wave_coe_tp, sum_r+sum_s[None,None,:,:])/ncells_expand # shape: (natoms, ng)
        inner_product = np.einsum('mn, mn', wave_coe_tp, sum_s) / self.ncells_expand
        
        temp1 = Hamcts.TWOPI*np.einsum('gi,kij,kj -> kg', q_vec[None,:]+G_vec, self.BECs, phon_vec) # shape: (natoms, ng)
        temp2 = Hamcts.TWOPI_SQUARE*np.einsum('gi,ij,gj -> g', q_vec[None,:]+G_vec, self.DL, q_vec[None,:]+G_vec) # shape: (ng,)
        temp3 = temp1/temp2[None,:] # shape: (natoms, ng)
        
        temp4 = (temp3*inner_product).sum(-1) # shape: (natoms,)
        temp5 = np.sqrt(1/(2.0*atomic_mass*freq)) # shape: (natoms,)
        
        ret = Hamcts.JFOURPI * (temp4 * temp5).sum() / self.volume_uc
        
        return ret

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
        nbnd, nks = enks.shape
        result = 0.0
        for ik in range(nks):
            tmp = np.sum(wgauss((ene - enks[:,ik]) / degauss, ngauss))
            result = result + self.weight_k[ik] * tmp
        return result

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
            ks_exp = np.zeros((nbnd, nks))
            for ibnd in range(nbnd):
                for ik in range(nks):
                    arg = (enks[ibnd,ik] - evbm) / self.temperature
                    if arg < -self.maxarg:
                        ks_exp[ibnd,ik] = 0.0
                    else:
                        ks_exp[ibnd,ik] = np.exp(arg)
                    arg = (enks[ibnd,ik] - ecbm) / self.temperature
            eup = Hamcts.TENPM160
            elw = 1.0
            for i in range(self.fermi_maxiter):
                ef = np.sqrt(eup) * np.sqrt(elw)
                hole_density = 0.0
                for ibnd in range(iband_edge+1):
                    for ik in range(nks):
                        if ks_exp[ibnd,ik] * ef > Hamcts.TENPP60:
                            fnk = 0.0
                        else:
                            fnk = 1.0 / (ks_exp[ibnd,ik] * ef + 1.0)
                        hole_density += (1.0 - fnk) * self.weight_k[ik]
                hole_density *= self.inv_cell
                if np.abs(hole_density) < carrier_small_judge:
                    rel_err = -Hamcts.TENPP3
                else:
                    rel_err = (hole_density - np.abs(self.ncarrier)) / hole_density
                if rel_err < Hamcts.TENPM5:
                    efermi = evbm - (np.log(ef) * self.temperature)
                    break
                elif rel_err > Hamcts.TENPM5:
                    elw = ef
                else:
                    eup = ef
            for ibnd in range(iband_edge+1):
                for ik in range(nks):
                    fnk = fermi_weight(enks[ibnd,ik] - efermi, self.temperature)
                    carrier_density += (1.0 - fnk) * self.weight_k[ik]
        else:
            ecbm = self._get_ecbm(enks, iband_edge)
            if self.rank == 0:
                print("CBM = {} eV".format(ecbm * Hamcts.HARTREEtoEV))
            ks_expcb = np.zeros((nbnd, nks))
            for ibnd in range(nbnd):
                for ik in range(nks):
                    arg = (enks[ibnd,ik] - ecbm) / self.temperature
                    if arg > self.maxarg:
                        ks_expcb[ibnd,ik] = Hamcts.TENPP200
                    else:
                        ks_expcb[ibnd,ik] = np.exp(arg)
            eup = 1.0
            elw = Hamcts.TENPP80
            for i in range(self.fermi_maxiter):
                ef = np.sqrt(eup) * np.sqrt(elw)
                electron_density = 0.0
                for ibnd in range(iband_edge, nbnd):
                    for ik in range(nks):
                        if ks_expcb[ibnd,ik] * ef > Hamcts.TENPP60:
                            fnk = 0.0
                        else:
                            fnk = 1.0 / (ks_expcb[ibnd,ik] * ef + 1.0)
                        electron_density += fnk * self.weight_k[ik]
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
            for ibnd in range(iband_edge, nbnd):
                for ik in range(nks):
                    fnk = fermi_weight(enks[ibnd,ik] - efermi, self.temperature)
                    carrier_density += fnk * self.weight_k[ik]
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
        dos = 0.0
        for _, ekks in enumerate(enks):
            for ik, enk in enumerate(ekks):
                delta_f3 = w0gauss((enk - self.efermi) * self.inv_smeark, ngauss=1) * self.inv_smeark
                dos = dos + delta_f3 * self.weight_k[ik]
        return dos

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
                            tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*eigen_vec_phon[branch_idx], tmp1)
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
                eliashberg_spectrum_cal_helper_sparse(freq_grid[iq].copy(), self.atomic_mass.copy(),eigen_vec_phon.copy(), 
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
        nbands = len(band_indice)
        nks = len(k_grid)
        nqs = len(q_grid)

        longitude_branches = self._get_longitude_phonon_indice()

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
            eigen = eigen[band_indice, :]
            eigen_vecs = eigen_vecs[:, band_indice, :]
        else:
            eigen, eigen_vecs = np.empty((nbands, 0)), np.empty((0, nbands, self.norbs))

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='rate_cal')

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
        
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik, :]
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
                                    factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * abs(freq[branch_idx])) # shape:(natoms,)
                                    tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*eigen_vec_phon[branch_idx], tmp1)
                                    
                                    epc = 0.0
                                    for i_m, m in enumerate(self.cell_cut_list): # ncells
                                        for i_n, n in enumerate(self.cell_cut_list): # ncells  
                                            epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                                    
                                    # Correction of long-range interactions
                                    if apply_correction_for_this_q and (branch_idx in longitude_branches):
                                        epc_corr = self._dipole_correction(tmp1, k, q, abs(freq_grid[iq, branch_idx]), eigen_vec_phon[branch_idx])
                                    else:
                                        epc_corr = 0.0
                                    epc = epc + epc_corr
                                    delta_f1 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] + freq[branch_idx]) * self.inv_smearq) * self.inv_smearq
                                    delta_f2 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] - freq[branch_idx]) * self.inv_smearq) * self.inv_smearq
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                    rate_all[ibnd,ik_all] += g2_tmp * ((bose_qvs[branch_idx] + fermi_kpqs[jbnd]) * delta_f1 + 
                                                                    (bose_qvs[branch_idx] + 1.0 - fermi_kpqs[jbnd]) * delta_f2) * weights_q_local[iq] 
            if self.rank == 0:
                logger.step(send_rank+1)
        rate_all *= Hamcts.TWOPI
        # The rate_all of the whole q is obtained by allreducing the rate_all of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
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

        longitude_branches = self._get_longitude_phonon_indice()

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
                                    tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*eigen_vec_phon[branch_idx], tmp1)
                                    epc = 0.0
                                    for i_m, m in enumerate(self.cell_cut_list): # ncells
                                        for i_n, n in enumerate(self.cell_cut_list): # ncells  
                                            epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                                    
                                    # Correction of long-range interactions
                                    if apply_correction_for_this_q and branch_idx in longitude_branches:
                                        epc_corr = self._dipole_correction(tmp1, k, q, abs(freq[branch_idx]), eigen_vec_phon[branch_idx])
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

        longitude_branches = self._get_longitude_phonon_indice()

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
                                tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*eigen_vec_phon[branch_idx], tmp1)
                                
                                epc = 0.0
                                for i_m, m in enumerate(self.cell_cut_list): # ncells
                                    for i_n, n in enumerate(self.cell_cut_list): # ncells  
                                        epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                                
                                # Correction of long-range interactions
                                if apply_correction_for_this_q and (branch_idx in longitude_branches):
                                    epc_corr = self._dipole_correction(tmp1, k, q, abs(freq[branch_idx]), eigen_vec_phon[branch_idx])
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

        longitude_branches = self._get_longitude_phonon_indice()

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
            freq_grid = freq_grid[:, longitude_branches]
            nqs_local = len(freq_grid)
            phon_vecs = phon_vecs.reshape(nqs_local, nmodes, self.natoms, 3)[:,longitude_branches,:,:]
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
            eigen = eigen[band_indice, :]
            eigen_vecs = eigen_vecs[:, band_indice, :]
        else:
            eigen, eigen_vecs = np.empty((nbands, 0)), np.empty((0, nbands, self.norbs))

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='rate_cal_polar')

        ik_all = -1

        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik, :]
                for ibnd in range(nbands):
                    if eig_k[ibnd] > efocus_max:
                        rate_all[ibnd, ik_all] = np.inf
                        continue
                    for iq, q in enumerate(q_grid[self.rank]):
                        if np.linalg.norm(q) < self.q_cut:
                            # phonon spectrum
                            freq = freq_grid[iq]
                            eigen_vec_phon = phon_vecs[iq]
                            bose_qvs = bose_weight(freq, self.temperature)
                            # calculate the electronic info for k+q
                            kpq = k + q
                            eig_kpq, wave_kpq = self._elec_cal(kpq)
                            eig_kpq = eig_kpq[band_indice, 0]
                            wave_kpq = wave_kpq[0, band_indice, :]
                            match_table = self._get_match_table(eig_k, eig_kpq, freq)
                            fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                            # cal epc
                            for jbnd in range(nbands):
                                tmp1 = np.einsum('m,n -> mn', np.conj(wave_kpq[jbnd]), wave_k[ibnd])
                                for imode in range(len(longitude_branches)):
                                    if match_table[ibnd, jbnd, imode]:
                                        epc = self._dipole_correction(tmp1, k, q, abs(freq[imode]), eigen_vec_phon[imode])
                                        delta_f1 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] + freq[imode]) * self.inv_smearq) * self.inv_smearq
                                        delta_f2 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] - freq[imode]) * self.inv_smearq) * self.inv_smearq
                                        g2_tmp = np.abs(epc) * np.abs(epc)
                                        rate_all[ibnd, ik_all] += weights_q_local[iq] * g2_tmp * ((bose_qvs[imode] + fermi_kpqs[jbnd]) * delta_f1 + (bose_qvs[imode] + 1.0 - fermi_kpqs[jbnd]) * delta_f2)
            if self.rank == 0:
                logger.step(send_rank+1)

        rate_all *= Hamcts.TWOPI
        # The rate_all of the whole q is obtained by allreducing the rate_all of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
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

        longitude_branches = self._get_longitude_phonon_indice()
        all_branches = np.arange(nmodes)
        transverse_branches = []
        for imode in all_branches:
            if imode not in longitude_branches:
                transverse_branches.append(imode)
        transverse_branches = np.array(transverse_branches, dtype=int)

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
            eigen = eigen[band_indice, :]
            eigen_vecs = eigen_vecs[:, band_indice, :]
        else:
            eigen, eigen_vecs = np.empty((nbands, 0)), np.empty((0, nbands, self.norbs))

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='rate_cal_rmp')

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik, :]
                phase_k = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                for ibnd in range(nbands):
                    if eig_k[ibnd] > efocus_max:
                        rate_all[ibnd, ik_all] = np.inf
                        continue
                    for iq, q in enumerate(q_grid[self.rank]):
                        apply_correction_for_this_q = np.linalg.norm(q) < self.q_cut
                        # phonon spectrum
                        freq = freq_grid[iq]
                        eigen_vec_phon = phon_vecs[iq]
                        bose_qvs = bose_weight(freq, self.temperature)
                        # calculate the electronic info for k+q
                        kpq = k + q
                        eig_kpq, wave_kpq = self._elec_cal(kpq)
                        eig_kpq = eig_kpq[band_indice, 0]
                        wave_kpq = wave_kpq[0, band_indice, :]
                        phase_kpq = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                        match_table = self._get_match_table(eig_k, eig_kpq, freq)
                        fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                        # cal epc
                        for jbnd in range(nbands):
                            tmp1 = np.einsum('m,n -> mn', np.conj(wave_kpq[jbnd]), wave_k[ibnd])
                            # calculate epc for nonLO
                            for imode in transverse_branches:
                                if match_table[ibnd, jbnd, imode]:
                                    factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * abs(freq[imode])) # shape:(natoms,)
                                    tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*eigen_vec_phon[imode], tmp1)
                                    epc = 0.0
                                    for i_m, m in enumerate(self.cell_cut_list): # ncells
                                        for i_n, n in enumerate(self.cell_cut_list): # ncells  
                                            epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                    delta_f1 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] + freq[imode]) * self.inv_smearq) * self.inv_smearq
                                    delta_f2 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] - freq[imode]) * self.inv_smearq) * self.inv_smearq
                                    rate_all[ibnd, ik_all] += weights_q_local[iq] * g2_tmp * \
                                                              ((bose_qvs[imode] + fermi_kpqs[jbnd]) * delta_f1 + \
                                                               (bose_qvs[imode] + 1.0 - fermi_kpqs[jbnd]) * delta_f2)
                            # calculate epc for LO
                            for imode in longitude_branches:
                                if match_table[ibnd, jbnd, imode]:
                                    factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * abs(freq[imode])) # shape:(natoms,)
                                    tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*eigen_vec_phon[imode], tmp1)
                                    epc = 0.0
                                    for i_m, m in enumerate(self.cell_cut_list): # ncells
                                        for i_n, n in enumerate(self.cell_cut_list): # ncells  
                                            epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, self.grad_mat[i_m,i_n])
                                    # Correction of long-range interactions
                                    if apply_correction_for_this_q:
                                        epc_corr = self._dipole_correction(tmp1, k, q, abs(freq[imode]), eigen_vec_phon[imode])
                                        epc = epc + epc_corr
                                        g2_tmp = np.abs(epc) * np.abs(epc) - np.abs(epc_corr) * np.abs(epc_corr)
                                    else:
                                        g2_tmp = np.abs(epc) * np.abs(epc)
                                    delta_f1 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] + freq[imode]) * self.inv_smearq) * self.inv_smearq
                                    delta_f2 = w0gauss((eig_k[ibnd] - eig_kpq[jbnd] - freq[imode]) * self.inv_smearq) * self.inv_smearq
                                    rate_all[ibnd, ik_all] += weights_q_local[iq] * g2_tmp * \
                                                              ((bose_qvs[imode] + fermi_kpqs[jbnd]) * delta_f1 + \
                                                               (bose_qvs[imode] + 1.0 - fermi_kpqs[jbnd]) * delta_f2)
            if self.rank == 0:
                logger.step(send_rank+1)
        rate_all *= Hamcts.TWOPI
        # The rate_all of the whole q is obtained by allreducing the rate_all of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
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
        
        enks, _ = self._elec_cal(k_grid) # (nbandtots, nk)
        enks = enks[self.bands_indices, :] # (nbnd, nk)
        
        # carrier_density has been multiplied by unit cell volume
        self.efermi, self.carrier_density = self._get_fermi_level_insulator(enks, iband_edge)
        if self.rank == 0:
            print("Fermi energy = {} eV, Carrier density = {} cm^(-3).".format(
                self.efermi * Hamcts.HARTREEtoEV, self.carrier_density * self.inv_cell / (Hamcts.BOHRtoCM ** 3)
                ))
            
        ecbm = self._get_ecbm(enks, iband_edge)
        
        # k points are parallelized and k grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(len(grid_all)):
            split_sections[i%self.rank_size] += 1
        
        split_sections = np.cumsum(split_sections, axis=0)
        grid_all = np.split(grid_all, indices_or_sections=split_sections, axis=0)
        
        if grid_all[self.rank].size>0:
            # calculate the electron velocity in parallel
            elec_velocities = self.vel_nk_cal_from_HS(self.bands_indices, grid_all[self.rank]) # (nk_split, nbands, 3)
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
        
        if grid_all[self.rank].size>0:
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
        epcs = np.abs(self.EPC_cal_path(self.epc_path_fix_k, self.high_symmetry_k_vecs, 
                                                    self.dispersion_select_index[0], self.dispersion_select_index[1],
                                                    do_symm=False)) # shape: (nqs, nmodes)
        epcs = epcs * Hamcts.HARTREEtoMEV
        nqs, nmodes = epcs.shape
        fout = open(os.path.join(self.outdir, "epc.dispersion"), 'w')
        fout.write(f"# k_lable: {' '.join(self.high_symmetry_labels)}\n")
        fout.write(f"# k_node: { '  '.join([str(round(each, 10)) for each in self.high_symmetry_k_nodes]) }\n")
        for imode in range(nmodes):
            for iq in range(nqs):
                fout.write(f"{str(round(self.high_symmetry_k_dist[iq], 10))}    {str(round(epcs[iq, imode], 10))}\n")
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
                kpq = k + q
                _, wave_kpq = self._elec_cal(kpq)
                wave_kpq = wave_kpq[0, bands_indices, :]
                phase_kpq = np.exp(Hamcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                for ibnd in range(nbands):
                    for jbnd in range(nbands):
                        tmp1 = np.einsum('m,n -> mn', np.conj(wave_kpq[jbnd]), wave_k[ibnd])
                        for imode in range(nmodes):
                            factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * abs(freqs[imode])) # shape:(natoms,)
                            tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*phon_vecs[imode], tmp1)
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

