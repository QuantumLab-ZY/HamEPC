'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2023-01-14 11:07:06
LastEditors: Yang Zhong
LastEditTime: 2024-05-16 21:25:27
'''
from ctypes import Union
import numpy as np
from typing import Tuple, Union, List, Any
from scipy.special import erfc
from pymatgen.core.periodic_table import Element
import math
import time
import sys

################################################## Constants ##################################################
class Constants(object):
    def __init__(self):
        # numbers
        self.PI = math.pi
        self.TWOPI = 2.0 * math.pi
        self.TWOPI_SQUARE = self.TWOPI * self.TWOPI
        self.SQRT_0P5 = np.sqrt(0.5)
        self.SQRT_2 = np.sqrt(2.0)
        self.INV_SQRT_2 = 1.0 / np.sqrt(2.0)
        self.SQRT_INVPI = np.sqrt(1.0 / math.pi)
        self.TENPM160 = 1.0E-160
        self.TENPM80 = 1.0E-80
        self.TENPM10 = 1.0E-10
        self.TENPM5 = 1.0E-5
        self.TENPM3 = 1.0E-3
        self.TENPP3 = 1.0E3
        self.TENPP60 = 1.0E60
        self.TENPP80 = 1.0E80
        self.TENPP200 = 1.0E200
        self.JTWOPI = 2j * math.pi
        self.JFOURPI = 2.0 * self.JTWOPI        

        # physical constants
        self.AMU = 1.6605402e-27  # [kg]
        self.EV = 1.60217733e-19    # [J]
        self.MASS_E = 9.10938215e-31    # [kg]
        self.SPEED_LIGHT = 299792458    # [m/s]
        self.PERMEABILITY_VACUUM = 4.0e-7 * np.pi   # [Hartree/m]
        self.PERMITTIVITY_VACUUM = 1.0 / self.PERMEABILITY_VACUUM  / (self.SPEED_LIGHT**2)  # [(C^2)/N*(m^2)]
        self.PLANCK_CONSTANT_EV = 4.13566733e-15   # [eV s]
        self.HBAR_EV = self.PLANCK_CONSTANT_EV / (2.0 * np.pi)    # [eV s]
        self.HARTREE = self.MASS_E * self.EV / 16.0 / (np.pi**2) / (self.PERMITTIVITY_VACUUM**2) / (self.HBAR_EV**2)   # [eV]
        self.BOHR = 4.0e+10 * np.pi * self.PERMITTIVITY_VACUUM * (self.HBAR_EV**2) / self.MASS_E  # [angstrom]
        self.BOLTZMANN_CONSTANT_SI = 1.3806504e-23  # [J/K]
        self.BOLTZMANN_CONSTANT_EV = self.BOLTZMANN_CONSTANT_SI / self.EV   # [eV/K]

        # tranformation constants
        self.EVtoPS = self.HBAR_EV * 1.0E12
        self.EVtoS = self.HBAR_EV
        self.EVtoINVPS = 1.0 / self.EVtoPS
        self.EVtoINVS = 1.0 / self.EVtoS
        self.KELVINtoHARTREE = self.BOLTZMANN_CONSTANT_EV / self.HARTREE
        self.HARTREEtoKELVIN = 1.0 / self.KELVINtoHARTREE
        self.EVtoHARTREE = 1.0 / self.HARTREE
        self.MEVtoHARTREE = self.EVtoHARTREE / 1000.0   # MEV means millielectron volts here
        self.HARTREEtoEV = self.HARTREE
        self.HARTREEtoMEV = 1.0 / self.MEVtoHARTREE # MEV means millielectron volts here
        self.HARTREEtoJ = self.HARTREEtoEV * self.EV
        self.BOHRtoANG = self.BOHR
        self.ANGtoBOHR = 1.0 / self.BOHR
        self.HARTREEtoINVS = self.HARTREEtoEV * self.EVtoINVS
        self.HARTREEtoINVPS = self.HARTREEtoEV * self.EVtoINVPS
        self.MtoBOHR = 1.0E10 * self.ANGtoBOHR
        self.BOHRtoM = 1.0 / self.MtoBOHR
        self.CMtoBOHR = 1.0E8 * self.ANGtoBOHR
        self.BOHRtoCM = 1.0 / self.CMtoBOHR
        self.PHONOPYtoEV = self.HBAR_EV * math.sqrt(1.0E20 * self.EV / self.AMU) # Convert the phonopy internal unit of SQRT(eV / Ang^2 / AMU) to eV
        self.PHONOPYtoHARTREE = self.PHONOPYtoEV * self.EVtoHARTREE

        # lock the constant values
        self.LOCK = True

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            raise TypeError(f"{name} is a constant!")
        elif 'LOCK' in self.__dict__:
            raise TypeError(f"Cannot define constant while running!")
        else:
            self.__dict__[name] = value

Hamcts = Constants()

################################################## Time Logger ##################################################
class time_logger(object):
    def __init__(self, total_cycles, routine_name):
        self.scale = 50
        self.total_cycles = total_cycles
        self.start = time.perf_counter()
        self.last_time = time.perf_counter()
        self.routine_name = routine_name

    def step(self, current_cycle):
        if current_cycle == 0:
            print(f"Running {self.routine_name}".center(100,"-"))
        i = int(current_cycle/self.total_cycles*self.scale)
        a = "*" * i
        b = "." * (self.scale - i)        
        c = (current_cycle / self.total_cycles) * 100
        dur = time.perf_counter() - self.start
        time_cycle = time.perf_counter() - self.last_time
        self.last_time = time.perf_counter()
        remaining_time = time_cycle*(self.total_cycles-current_cycle)
        print("\r{:^3.0f}%[{}->{}][total: {:.2f}s, step: {:.2f}s, remaining time: {:.2f}s]".format(c,a,b,dur,time_cycle,remaining_time),end = "")
        if current_cycle == self.total_cycles:
            print("\n"+f"{self.routine_name} has run successfully!".center(100,"-"))
        sys.stdout.flush()

def _nice_float(x,just,rnd):
    return str(round(x,rnd)).rjust(just)

################################################## KPOINTS Generator ##################################################
class kpoints_generator:
    """
    Used to generate K point path
    """
    def __init__(self, dim_k: int=3, lat: Union[np.array, list]=None, per: Union[List, Tuple] = None):
        self._dim_k = dim_k
        self._lat = lat
        # choose which self._dim_k out of self._dim_r dimensions are
        # to be considered periodic.        
        if per==None:
            # by default first _dim_k dimensions are periodic
            self._per=list(range(self._dim_k))
        else:
            if len(per)!=self._dim_k:
                raise Exception("\n\nWrong choice of periodic/infinite direction!")
            # store which directions are the periodic ones
            self._per=per
        
    def k_path(self,kpts,nk,report=True):
    
        # processing of special cases for kpts
        if kpts=='full':
            # full Brillouin zone for 1D case
            k_list=np.array([[0.],[0.5],[1.]])
        elif kpts=='fullc':
            # centered full Brillouin zone for 1D case
            k_list=np.array([[-0.5],[0.],[0.5]])
        elif kpts=='half':
            # half Brillouin zone for 1D case
            k_list=np.array([[0.],[0.5]])
        else:
            k_list=np.array(kpts)
    
        # in 1D case if path is specified as a vector, convert it to an (n,1) array
        if len(k_list.shape)==1 and self._dim_k==1:
            k_list=np.array([k_list]).T

        # make sure that k-points in the path have correct dimension
        if k_list.shape[1]!=self._dim_k:
            print('input k-space dimension is',k_list.shape[1])
            print('k-space dimension taken from model is',self._dim_k)
            raise Exception("\n\nk-space dimensions do not match")

        # must have more k-points in the path than number of nodes
        if nk<k_list.shape[0]:
            raise Exception("\n\nMust have more points in the path than number of nodes.")

        # number of nodes
        n_nodes=k_list.shape[0]
    
        # extract the lattice vectors from the TB model
        lat_per=np.copy(self._lat)
        # choose only those that correspond to periodic directions
        lat_per=lat_per[self._per]    
        # compute k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per,lat_per.T))

        # Find distances between nodes and set k_node, which is
        # accumulated distance since the start of the path
        #  initialize array k_node
        k_node=np.zeros(n_nodes,dtype=float)
        for n in range(1,n_nodes):
            dk = k_list[n]-k_list[n-1]
            dklen = np.sqrt(np.dot(dk,np.dot(k_metric,dk)))
            k_node[n]=k_node[n-1]+dklen
    
        # Find indices of nodes in interpolated list
        node_index=[0]
        for n in range(1,n_nodes-1):
            frac=k_node[n]/k_node[-1]
            node_index.append(int(round(frac*(nk-1))))
        node_index.append(nk-1)
    
        # initialize two arrays temporarily with zeros
        #   array giving accumulated k-distance to each k-point
        k_dist=np.zeros(nk,dtype=float)
        #   array listing the interpolated k-points    
        k_vec=np.zeros((nk,self._dim_k),dtype=float)
    
        # go over all kpoints
        k_vec[0]=k_list[0]
        for n in range(1,n_nodes):
            n_i=node_index[n-1]
            n_f=node_index[n]
            kd_i=k_node[n-1]
            kd_f=k_node[n]
            k_i=k_list[n-1]
            k_f=k_list[n]
            for j in range(n_i,n_f+1):
                frac=float(j-n_i)/float(n_f-n_i)
                k_dist[j]=kd_i+frac*(kd_f-kd_i)
                k_vec[j]=k_i+frac*(k_f-k_i)
    
        if report==True:
            if self._dim_k==1:
                print(' Path in 1D BZ defined by nodes at '+str(k_list.flatten()))
            else:
                print('----- k_path report begin ----------')
                original=np.get_printoptions()
                np.set_printoptions(precision=5)
                print('real-space lattice vectors\n', lat_per)
                print('k-space metric tensor\n', k_metric)
                print('internal coordinates of nodes\n', k_list)
                if (lat_per.shape[0]==lat_per.shape[1]):
                    # lat_per is invertible
                    lat_per_inv=np.linalg.inv(lat_per).T
                    print('reciprocal-space lattice vectors\n', lat_per_inv)
                    # cartesian coordinates of nodes
                    kpts_cart=np.tensordot(k_list,lat_per_inv,axes=1)
                    print('cartesian coordinates of nodes\n',kpts_cart)
                print('list of segments:')
                for n in range(1,n_nodes):
                    dk=k_node[n]-k_node[n-1]
                    dk_str=_nice_float(dk,7,5)
                    print('  length = '+dk_str+'  from ',k_list[n-1],' to ',k_list[n])
                print('node distance list:', k_node)
                print('node index list:   ', np.array(node_index))
                np.set_printoptions(precision=original["precision"])
                print('----- k_path report end ------------')
            print()

        return (k_vec,k_dist,k_node,lat_per_inv)

################################################## Default Parameters ##################################################
default_parameters:dict[str, dict[str, Any]] = {
    'advanced': {
        'read_large_grad_mat': False,
        'split_orbits': False,
        'split_orbits_num_blocks': 6,
        'soc_switch': False,
    },
    'dispersion': {
        'high_symmetry_points': [],
        'high_symmetry_labels': [],
        'nks_path': 200,
        'dispersion_select_index': '',
        'epc_path_fix_k': [],
    },
    'phonon': {
        'supercell_matrix': [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        'primitive_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        'unitcell_filename': "./POSCAR",
        'force_sets_filename': "./FORCE_SETS",
        'apply_correction': False,
        'q_cut': 100.0,    # The q cut for dipole correction. We can have a test to determine the best q_cut.
        'BECs': [],
        'DL': []
    },
    'gradmat': {
        'graph_data_path_sc': "",
        'dSon_path': {},
        'perturb_cell_matrix': [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
    },
    'epc': {
        'grad_mat_path': "./grad_mat.npy",
        'mat_info_rc_path': "./mat_info_rc.npy",
        'cell_range_cut': [2,2,2],
    },
    'transport': {
        'k_size': [32,32,32],
        'q_size': [32,32,32],
        'bands_indices': [],
        'maxarg': 200,  # The cutoff for gaussain function
        'fermi_maxiter': 100,
        'temperature': 300.0,    # K
        'phonon_cutoff': 2.0,    # meV
        'smeark': 25,    # meV
        'smearq': 25,    # meV
        'gauss_type': 0,    # the type of gaussian function
        'e_thr': 75,  # meV
    },
    'mobility': {
        'read_momentum': False,   # When calculating the carrier mobility, read_momentum should be true.
        'over_cbm': 0.2,    # eV
        'MC_sampling': "none",
        'polar_split': "none",
        'cauchy_scale': 0.035,
        'sampling_seed': 1,
        'nsamples': 1000000,
        'ncarrier': 10000000000000000,  # 10^16 cm^-3
        'ishole': False,
        'mob_level': "ERTA", # 'ERTA'
        'polar_rate_path': '',
        'rmp_rate_path': '',
    },
    'superconduct': {
        'mius': [0.05, 0.25, 0.01],
        'omega_range': [0, 100], # meV
        'omega_step': 0.01,  # meV
        'eliashberg': False,
        'anisotropy': False,
        'matsubara_cutoff': 10, # phonon cutoff frequency
        'eff_Coul_pot': 0.15,
        'BCS_ratio': 1.76,
        'eliashberg_iter_max': 40,
        'eliashberg_iter_thr': 0.00001,
        'ME_solver_restart': False,
    },
    'breakdown': {
        'fbd_erange': [0, 1, 0.01], # eV [emin, emax, estep]
        'fbd_rate_file': '',
        'fbd_effective_mass': 1.0,
        'fbd_exp_bandgap': 0.0,
    },
}

################################################## Species Information ##################################################
# Warning: this dict is not complete!!!
spin_set = {'H':[0.5, 0.5],
            'He':[1.0,1.0],
            'Li':[1.5,1.5],
            'Be':[1.0,1.0],
            'B':[1.5,1.5], 
            'C':[2.0, 2.0], 
            'N': [2.5,2.5], 
            'O':[3.0,3.0], 
            'F':[3.5,3.5],
            'Ne':[4.0,4.0],
            'Na':[4.5,4.5],
            'Mg':[4.0,4.0],
            'Al':[1.5,1.5],
            'Si':[2.0,2.0],
            'P':[2.5,2.5],
            'S':[3.0,3.0],
            'Cl':[3.5,3.5],
            'Ar':[4.0,4.0],
            'K':[4.5,4.5],
            'Ca':[5.0,5.0],
            'Si': [2.0, 2.0], 
            'Mo': [7.0, 7.0], 
            'S': [3.0, 3.0],
            'Bi':[7.5, 7.5],
            'Se':[3.0, 3.0],
            'Cr':[9.0, 5.0],
            'I':[3.5, 3.5],
            'As':[7.5, 7.5],
            'Ga':[6.5, 6.5],
            'Mo': [7.0, 7.0],
            'Cs':[4.5, 4.5],
            'Pb':[7.0, 7.0],
            'Te':[8.0, 8.0],
            'Hg':[9.0, 9.0],
            'V':[6.5, 6.5],
            'Sb':[7.5, 7.5],
            'Ge':[2.0,2.0]
            }

# Warning: this dict is not complete!!!
PAO_dict = {'H':'H6.0-s2p1',
            'He':'He8.0-s2p1',
            'Li':'Li8.0-s3p2',
            'Be':'Be7.0-s2p2',
            'B':'B7.0-s2p2d1',
            'C':'C6.0-s2p2d1',
            'N':'N6.0-s2p2d1',
            'O':'O6.0-s2p2d1',
            'F':'F6.0-s2p2d1',
            'Ne':'Ne9.0-s2p2d1',
            'Na':'Na9.0-s3p2d1',
            'Mg':'Mg9.0-s3p2d1',
            'Al':'Al7.0-s2p2d1',
            'Si':'Si7.0-s2p2d1',
            'P':'P7.0-s2p2d1',
            'S':'S7.0-s2p2d1',
            'Cl':'Cl7.0-s2p2d1',
            'Ar':'Ar9.0-s2p2d1',
            'K':'K10.0-s3p2d1',
            'Ca':'Ca9.0-s3p2d1',
            'Bi':'Bi8.0-s3p2d2',
            'Se':'Se7.0-s3p2d2',
            'Cr':'Cr6.0-s3p2d1',
            'I':'I7.0-s3p2d2',
            'As':'As7.0-s3p2d2',
            'Ga':'Ga7.0-s3p2d2',
            'Mo':'Mo7.0-s3p2d2',
            'Cs':'Cs12.0-s3p2d2',
            'Pb':'Pb8.0-s3p2d2',
            'Te': 'Te7.0-s3p2d2',
            'Hg':'Hg8.0-s3p2d2',
            'V': 'V6.0-s3p2d1',
            'Sb': 'Sb7.0-s3p2d2',
            'Ge': 'Ge7.0-s3p2d2'
            }

# Warning: this dict is not complete!!!
PBE_dict = {'H':'H_PBE19',
            'He':'He_PBE19',
            'Li':'Li_PBE19',
            'Be':'Be_PBE19',
            'B':'B_PBE19',
            'C':'C_PBE19',
            'N':'N_PBE19',
            'O':'O_PBE19',
            'F':'F_PBE19',
            'Ne':'Ne_PBE19',
            'Na':'Na_PBE19',
            'Mg':'Mg_PBE19',
            'Al':'Al_PBE19',
            'Si':'Si_PBE19',
            'P':'P_PBE19',
            'S':'S_PBE19',
            'Cl':'Cl_PBE19',
            'Ar':'Ar_PBE19',
            'K':'K_PBE19',
            'Ca':'Ca_PBE19',
            'Bi':'Bi_PBE19',
            'Se':'Se_PBE19',
            'Cr':'Cr_PBE19',
            'I':'I_PBE19',
            'As':'As_PBE19',
            'Ga':'Ga_PBE19',
            'Mo':'Mo_PBE19',
            'Cs':'Cs_PBE19',
            'Pb':'Pb_PBE19',
            'Te':'Te_PBE19',
            'Hg':'Hg_PBE19',
            'V':'V_PBE19',
            'Sb':'Sb_PBE19',
            'Ge':'Ge_PBE19'}

basis_def_19 = {
             1:np.array([0,1,3,4,5], dtype=int), # H
             2:np.array([0,1,3,4,5], dtype=int), # He
             3:np.array([0,1,2,3,4,5,6,7,8], dtype=int), # Li
             4:np.array([0,1,3,4,5,6,7,8], dtype=int), # Be
             5:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # B
             6:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # C
             7:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # N
             8:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # O
             9:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # F
             10:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ne
             11:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Na
             12:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Mg
             13:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Al
             14:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Si
             15:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # p
             16:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # S
             17:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Cl
             18:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ar
             19:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # K
             20:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ca 
             42:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Mo  
             83:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Bi  
             34:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Se 
             24:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Cr 
             53:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # I   
             82:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # pb
             55:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Cs
             31:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Ga
             33:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # As
             80:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Hg
             Element['V'].Z: np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # V
             Element['Sb'].Z: np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Sb
             Element['Ge'].Z: np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Sb
             }

basis_def_14 = {1:np.array([0,1,3,4,5], dtype=int), # H
             2:np.array([0,1,3,4,5], dtype=int), # He
             3:np.array([0,1,2,3,4,5,6,7,8], dtype=int), # Li
             4:np.array([0,1,3,4,5,6,7,8], dtype=int), # Be
             5:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # B
             6:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # C
             7:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # N
             8:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # O
             9:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # F
             10:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ne
             11:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Na
             12:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Mg
             13:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Al
             14:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Si
             15:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # p
             16:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # S
             17:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Cl
             18:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ar
             19:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # K
             20:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ca 
             Element['V'].Z: np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # V
             }

basis_def_13_siesta = (lambda s1=[0],s2=[1],p1=[2,3,4],p2=[5,6,7],d1=[8,9,10,11,12]: {
    1 : np.array(s1+s2+p1, dtype=int), # H
    2 : np.array(s1+s2+p1, dtype=int), # He
    3 : np.array(s1+s2+p1, dtype=int), # Li
    4 : np.array(s1+s2+p1, dtype=int), # Be
    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
    11: np.array(s1+s2+p1, dtype=int), # Na
    12: np.array(s1+s2+p1, dtype=int), # Mg
    13: np.array(s1+s2+p1+p2+d1, dtype=int), # Al
    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
    19: np.array(s1+s2+p1, dtype=int), # K
    20: np.array(s1+s2+p1, dtype=int), # Cl
    33: np.array(s1+s2+p1+p2+d1, dtype=int), # As
    31: np.array(s1+s2+p1+p2+d1, dtype=int), # Ga
})()

basis_def_19_siesta = (lambda s1=[0],s2=[1],s3=[2],p1=[3,4,5],p2=[6,7,8],d1=[9,10,11,12,13],d2=[14,15,16,17,18]: {
    1 : np.array(s1+s2+p1, dtype=int), # H
    2 : np.array(s1+s2+p1, dtype=int), # He
    3 : np.array(s1+s2+p1, dtype=int), # Li
    4 : np.array(s1+s2+p1, dtype=int), # Be
    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
    11: np.array(s1+s2+p1, dtype=int), # Na
    12: np.array(s1+s2+p1, dtype=int), # Mg
    13: np.array(s1+s2+p1+p2+d1, dtype=int), # Al
    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
    19: np.array(s1+s2+p1, dtype=int), # K
    20: np.array(s1+s2+p1, dtype=int), # Cl
    22: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Ti, created by Qin.
})()

num_valence_openmx = {1:1,2:2,3:3,4:2,5:3,6:4,7:5,8:6,9:7,10:8,11:9,12:8,13:3,14:4,15:5,16:6,17:7,18:8,19:9,20:10,42:14,83:15,34:6,
               35:7,55:9,82:14,53:7,33:15,80:18,31:13,Element['V'].Z:13,Element['Sb'].Z:15, Element['Ge'].Z:4}
num_val_openmx = np.zeros((99,), dtype=int)
for k in num_valence_openmx.keys():
    num_val_openmx[k] = num_valence_openmx[k]

num_valence_siesta = {
    1:1,2:2,
    3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,
    11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,
    19:1,20:2,22:12,31:3,33:5,
}
num_val_siesta = np.zeros((99,), dtype=int)
for k in num_valence_siesta.keys():
    num_val_siesta[k] = num_valence_siesta[k]

################################################## Gaussian Functions ##################################################
# ref from QE Modules/erf.f90
def gauss_freq(x):
    return 0.5 * erfc(-x * Hamcts.SQRT_0P5)

# ref from QE Modules/wgauss.f90
def wgauss(x, ngauss:int=0):
    if ngauss == 0:
        return gauss_freq(x*Hamcts.SQRT_2)

# ref from QE Modules/w0gauss.f90
def w0gauss(x, ngauss:int=0):
    result = 0.0
    if ngauss == -99:
        # Fermi-Dirac smearing
        if np.abs(x) < 36.0:
            result = 1.0 / (2.0 + np.exp(-x) + np.exp(x))
    elif ngauss == -1:
        # cold smearing  (Marzari-Vanderbilt-DeVita-Payne)
        arg = np.fmin(200.0, (x - Hamcts.INV_SQRT_2) * (x - Hamcts.INV_SQRT_2))
        result = Hamcts.SQRT_INVPI * np.exp (-arg) * (2.0 - Hamcts.SQRT_2 * x)
    elif ngauss > 10 or ngauss < 0:
        print('Higher order smearing of {} is untested and unstable'.format(np.abs(ngauss)))
    else:
        # Methfessel-Paxton
        arg = np.fmin(200.0, x * x)
        result = np.exp(-arg) * Hamcts.SQRT_INVPI
        if ngauss == 0:
            return result
        else:
            hd = 0.0
            hp = np.exp(-arg)
            ni = 0
            a = Hamcts.SQRT_INVPI
            for i in range(ngauss):
                hd = 2.0 * x * hp - 2.0 * ni * hd
                ni = ni + 1
                a = - a / ((i + 1) * 4.0)
                hp = 2.0 * x * hd - 2.0 * ni * hp
                ni = ni + 1
                result = result + a * hp

        return result

################################################## Distribution Functions ##################################################
# Here E means E - \miu 
def fermi_weight(E, kbT):
    return 1.0 / (np.exp(E / kbT) + 1.0)

def minus_dfermi(E, kbT):
    return fermi_weight(E, kbT) * fermi_weight(-E, kbT) / kbT

def bose_weight(E, kbT):
    return 1.0 / (np.exp(E / kbT) - 1.0)

################################################## Random Functions ##################################################
# ref from Perturbo pert_utils.f90
def random_uniform(nvec:int, random_seed:int=0):
    """
    Get a set of vectors which meet the uniform distribution.

    args:
        nvec (int): The number of vectors.
        random_seed (int): The random seed.

    returns:
        vecs (np.ndarray): The random vectors. # shape: (nvec, 3)
        weights (np.ndarray): The weights corresponding to the vecs. # shape: (nvec, )
    """
    np.random.seed(random_seed)
    vecs = np.random.rand(nvec, 3)
    weights = np.ones(nvec) / float(nvec)
    return vecs, weights

# ref from Perturbo pert_utils.f90
def random_cauchy(nvec:int, cauchy_scale:float=0.05, random_seed:int=0):
    """
    Get a set of vectors which meet the cauchy distribution.

    args:
        nvec (int): The number of vectors.
        cauchy_scale (float): The scale parameter sigma in cauchy distribution.
        random_seed (int): The random seed.

    returns:
        vecs (np.ndarray): The random vectors. # shape: (nvec, 3)
        weights (np.ndarray): The weights corresponding to the vecs. # shape: (nvec, )
    """
    
    vecs = np.zeros((nvec, 3))
    weights = np.zeros(nvec)

    # set seed
    np.random.seed(random_seed)
    pa = 0.5 + np.arctan(-0.5 / cauchy_scale) / np.pi
    pb = 0.5 + np.arctan(0.5 / cauchy_scale) / np.pi
    wtmp = 1.0 / nvec
    for i in range(nvec):
        temp = np.random.rand(3)
        temp = pa + temp * (pb - pa)
        vecs[i, :] = cauchy_scale * np.tan((temp - 0.5) * np.pi)
        temp = ((vecs[i, :] / cauchy_scale) ** 2 + 1.0) * cauchy_scale * np.pi * (pb - pa)
        weights[i] = wtmp
        for j in range(3):
            weights[i] = weights[i] * temp[j]

    return vecs, weights

################################################## Sparse Matrix ##################################################
def build_sparse_matrix_soc(species, cell_shift, nao_max, Hon, Hoff, iHon, iHoff, edge_index, Ham_type:str='openmx', return_raw_mat:bool=False):
    """
    Args:
        cell_shift (_type_): _description_
        natoms (_type_): _description_
        nao_max (_type_): _description_
        Hon (_type_): _description_
        Hoff (_type_): _description_
        edge_index (_type_): _description_
    """
    # Initialize cell index
    cell_shift_tuple = list(map(lambda x: tuple(x), cell_shift.tolist())) # len: (nedges,)
    cell_shift_set = set(cell_shift_tuple) # len: (ncells,)
    cell_shift_list = sorted(list(cell_shift_set)) # len: (ncells,)
    cell_shift_array = np.array(cell_shift_list) # shape: (ncells, 3)
    cell_index = [cell_shift_list.index(icell) for icell in cell_shift_tuple] # len: (nedges,)
    ncells = len(cell_shift_list)
    
    cell_index_map = dict()    
    for i, icell in enumerate(cell_shift_list):
        cell_index_map[icell] = i
    
    inv_cell_shift = map(lambda x: tuple(x), (-cell_shift_array).tolist())
    inv_cell_index = [cell_index_map[inv_cell] for inv_cell in inv_cell_shift] # len: (ncells,)

    # soc build
    natoms = len(species)
    na = np.arange(natoms)
    H_cell = np.zeros((ncells, natoms, natoms, 2*nao_max, 2*nao_max), dtype=complex)
    H_cell[cell_index, edge_index[0], edge_index[1], :, :] = (Hoff + 1.0j*iHoff).reshape(-1, 2*nao_max, 2*nao_max)  
    H_cell[cell_index_map[(0,0,0)], na, na, :, :] = (Hon + 1.0j*iHon).reshape(-1, 2*nao_max, 2*nao_max)
    
    H_cell = np.swapaxes(H_cell, 2, 3) # shape: (ncells, natoms, 2*nao_max, natoms, 2*nao_max)
    
    if return_raw_mat:
        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index) 
    else:
        H_cell = H_cell.reshape(-1, 2*natoms*nao_max, 2*natoms*nao_max)
        
        # parse the Atomic Orbital Basis Sets
        basis_definition = np.zeros((99, nao_max))
        # key is the atomic number, value is the index of the occupied orbits.
        if Ham_type.lower() == 'openmx':
            if nao_max == 14:
                basis_def = basis_def_14
            elif nao_max == 19:
                basis_def = basis_def_19
            else:
                raise NotImplementedError
        elif Ham_type.lower() in ['siesta', 'honpas']:
            if nao_max == 13:
                basis_def = basis_def_13_siesta
            elif nao_max == 19:
                basis_def = basis_def_19_siesta
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
        for k in basis_def.keys():
            basis_definition[k][basis_def[k]] = 1
          
        orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
        orb_mask = np.concatenate([orb_mask, orb_mask], axis=0) # shape: [2*natoms*nao_max] 
        orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [2*natoms*nao_max, 2*natoms*nao_max]
        
        H_cell = H_cell[:, orb_mask > 0]
        norbs = int(math.sqrt(H_cell.size/ncells))
        H_cell = H_cell.reshape(-1, norbs, norbs)
        
        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index)

def build_sparse_matrix(species, cell_shift, nao_max, Hon, Hoff, edge_index, Ham_type:str='openmx', return_raw_mat:bool=False):
    """
    Args:
        cell_shift (_type_): _description_
        natoms (_type_): _description_
        nao_max (_type_): _description_
        Hon (_type_): _description_
        Hoff (_type_): _description_
        edge_index (_type_): _description_
    """
    # Initialize cell index
    cell_shift_tuple_edge = list(map(lambda x: tuple(x), cell_shift.tolist())) # len: (nedges,)
    if (0,0,0) not in cell_shift_tuple_edge:
        cell_shift_tuple = [(0,0,0)]+cell_shift_tuple_edge
    else:
        cell_shift_tuple = cell_shift_tuple_edge
    cell_shift_set = set(cell_shift_tuple) # len: (ncells,)
    cell_shift_list = sorted(list(cell_shift_set)) # len: (ncells,)
    cell_shift_array = np.array(cell_shift_list) # shape: (ncells, 3)
    cell_index = [cell_shift_list.index(icell) for icell in cell_shift_tuple_edge] # len: (nedges,)
    ncells = len(cell_shift_list)
    
    cell_index_map = dict()    
    for i, icell in enumerate(cell_shift_list):
        cell_index_map[icell] = i
    
    inv_cell_shift = map(lambda x: tuple(x), (-cell_shift_array).tolist())
    inv_cell_index = [cell_index_map[inv_cell] for inv_cell in inv_cell_shift] # len: (ncells,)

    natoms = len(species)
    na = np.arange(natoms)
    H_cell = np.zeros((ncells, natoms, natoms, nao_max, nao_max))
    H_cell[cell_index, edge_index[0], edge_index[1], :, :] = Hoff.reshape(-1, nao_max, nao_max)  
    H_cell[cell_index_map[(0,0,0)], na, na, :, :] = Hon.reshape(-1, nao_max, nao_max)
    
    H_cell = np.swapaxes(H_cell, 2, 3) # shape: (ncells, natoms, nao_max, natoms, nao_max)
    
    if return_raw_mat:
        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index) 
    else:
        H_cell = H_cell.reshape(-1, natoms*nao_max, natoms*nao_max)
        # parse the Atomic Orbital Basis Sets
        basis_definition = np.zeros((99, nao_max))
        # key is the atomic number, value is the index of the occupied orbits.
        if Ham_type.lower() == 'openmx':
            if nao_max == 14:
                basis_def = basis_def_14
            elif nao_max == 19:
                basis_def = basis_def_19
            else:
                raise NotImplementedError
        elif Ham_type.lower() in ['siesta', 'honpas']:
            if nao_max == 13:
                basis_def = basis_def_13_siesta
            elif nao_max == 19:
                basis_def = basis_def_19_siesta
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
        for k in basis_def.keys():
            basis_definition[k][basis_def[k]] = 1
          
        orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
        orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [natoms*nao_max, natoms*nao_max]
        
        H_cell = H_cell[:, orb_mask > 0]
        norbs = int(math.sqrt(H_cell.size/ncells))
        H_cell = H_cell.reshape(-1, norbs, norbs)
        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index)

def build_reciprocal_from_sparseMat(H_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (ncells, norbs, norbs)
    """
    
    phase = np.exp(2j*np.pi*np.sum(nbr_shift_of_cell[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, ncells)
    HK = np.einsum('ijk, ni->njk', H_cell, phase) # (nk, norbs, norbs,)
    
    return HK

def build_reciprocal_from_sparseMat3(M_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (ncells, norbs, norbs, 3)
    """
    
    phase = np.exp(2j*np.pi*np.sum(nbr_shift_of_cell[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, ncells)
    MK = 1.0j*np.einsum('ijkl, ni->njkl', M_cell, phase) # (nk, norbs, norbs, 3)
    
    return MK

def build_reciprocal_from_sparseMat_soc(H_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (ncells, 2*norbs, 2*norbs)
    """
    
    phase = np.exp(2j*np.pi*np.sum(nbr_shift_of_cell[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, ncells)    
    norbs = int(H_cell.shape[1]/2)
    Hsoc = [H_cell[:,:norbs,:norbs], H_cell[:,:norbs,norbs:], H_cell[:,norbs:,:norbs], H_cell[:,norbs:,norbs:]]
    
    HK_list = []
    for H in Hsoc:
        HK = np.einsum('ijk, ni->njk', H, phase) # (nk, norbs, norbs,)
        HK_list.append(HK)

    HK = np.block([[HK_list[0],HK_list[1]],[HK_list[2],HK_list[3]]])
    
    return HK

################################################## Split Orbits ##################################################
def split_array_along_2axes(arr, axis1, axis2, sizes1, sizes2):
    """
    将一个多维数组按照指定的两个维度分成若干不同大小的子数组
    
    参数：
    arr (numpy.ndarray): 输入的6维数组
    axis1 (int): 第一个指定的分割维度
    axis2 (int): 第二个指定的分割维度
    sizes1 (list): 在第一个维度上的分割大小列表
    sizes2 (list): 在第二个维度上的分割大小列表
    
    返回：
    list of numpy.ndarray: 分割后的子数组列表
    """
    result = []
    split1 = np.split(arr, indices_or_sections = sizes1, axis=axis1)
    for s1 in split1:
        split2 = np.split(s1, indices_or_sections = sizes2, axis=axis2)
        result.append(split2)
    return result

################################################## EPC Long Range Correction ##################################################
def get_orb2atomidx(nao_max, species, Ham_type:str='openmx'):
    # key is the atomic number, value is the index of the occupied orbits.
    if Ham_type.lower() == 'openmx':
        if nao_max == 14:
            basis_def = basis_def_14
        elif nao_max == 19:
            basis_def = basis_def_19
        else:
            raise NotImplementedError
    elif Ham_type.lower() in ['siesta', 'honpas']:
        if nao_max == 13:
            basis_def = basis_def_13_siesta
        elif nao_max == 19:
            basis_def = basis_def_19_siesta
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    orb2atom_idx = []
    for ia, z in enumerate(species):
        orb2atom_idx += [ia]*len(basis_def[z])
    
    return np.array(orb2atom_idx)

if __name__ == '__main__':
    cos = Constants()
    print(cos.BOHRtoANG * 7.6099271072186741340646327724066)

            
    
