'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2023-02-22 20:34:28
LastEditors: Yang Zhong
LastEditTime: 2023-11-16 09:25:35
'''
import numpy as np
import os
import collections
from utils import (read_openmx_dat, atoms_dict_to_openmxfile, au2ang, save_dict_by_numpy,
                                        spin_set, PAO_dict, PBE_dict)
import collections
from phonopy.interface.vasp import Vasprun, read_vasp, write_vasp
from phonopy.structure.atoms import atom_data, symbol_map
import copy

######################## Input parameters begin #######################
unitcell_dir = "./CVS.vasp" # The path of poscar or cif files
supercell_path = './perturb/' # openmx file directory to save
Nrange1, Nrange2, Nrange3 = 1, 1, 1
delta_r = 0.01
system_name = 'CVS'
mat_info_path = './perturb/'
use_central_difference = False
######################## Input parameters end #########################

################################ OPENMX calculation parameters Set begin #####################
openmx_basic_commad = f"""#
#      File Name      
#

System.CurrrentDirectory         ./    # default=./
System.Name                      {system_name}
DATA.PATH                       ../DFT_DATA19
level.of.stdout                   1    # default=1 (1-3)
level.of.fileout                  1    # default=1 (0-2)
HS.fileout                   on       # on|off, default=off

#
# SCF or Electronic System
#

scf.XcType                  GGA-PBE    # LDA|LSDA-CA|LSDA-PW|GGA-PBE
scf.partialCoreCorrection   on 
scf.SpinPolarization        off        # On|Off|NC
scf.ElectronicTemperature  300.0       # default=300 (K)
scf.energycutoff           200.0       # default=150 (Ry)
scf.maxIter                 300         # default=40
scf.EigenvalueSolver       Band    # DC|GDC|Cluster|Band
scf.Kgrid                  6 6 6       # means 4x4x4
scf.Mixing.Type           rmm-diis     # Simple|Rmm-Diis|Gr-Pulay|Kerker|Rmm-Diisk
scf.Init.Mixing.Weight     0.0010      # default=0.30 
scf.Min.Mixing.Weight      0.0001      # default=0.001 
scf.Max.Mixing.Weight      0.3000      # default=0.40 
scf.Mixing.History           50
scf.Mixing.StartPulay        30
scf.Mixing.EveryPulay        1
scf.criterion             1.0e-14      # default=1.0e-6 (Hartree)


#
# MD or Geometry Optimization
#

MD.Type                      Nomd        # Nomd|Opt|NVE|NVT_VS|NVT_NH
                                       # Constraint_Opt|DIIS2|Constraint_DIIS2
MD.Opt.DIIS.History          4
MD.Opt.StartDIIS             5         # default=5
MD.maxIter                 100         # default=1
MD.TimeStep                1.0         # default=0.5 (fs)
MD.Opt.criterion          1.0e-4       # default=1.0e-4 (Hartree/bohr)


\n"""

################################ OPENMX calculation parameters Set end #######################

if not os.path.exists(supercell_path):
    os.mkdir(supercell_path)

# Read crystal structure of unit cell
if os.path.splitext(unitcell_dir)[1] == '.vasp':
    unitcell = read_vasp(unitcell_dir)
    coordinates = unitcell.get_positions()
    latt = unitcell.get_cell()
    atomic_nums = [symbol_map[s] for s in unitcell.symbols]
elif os.path.splitext(unitcell_dir)[1] == '.dat':
    atomic_nums, latt, coordinates = read_openmx_dat(filename = unitcell_dir)
    latt = latt*au2ang
    coordinates = coordinates*au2ang
else:
    pass

# expand cell and save supercell
sc_atoms_dict = dict()
sc_positions_dict = collections.OrderedDict()

for i in range(-Nrange1, Nrange1+1):
    for j in range(-Nrange2, Nrange2+1):
        for k in range(-Nrange3, Nrange3+1):
            for ia in range(len(atomic_nums)): # len: (natoms_uc,)
                # Use a dict to store the expanded coordinates of each specise separately
                if atomic_nums[ia] not in sc_positions_dict:
                    sc_positions_dict[atomic_nums[ia]] = []
                sc_positions_dict[atomic_nums[ia]].append(coordinates[ia]+i*latt[0]+j*latt[1]+k*latt[2])

sc_atoms_dict['chemical_symbols'] = [atom_data[z][1] for z in sc_positions_dict.keys() for j in range(len(sc_positions_dict[z]))]
sc_atoms_dict['positions'] = np.concatenate([sc_positions_dict[z] for z in sc_positions_dict.keys()], axis=0)
sc_atoms_dict['cell'] = np.array([2*Nrange1+1, 2*Nrange2+1, 2*Nrange3+1]).reshape(-1, 1)*latt
atoms_dict_to_openmxfile(sc_atoms_dict, openmx_basic_commad, spin_set, PAO_dict, PBE_dict, os.path.join(supercell_path, f'{system_name}_1.dat'))

# Initialize the mapping between supercell and unitcell
s2u_map = collections.OrderedDict()
cell_shift_map = collections.OrderedDict()
cell_index_map = collections.OrderedDict()
nbr_shift_of_cell = []

cell_index = 0
for i in range(-Nrange1, Nrange1+1):
    for j in range(-Nrange2, Nrange2+1):
        for k in range(-Nrange3, Nrange3+1):
            cell_index_map[(i,j,k)] = cell_index # index each cell_shift
            cell_index += 1
            nbr_shift_of_cell.append(i*latt[0]+j*latt[1]+k*latt[2])
            for ia in range(len(atomic_nums)): # len: (natoms_uc,)
                # store the atom index in unit cell and cell_shift for each species
                if atomic_nums[ia] not in s2u_map:
                    s2u_map[atomic_nums[ia]] = []
                    cell_shift_map[atomic_nums[ia]] = []
                s2u_map[atomic_nums[ia]].append(ia)
                cell_shift_map[atomic_nums[ia]].append((i,j,k))

nbr_shift_of_cell = np.array(nbr_shift_of_cell) # shape: (ncells, 3)
cell_shift_list = [] # len: (natoms_sc, 3)
s2u_list = [] # len: (natoms_sc,)
for z in cell_shift_map.keys():
    cell_shift_list += cell_shift_map[z]
    s2u_list += s2u_map[z]

p2s_indices  = [index for (index, item) in enumerate(cell_shift_list) if item == (0,0,0)]
cell_index_array = np.array([cell_index_map[cell_shift_tuple] for cell_shift_tuple in cell_shift_list]) # shape: (natoms_sc,)

# Save matrix information
cell_shift_array = np.array(list(cell_index_map.keys()))
cell_shift_of_each_atom = np.array(cell_shift_list)

mat_info =  { 'species':atomic_nums,
            'cell_shift_array':cell_shift_array, # shape: (ncells, 3)
            'cell_index_map':cell_index_map, # len: (ncells,)
            'p2s_indices': np.array(p2s_indices), # shape: (natoms_uc,)
            's2u_list': np.array(s2u_list), # shape: (natoms_sc,)
            'cell_shift_of_each_atom': cell_shift_of_each_atom # shape: (natoms_sc,)
            }
save_dict_by_numpy(os.path.join(mat_info_path, 'mat_info_rc.npy'), mat_info)

# perturb the atoms in the (0,0,0) cell of the supercell
vectors = delta_r*np.identity(3)
for i, idx in enumerate(p2s_indices):
    for j in range(3):
        temp = copy.deepcopy(sc_atoms_dict)
        temp['positions'][idx] += vectors[j]
        atoms_dict_to_openmxfile(temp, openmx_basic_commad, spin_set, PAO_dict, PBE_dict, 
                                 os.path.join(supercell_path,f'{system_name}_{3*i+j+2}.dat'))
if use_central_difference:
    vectors = -delta_r*np.identity(3)
    for i, idx in enumerate(p2s_indices):
        for j in range(3):
            temp = copy.deepcopy(sc_atoms_dict)
            temp['positions'][idx] += vectors[j]
            atoms_dict_to_openmxfile(temp, openmx_basic_commad, spin_set, PAO_dict, PBE_dict, 
                                     os.path.join(supercell_path,f'{system_name}_{3*i+j+2+3*len(p2s_indices)}.dat'))
