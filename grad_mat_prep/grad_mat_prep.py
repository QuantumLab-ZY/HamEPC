import numpy as np
import math
import json
import os
import collections
from utils import (read_openmx_dat, atoms_dict_to_openmxfile, au2ang, save_dict_by_numpy,
                    spin_set, PAO_dict, PBE_dict, basis_def_14, basis_def_19, basis_def_26,
                    build_sparse_matrix)
import collections
from phonopy.interface.vasp import read_vasp
from phonopy.structure.atoms import atom_data, symbol_map

######################## IMPORTANT #######################
# This program can only run on a single core without mpi.
# The openmx_executable_path should be organized as follow:
# openmx_executable_path:
#   - read_openmx
#   - openmx_postprocess
######################## IMPORTANT #######################

######################## Input parameters begin #######################
system_name = 'CVS'
openmx_executable_path = "./"
work_path = "./"
graph_data_dir = "./perturb/graph_data.npz"
mat_info_sc_dir = "./perturb/mat_info_rc.npy"
unitcell_dir = "./CVS.vasp"
H_pred_result_dir = "./perturb/prediction_hamiltonian.npy"
delta_r = 0.01 # ang
use_central_difference = False
nao_max = 19
gather_H_pred = True    # If False, the code will use H_all.npy instead of H_all_pred.npy
read_dSon = False   # If true, read each dSon_xxx.npy from {work_path}/dSon/xxx/
dSon_delta_r = 0.001
######################## Input parameters end #########################

################################ OPENMX calculation parameters Set begin #####################
openmx_basic_command = f"""#
#      File Name      
#

System.CurrrentDirectory         ./    # default=./
System.Name                      {system_name}
DATA.PATH                      ../DFT_DATA19
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

######################################## Path set ########################################
# Note: xxx_dir include filename, xxx_path is a folder.
openmx_executable_path = os.path.abspath(openmx_executable_path)
work_path = os.path.abspath(work_path)
graph_data_dir = os.path.abspath(graph_data_dir)
mat_info_sc_dir = os.path.abspath(mat_info_sc_dir)
unitcell_dir = os.path.abspath(unitcell_dir)
H_pred_result_dir = os.path.abspath(H_pred_result_dir)
openmx_postprocess_dir = os.path.join(openmx_executable_path, 'openmx_postprocess')
read_openmx_dir = os.path.join(openmx_executable_path, 'read_openmx')
H_all_dir = os.path.join(work_path, 'H_all.npy')
H_all_pred_dir = os.path.join(work_path, 'H_all_pred.npy')
S_all_dir = os.path.join(work_path, 'S_all.npy')
nabla_H_dir = os.path.join(work_path, 'nabla_H.npy')
nabla_S_dir = os.path.join(work_path, 'nabla_S.npy')
dSon_path = os.path.join(work_path, 'dSon')
grad_mat_dir = os.path.join(work_path, 'grad_mat.npy')

######################################## Unit change ########################################
delta_r_au = delta_r / au2ang
dSon_delta_r_au = dSon_delta_r / au2ang

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
    raise RuntimeError("Unitcell file format not supperted.")

use_H_gamma = False
natoms = len(coordinates)
# read unitcell info   
uc_atoms_dict = dict()
uc_positions_dict = collections.OrderedDict()
for ia in range(len(atomic_nums)): # len: (natoms_uc,)
    if atomic_nums[ia] not in uc_positions_dict:
        uc_positions_dict[atomic_nums[ia]] = []
    uc_positions_dict[atomic_nums[ia]].append(coordinates[ia])
uc_atoms_dict['chemical_symbols'] = [atom_data[z][1] for z in uc_positions_dict.keys() for j in range(len(uc_positions_dict[z]))]
uc_atoms_dict['positions'] = np.concatenate([uc_positions_dict[z] for z in uc_positions_dict.keys()], axis=0)
uc_atoms_dict['cell'] = np.array(latt)

if nao_max == 14:
    basis_def = basis_def_14
elif nao_max == 19:
    basis_def = basis_def_19
elif nao_max == 26:
    basis_def = basis_def_26
else:
    raise NotImplementedError(f"nao_max = {nao_max} is not supperted.")

######################################## Gather HS ########################################
os.chdir(work_path)
mat_info_sc = np.load(mat_info_sc_dir, allow_pickle=True).item()
cell_shift_array_reduced = mat_info_sc['cell_shift_array'] # shape: (ncells, 3)
cell_index_map_reduced = mat_info_sc['cell_index_map'] # len: (ncells,)
p2s_indices_reduced = mat_info_sc['p2s_indices'] # shape: (natoms_uc,)
s2u_list_reduced = mat_info_sc['s2u_list'] # shape: (natoms_sc,)
cell_shift_of_each_atom_in_sc = mat_info_sc['cell_shift_of_each_atom'] # len: (natoms_sc,3)
species_uc = mat_info_sc['species']

ncells_reduced = len(cell_shift_array_reduced)
natoms_uc = len(p2s_indices_reduced)

# parse the Atomic Orbital Basis Sets
basis_definition = np.zeros((99, nao_max))
# key is the atomic number, value is the index of the occupied orbits.
for k in basis_def.keys():
    basis_definition[k][basis_def[k]] = 1

orb_mask = basis_definition[species_uc].reshape(-1) # shape: [natoms_uc*nao_max] 
orb_mask = orb_mask[:,None] * orb_mask[None,:]   # shape: [natoms_uc*nao_max, natoms_uc*nao_max]

graph_data = np.load(graph_data_dir, allow_pickle=True)
graph_data = graph_data['graph'].item()
graph_dataset = list(graph_data.values())

if gather_H_pred:
    # Calculate the length of H for each structure
    len_H = []
    for i in range(len(graph_dataset)):
        len_H.append(len(graph_dataset[i].Hon))
        len_H.append(len(graph_dataset[i].Hoff))
    
    H = np.load(H_pred_result_dir).reshape(-1, nao_max, nao_max)
    
    Hon_all, Hoff_all = [], []
    idx = 0
    for i in range(0, len(len_H), 2):
        Hon_all.append(H[idx:idx + len_H[i]])
        idx = idx+len_H[i]
        Hoff_all.append(H[idx:idx + len_H[i+1]])
        idx = idx+len_H[i+1]

H_all = []
S_all = []
H_all_pred = []
for idx, data in enumerate(graph_dataset):
    # build crystal structure
    Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
    Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
    Hon = data.Hon.numpy().reshape(-1, nao_max, nao_max)
    Hoff = data.Hoff.numpy().reshape(-1, nao_max, nao_max)
    latt = data.cell.numpy().reshape(3,3)
    cell_shift = data.cell_shift.numpy()
    nbr_shift = data.nbr_shift.numpy()
    edge_index = data.edge_index.numpy()
    species = data.z.numpy()
    natoms_sc = len(species)
    
    # shape: (Ncells, natoms_sc, nao_max, natoms_sc, nao_max)
    H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index = build_sparse_matrix(species, cell_shift, nao_max, 
                                                                            Hon, Hoff, edge_index,return_raw_mat=True)
    S_cell, _, _, _, _ = build_sparse_matrix(species, cell_shift, nao_max, Son, Soff, edge_index, return_raw_mat=True)

    if use_H_gamma:
        H_000 = np.einsum('nijkl->ijkl', H_cell) # shape: (natoms_sc, nao_max, natoms_sc, nao_max)
        S_000 = np.einsum('nijkl->ijkl', S_cell) # shape: (natoms_sc, nao_max, natoms_sc, nao_max)
    else:
        H_000 = H_cell[cell_index_map[(0,0,0)]] # shape: (natoms_sc, nao_max, natoms_sc, nao_max)
        S_000 = S_cell[cell_index_map[(0,0,0)]] # shape: (natoms_sc, nao_max, natoms_sc, nao_max)
    
    del H_cell, S_cell
    
    H_reduced = np.zeros((ncells_reduced, ncells_reduced, natoms_uc, nao_max, natoms_uc, nao_max))
    S_reduced = np.zeros((ncells_reduced, ncells_reduced, natoms_uc, nao_max, natoms_uc, nao_max))
    
    for i, Ci in enumerate(cell_shift_of_each_atom_in_sc): # ncells
        ia = s2u_list_reduced[i]
        Ci = cell_index_map_reduced[tuple(Ci.tolist())]
        for j, Cj in enumerate(cell_shift_of_each_atom_in_sc): # ncells
            ja = s2u_list_reduced[j]
            Cj = cell_index_map_reduced[tuple(Cj.tolist())]
            H_reduced[Ci,Cj,ia,:,ja,:] = H_000[i,:,j,:]
            S_reduced[Ci,Cj,ia,:,ja,:] = S_000[i,:,j,:]
    
    H_reduced = H_reduced.reshape(ncells_reduced, ncells_reduced, natoms_uc*nao_max, natoms_uc*nao_max)
    H_reduced = H_reduced[:, :, orb_mask > 0] # shape: (ncells, ncells, norbs*norbs)
    norbs = int(math.sqrt(H_reduced.size/(ncells_reduced*ncells_reduced)))
    H_reduced = H_reduced.reshape(ncells_reduced, ncells_reduced, norbs, norbs)
    
    S_reduced = S_reduced.reshape(ncells_reduced, ncells_reduced, natoms_uc*nao_max, natoms_uc*nao_max)
    S_reduced = S_reduced[:, :, orb_mask > 0] # shape: (ncells, ncells, norbs, norbs)
    S_reduced = S_reduced.reshape(ncells_reduced, ncells_reduced, norbs, norbs)
    
    H_all.append(H_reduced)
    S_all.append(S_reduced)
    
    if gather_H_pred:
        Hon_pred = Hon_all[idx].reshape(-1, nao_max, nao_max)
        Hoff_pred = Hoff_all[idx].reshape(-1, nao_max, nao_max)
        
        H_cell_pred, _, _, _, _ = build_sparse_matrix(species, cell_shift, nao_max, Hon_pred, Hoff_pred, edge_index, return_raw_mat=True)
        
        if use_H_gamma:
            HP_000 = np.einsum('nijkl->ijkl', H_cell_pred) # shape: (natoms_sc, nao_max, natoms_sc, nao_max)
        else:
            HP_000 = H_cell_pred[cell_index_map[(0,0,0)]]
        
        del H_cell_pred
        
        HP_reduced = np.zeros((ncells_reduced, ncells_reduced, natoms_uc, nao_max, natoms_uc, nao_max))
        for i, Ci in enumerate(cell_shift_of_each_atom_in_sc): # ncells
            ia = s2u_list_reduced[i]
            Ci = cell_index_map_reduced[tuple(Ci.tolist())]
            for j, Cj in enumerate(cell_shift_of_each_atom_in_sc): # ncells
                ja = s2u_list_reduced[j]
                Cj = cell_index_map_reduced[tuple(Cj.tolist())]
                HP_reduced[Ci,Cj,ia,:,ja,:] = HP_000[i,:,j,:]
        
        HP_reduced = HP_reduced.reshape(ncells_reduced, ncells_reduced, natoms_uc*nao_max, natoms_uc*nao_max)
        HP_reduced = HP_reduced[:, :, orb_mask > 0] # shape: (ncells, ncells, norbs, norbs)
        HP_reduced = HP_reduced.reshape(ncells_reduced, ncells_reduced, norbs, norbs)
        H_all_pred.append(HP_reduced)
    
del graph_data, graph_dataset, mat_info_sc

H_all = np.stack(H_all, 0) # shape: (natoms_uc*3+1, ncells, ncells, norbs, norbs)
np.save(file=H_all_dir, arr=H_all)
del H_all

S_all = np.stack(S_all, 0) # shape: (natoms_uc*3+1, ncells, ncells, norbs, norbs)
np.save(file=S_all_dir, arr=S_all)
del S_all
if gather_H_pred:
    H_all_pred = np.stack(H_all_pred, 0) # shape: (natoms_uc*3+1, ncells, ncells, norbs, norbs)
    np.save(file=H_all_pred_dir, arr=H_all_pred)
print("Done gather HS.")
######################################## dS and dH ########################################
os.chdir(work_path)
if gather_H_pred:
    H_all = np.load(H_all_pred_dir)  # shape: (natoms*3+1, ncell, ncell, norbs, norbs)
else:
    H_all = np.load(H_all_dir)  # shape: (natoms*3+1, ncell, ncell, norbs, norbs)
S_all = np.load(S_all_dir)  # shape: (natoms*3+1, ncell, ncell, norbs, norbs)
if use_central_difference:
    nabla_H = []
    nabla_S = []
    for i in range(1, natoms*3+1):
        dH = H_all[i] - H_all[i+natoms*3]
        dH = dH/(2*delta_r_au)
        dS = S_all[i] - S_all[i+natoms*3]
        dS = dS/(2*delta_r_au)
        nabla_H.append(dH)
        nabla_S.append(dS)
else:
    nabla_H = []
    nabla_S = []
    for i in range(1, natoms*3+1):
        dH = H_all[i] - H_all[0]
        dH = dH/delta_r_au
        dS = S_all[i] - S_all[0]
        dS = dS/delta_r_au
        nabla_H.append(dH)
        nabla_S.append(dS)
nabla_H = np.stack(nabla_H, -1).reshape(*dH.shape, natoms, 3)
nabla_S = np.stack(nabla_S, -1).reshape(*dS.shape, natoms, 3)
np.save(file=nabla_H_dir, arr=nabla_H) # shape: (ncells_reduced, ncells_reduced, norbs, norbs, natoms, 3)
np.save(file=nabla_S_dir, arr=nabla_S) # shape: (ncells_reduced, ncells_reduced, norbs, norbs, natoms, 3)
del H_all, S_all, nabla_H, nabla_S
print("Done dS dH calculation.")
######################################## dSon calulation ########################################
os.chdir(work_path)
element_unique = []
if read_dSon:
    for each in uc_atoms_dict['chemical_symbols']:
        if each not in element_unique:
            element_unique.append(each)
else:
    if os.path.exists(dSon_path):
        print(f"Warnning: {dSon_path} already exist...")
        print(f"The programme will continue running, and cover the files in it...")
    else:
        os.mkdir(dSon_path)
    for iatom, each in enumerate(uc_atoms_dict['chemical_symbols']):
        if each not in element_unique:
            element_unique.append(each)
            dSon_atoms_dict = dict()
            dSon_atoms_dict['chemical_symbols'] = [each, each, each, each]
            dSon_atom_positions = [np.dot(np.array([0.5, 0.5, 0.5]), uc_atoms_dict['cell'])]
            for i in range(3):
                tmp_atom_position = dSon_atom_positions[0].copy()
                tmp_atom_position[i] += dSon_delta_r
                dSon_atom_positions.append(tmp_atom_position)
            dSon_atoms_dict['positions'] = np.array(dSon_atom_positions)
            dSon_atoms_dict['cell'] = uc_atoms_dict['cell']
            this_run_path = os.path.join(dSon_path, each)
            if os.path.exists(this_run_path):
                print(f"Warnning: {this_run_path} already exist...")
                print(f"The programme will continue running, and cover the files in it...")
            else:
                os.mkdir(this_run_path)
            atoms_dict_to_openmxfile(dSon_atoms_dict, openmx_basic_command,
                                    spin_set, PAO_dict, PBE_dict,
                                    os.path.join(this_run_path, 'openmx.dat'))
            os.chdir(this_run_path)
            os.system(f"{openmx_postprocess_dir} openmx.dat > log.out 2>&1")
            os.system(f"{read_openmx_dir} overlap.scfout")
            if not os.path.exists(os.path.join(this_run_path, 'HS.json')):
                raise RuntimeError(f"read_openmx fail in {this_run_path}, while dealing with overlap.scfout.")
            f_HS_dSon_tmp = open(os.path.join(this_run_path, 'HS.json'))
            data_HS_dSon_tmp = json.load(f_HS_dSon_tmp)
            edge_index_HS_dSon_tmp = np.array(data_HS_dSon_tmp['edge_index'])
            cell_shift_HS_dSon_tmp = [tuple(cell_shift) for cell_shift in data_HS_dSon_tmp['cell_shift']]
            Soff_HS_dSon_tmp = np.array(data_HS_dSon_tmp['Soff'])
            Son_HS_dSon_tmp = np.array(data_HS_dSon_tmp['Son'])
            nao_max_element = int(math.sqrt(Son_HS_dSon_tmp.size / 4))
            Soff_HS_dSon_tmp = Soff_HS_dSon_tmp.reshape(-1, nao_max_element, nao_max_element)
            Son_HS_dSon_tmp = Son_HS_dSon_tmp.reshape(-1, nao_max_element, nao_max_element)
            n_src_0 = data_HS_dSon_tmp['edge_index'][0].count(0)
            Son_cell = dict()
            iedge = 0
            for tar, icell in zip(edge_index_HS_dSon_tmp[1][:n_src_0], cell_shift_HS_dSon_tmp[:n_src_0]):
                if icell not in Son_cell:        
                    Son_cell[icell] = dict()  
                Son_cell[icell][tar] = Soff_HS_dSon_tmp[iedge]
                iedge = iedge + 1
            dSon_cell = dict() # dSon_cell[(x,x,x)][1,2,3]: shape: (nao_max, nao_max)
            for tar, icell in zip(edge_index_HS_dSon_tmp[1][:n_src_0], cell_shift_HS_dSon_tmp[:n_src_0]):
                if tar == 0:
                    continue
                if icell not in dSon_cell:        
                    dSon_cell[icell] = dict()
                if icell == (0,0,0):
                    dSon_cell[icell][tar] = (Son_cell[icell][tar]-Son_HS_dSon_tmp[0])/dSon_delta_r_au
                else:
                    dSon_cell[icell][tar] = (Son_cell[icell][tar]-Son_cell[icell][0])/dSon_delta_r_au
            save_dict_by_numpy(os.path.join(this_run_path, f"dSon_{each}.npy"), dSon_cell)
            print(f"Done dSon calculation for {each}")

######################################## Gradient matrix calculation ########################################
os.chdir(work_path)
if gather_H_pred:
    H_all = np.load(H_all_pred_dir)
else:
    H_all = np.load(H_all_dir)
S_all = np.load(S_all_dir)
nabla_H = np.load(nabla_H_dir)
nabla_S = np.load(nabla_S_dir)
mat_info_sc = np.load(mat_info_sc_dir, allow_pickle=True).item()
dSon = dict()
for each in element_unique:
    dSon[symbol_map[each]] = np.load(os.path.join(dSon_path, each, f"dSon_{each}.npy"), allow_pickle=True).item()
atomic_nums = [symbol_map[s] for s in uc_atoms_dict['chemical_symbols']]
cell_shift_array_reduced = mat_info_sc['cell_shift_array'] # shape: (ncells, 3)
cell_index_map_reduced = mat_info_sc['cell_index_map'] # len: (ncells,)
p2s_indices_reduced = mat_info_sc['p2s_indices'] # shape: (natoms_uc,)
s2u_list_reduced = mat_info_sc['s2u_list'] # shape: (natoms_sc,)
cell_shift_of_each_atom_in_sc = mat_info_sc['cell_shift_of_each_atom'] # shape: (natoms_sc, 3)
natoms_uc = len(p2s_indices_reduced)
ncells = len(cell_shift_array_reduced)
H, S = H_all[0], S_all[0] # shape: (ncells_reduced, ncells_reduced, norbs, norbs)
del H_all, S_all
# build dense matrix
norbs = H.shape[2]
H_den = np.swapaxes(H, axis1=1, axis2=2).reshape(ncells*norbs, ncells*norbs)
S_den = np.swapaxes(S, axis1=1, axis2=2).reshape(ncells*norbs, ncells*norbs)
S_den_inv = np.linalg.inv(S_den)
# Initialize dSbar
repeats = []
for ia in range(natoms_uc):
    repeats.append(len(basis_def[atomic_nums[ia]]))
orb2atom_idx = np.repeat(np.arange(natoms_uc), repeats=repeats, axis=0)
dSbar = np.zeros_like(nabla_S) # shape: (ncells_reduced, ncells_reduced, norbs, norbs, natoms, 3)
center_cell_idx = cell_index_map_reduced[(0,0,0)]
m = np.arange(norbs)
dSbar[center_cell_idx,:,m,:,orb2atom_idx[m],:] = -nabla_S[center_cell_idx,:,m,:,orb2atom_idx[m],:]
start = 0
for ia in range(natoms_uc):
    dSbar[center_cell_idx,center_cell_idx,start:start+repeats[ia],start:start+repeats[ia],ia,0] = dSon[atomic_nums[ia]][(0,0,0)][1] # x
    dSbar[center_cell_idx,center_cell_idx,start:start+repeats[ia],start:start+repeats[ia],ia,1] = dSon[atomic_nums[ia]][(0,0,0)][2] # y
    dSbar[center_cell_idx,center_cell_idx,start:start+repeats[ia],start:start+repeats[ia],ia,2] = dSon[atomic_nums[ia]][(0,0,0)][3] # z
    start += repeats[ia]
dSbar = np.swapaxes(dSbar, axis1=1, axis2=2).reshape(ncells*norbs, ncells*norbs, natoms_uc, 3)
tmp1 = np.dot(S_den_inv, H_den)
sum1 = []
for ia in range(natoms_uc):
    for dir in range(3):
        sum1.append(np.dot(dSbar[:,:,ia, dir], tmp1))
sum1 = np.stack(sum1, axis=-1).reshape(ncells*norbs, ncells*norbs, natoms_uc, 3)
sum2 = np.swapaxes(sum1, axis1=0, axis2=1)
grad_mat = (sum1+sum2).reshape(ncells, norbs, ncells, norbs, natoms_uc, 3)
grad_mat = np.swapaxes(grad_mat, axis1=1, axis2=2) # shape: (ncells, ncells, norbs, norbs, natoms_uc, 3)
grad_mat += nabla_H

# output grad_mat
np.save(file=grad_mat_dir, arr=grad_mat)
print("Done gradient matrix calculation.")
