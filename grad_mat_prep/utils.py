'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2023-01-14 11:07:06
LastEditors: Yang Zhong
LastEditTime: 2023-11-18 16:17:29
'''
from ctypes import Union
import numpy as np
from typing import Tuple, Union, List
from pymatgen.core.periodic_table import Element
from ase import Atoms
import math
import os
import re
import threading


def _nice_float(x,just,rnd):
    return str(round(x,rnd)).rjust(just)

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

        return (k_vec,k_dist,k_node,lat_per_inv, node_index)

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
            'Ge':[2.0,2.0],
            'Y':[5.5,5.5]
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
            'Ge': 'Ge7.0-s3p2d2',
            'Y':'Y10.0-s3p2d2'
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
            'Ge':'Ge_PBE19',
            'Y':'Y_PBE19'}

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

basis_def_26 = (lambda s1=[0],s2=[1],s3=[2],p1=[3,4,5],p2=[6,7,8],d1=[9,10,11,12,13],d2=[14,15,16,17,18],f1=[19,20,21,22,23,24,25]: {
    Element['H'].Z : np.array(s1+s2+p1, dtype=int), # H6.0-s2p1
    Element['He'].Z : np.array(s1+s2+p1, dtype=int), # He8.0-s2p1
    Element['Li'].Z : np.array(s1+s2+s3+p1+p2, dtype=int), # Li8.0-s3p2
    Element['Be'].Z : np.array(s1+s2+p1+p2, dtype=int), # Be7.0-s2p2
    Element['B'].Z : np.array(s1+s2+p1+p2+d1, dtype=int), # B7.0-s2p2d1
    Element['C'].Z : np.array(s1+s2+p1+p2+d1, dtype=int), # C6.0-s2p2d1
    Element['N'].Z : np.array(s1+s2+p1+p2+d1, dtype=int), # N6.0-s2p2d1
    Element['O'].Z : np.array(s1+s2+p1+p2+d1, dtype=int), # O6.0-s2p2d1
    Element['F'].Z : np.array(s1+s2+p1+p2+d1, dtype=int), # F6.0-s2p2d1
    Element['Ne'].Z: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne9.0-s2p2d1
    Element['Na'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Na9.0-s3p2d1
    Element['Mg'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Mg9.0-s3p2d1
    Element['Al'].Z: np.array(s1+s2+p1+p2+d1, dtype=int), # Al7.0-s2p2d1
    Element['Si'].Z: np.array(s1+s2+p1+p2+d1, dtype=int), # Si7.0-s2p2d1
    Element['P'].Z: np.array(s1+s2+p1+p2+d1, dtype=int), # P7.0-s2p2d1
    Element['S'].Z: np.array(s1+s2+p1+p2+d1, dtype=int), # S7.0-s2p2d1
    Element['Cl'].Z: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl7.0-s2p2d1
    Element['Ar'].Z: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar9.0-s2p2d1
    Element['K'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # K10.0-s3p2d1
    Element['Ca'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Ca9.0-s3p2d1
    Element['Sc'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Sc9.0-s3p2d1
    Element['Ti'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Ti7.0-s3p2d1
    Element['V'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # V6.0-s3p2d1
    Element['Cr'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Cr6.0-s3p2d1
    Element['Mn'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Mn6.0-s3p2d1
    Element['Fe'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Fe5.5H-s3p2d1
    Element['Co'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Co6.0H-s3p2d1
    Element['Ni'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Ni6.0H-s3p2d1
    Element['Cu'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Cu6.0H-s3p2d1
    Element['Zn'].Z: np.array(s1+s2+s3+p1+p2+d1, dtype=int), # Zn6.0H-s3p2d1
    Element['Ga'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Ga7.0-s3p2d2
    Element['Ge'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Ge7.0-s3p2d2
    Element['As'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # As7.0-s3p2d2
    Element['Se'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Se7.0-s3p2d2
    Element['Br'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Br7.0-s3p2d2
    Element['Kr'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Kr10.0-s3p2d2
    Element['Rb'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Rb11.0-s3p2d2
    Element['Sr'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Sr10.0-s3p2d2
    Element['Y'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Y10.0-s3p2d2
    Element['Zr'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Zr7.0-s3p2d2
    Element['Nb'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Nb7.0-s3p2d2
    Element['Mo'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Mo7.0-s3p2d2
    Element['Tc'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Tc7.0-s3p2d2
    Element['Ru'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Ru7.0-s3p2d2
    Element['Rh'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Rh7.0-s3p2d2
    Element['Pd'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Pd7.0-s3p2d2
    Element['Ag'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Ag7.0-s3p2d2
    Element['Cd'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Cd7.0-s3p2d2
    Element['In'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # In7.0-s3p2d2
    Element['Sn'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Sn7.0-s3p2d2
    Element['Sb'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Sb7.0-s3p2d2
    Element['Te'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Te7.0-s3p2d2f1
    Element['I'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # I7.0-s3p2d2f1
    Element['Xe'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Xe11.0-s3p2d2
    Element['Cs'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Cs12.0-s3p2d2
    Element['Ba'].Z: np.array(s1+s2+s3+p1+p2+d1+d2, dtype=int), # Ba10.0-s3p2d2
    Element['La'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # La8.0-s3p2d2f1
    Element['Ce'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Ce8.0-s3p2d2f1
    Element['Pr'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Pr8.0-s3p2d2f1
    Element['Nd'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Nd8.0-s3p2d2f1
    Element['Pm'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Pm8.0-s3p2d2f1
    Element['Sm'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Sm8.0-s3p2d2f1
    Element['Dy'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Dy8.0-s3p2d2f1
    Element['Ho'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Ho8.0-s3p2d2f1
    Element['Lu'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Lu8.0-s3p2d2f1
    Element['Hf'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Hf9.0-s3p2d2f1
    Element['Ta'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Ta7.0-s3p2d2f1
    Element['W'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # W7.0-s3p2d2f1
    Element['Re'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Re7.0-s3p2d2f1
    Element['Os'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Os7.0-s3p2d2f1
    Element['Ir'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Ir7.0-s3p2d2f1
    Element['Pt'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Pt7.0-s3p2d2f1
    Element['Au'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Au7.0-s3p2d2f1
    Element['Hg'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Hg8.0-s3p2d2f1
    Element['Tl'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Tl8.0-s3p2d2f1
    Element['Pb'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Pb8.0-s3p2d2f1
    Element['Bi'].Z: np.array(s1+s2+s3+p1+p2+d1+d2+f1, dtype=int), # Bi8.0-s3p2d2f1 
})()

num_valence = {1:1,2:2,3:3,4:2,5:3,6:4,7:5,8:6,9:7,10:8,11:9,12:8,13:3,14:4,15:5,16:6,17:7,18:8,19:9,20:10,42:14,83:15,34:6,
               35:7,55:9,82:14,53:7,33:15,80:18,31:13,Element['V'].Z:13,Element['Sb'].Z:15, Element['Ge'].Z:4}
num_val = np.zeros((99,), dtype=int)
for k in num_valence.keys():
    num_val[k] = num_valence[k]

pattern_eng = re.compile(r'Enpy  =(\W+)(\-\d+\.?\d*)')
pattern_md = re.compile(r'MD= 1  SCF=(\W*)(\d+)')
pattern_latt = re.compile(r'<Atoms.UnitVectors.+?\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+Atoms.UnitVectors>')
pattern_coor = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+')
num = r'-?\d+\.?\d*'
wht = r'\s+'
pattern_eng_siesta = re.compile(r'siesta: Etot\s+=\s+(\-\d+\.?\d*)')
pattern_md_siesta = re.compile(r'scf:\s+(\d+)')
pattern_latt_siesta = re.compile(r'%block LatticeVectors.*' + f'{wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num})' + r'\s+%endblock LatticeVectors')
pattern_cblk_siesta = re.compile(r'%block AtomicCoordinatesAndAtomicSpecies(.+)%endblock AtomicCoordinatesAndAtomicSpecies', flags=re.S)
pattern_coor_siesta = re.compile(f'{wht}({num}){wht}({num}){wht}({num}){wht}(\d+)')
pattern_sblk_siesta = re.compile(r'%block ChemicalSpeciesLabel(.+)%endblock ChemicalSpeciesLabel', flags=re.S)
pattern_spec_siesta = re.compile(r'\s+(\d+)\s+(\d+)\s+(\w+)')
    
au2ang = 0.5291772490000065
au2ev = 27.211324570273
max_SCF_skip = 100

def atoms_dict_to_openmxfile(atoms_dict:dict, basic_commad:str, spin_set:dict, PAO_dict:dict, PBE_dict:dict, filename:str):
    chemical_symbols = atoms_dict['chemical_symbols']
    species = set(chemical_symbols)
    positions = atoms_dict['positions']
    cell = atoms_dict['cell']
    openmx = basic_commad
    openmx += "#\n# Definition of Atomic Species\n#\n"
    openmx += f'Species.Number       {len(species)}\n'
    openmx += '<Definition.of.Atomic.Species\n'
    for s in species:
        openmx += f"{s}   {PAO_dict[s]}       {PBE_dict[s]}\n"    
    openmx += "Definition.of.Atomic.Species>\n\n"
    openmx += "#\n# Atoms\n#\n"
    openmx += "Atoms.Number%12d" % len(chemical_symbols)
    openmx += "\nAtoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU"
    openmx += "\n<Atoms.SpeciesAndCoordinates           # Unit=Ang."
    for num, sym in enumerate(chemical_symbols):
        openmx += "\n%3d  %s  %10.7f  %10.7f  %10.7f   %.2f   %.2f" % (num+1, sym, *positions[num], *spin_set[chemical_symbols[num]])
    openmx += "\nAtoms.SpeciesAndCoordinates>"
    openmx += "\nAtoms.UnitVectors.Unit             Ang #  Ang|AU"
    openmx += "\n<Atoms.UnitVectors                     # unit=Ang."
    openmx += "\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f" % (*cell[0], *cell[1], *cell[2])
    openmx += "\nAtoms.UnitVectors>"
    with open(filename,'w') as wf:
        wf.write(openmx)

def ase_atoms_to_openmxfile(atoms:Atoms, basic_commad:str, spin_set:dict, PAO_dict:dict, PBE_dict:dict, filename:str):
    chemical_symbols = atoms.get_chemical_symbols()
    species = set(chemical_symbols)
    positions = atoms.get_array(name='positions')
    cell = atoms.get_cell().array
    openmx = basic_commad
    openmx += "#\n# Definition of Atomic Species\n#\n"
    openmx += f'Species.Number       {len(species)}\n'
    openmx += '<Definition.of.Atomic.Species\n'
    for s in species:
        openmx += f"{s}   {PAO_dict[s]}       {PBE_dict[s]}\n"    
    openmx += "Definition.of.Atomic.Species>\n\n"
    openmx += "#\n# Atoms\n#\n"
    openmx += "Atoms.Number%12d" % len(chemical_symbols)
    openmx += "\nAtoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU"
    openmx += "\n<Atoms.SpeciesAndCoordinates           # Unit=Ang."
    for num, sym in enumerate(chemical_symbols):
        openmx += "\n%3d  %s  %10.7f  %10.7f  %10.7f   %.2f   %.2f" % (num+1, sym, *positions[num], *spin_set[chemical_symbols[num]])
    openmx += "\nAtoms.SpeciesAndCoordinates>"
    openmx += "\nAtoms.UnitVectors.Unit             Ang #  Ang|AU"
    openmx += "\n<Atoms.UnitVectors                     # unit=Ang."
    openmx += "\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f" % (*cell[0], *cell[1], *cell[2])
    openmx += "\nAtoms.UnitVectors>"
    with open(filename,'w') as wf:
        wf.write(openmx)

def read_openmx_dat(filename: str = None):
    
    pattern_latt = re.compile(r'<Atoms.UnitVectors.+\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+Atoms.UnitVectors>')
    pattern_coor = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+')

    with open(filename,'r') as f:
        content = f.read()
        speciesAndCoordinates = pattern_coor.findall((content).strip())
        latt = pattern_latt.findall((content).strip())[0]
        latt = np.array([float(var) for var in latt]).reshape(-1, 3)/au2ang    
        species = []
        coordinates = []
        for item in speciesAndCoordinates:
            species.append(item[0])
            coordinates += item[1:]
        atomic_numbers = np.array([Element[s].Z for s in species])
        coordinates = np.array([float(pos) for pos in coordinates]).reshape(-1, 3)/au2ang
    
    return atomic_numbers, latt, coordinates

def build_sparse_matrix(species, cell_shift, nao_max, Hon, Hoff, edge_index, return_raw_mat:bool=False):
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
        if nao_max == 14:
            basis_def = basis_def_14
        elif nao_max == 19:
            basis_def = basis_def_19
        elif nao_max == 26:
            basis_def = basis_def_26
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

def sparse_matrix_mm(A, B, cell_shift_array, cell_index_map):
    """
    Args:
        A (np.array): (ncells, norbs, norbs)
        B (np.array): (ncells, norbs, norbs)
        cell_shift_array (np.array): _description_
        cell_index_map (dict): _description_
    """
    Ret = np.zeros_like(A)
    for n in range(len(Ret)):
        for nn in range(len(A)):
            relative_shift = tuple((cell_shift_array[n] - cell_shift_array[nn]).tolist())
            if relative_shift in cell_index_map:
                Ret[n] += np.dot(A[nn], B[cell_index_map[relative_shift]])
    return Ret

def sparse_matrix_mm2(A, B, cell_shift_array, cell_index_map):
    """
    Args:
        A (np.array): (ncells, norbs, norbs, natoms, 3)
        B (np.array): (ncells, norbs, norbs)
        cell_shift_array (np.array): _description_
        cell_index_map (dict): _description_
    """
    Ret = np.zeros_like(A)
    for n in range(len(Ret)):
        for nn in range(len(A)):
            relative_shift = tuple((cell_shift_array[n] - cell_shift_array[nn]).tolist())
            if relative_shift in cell_index_map:
                Ret[n] += np.einsum('ijkl, jo -> iokl', A[nn], B[cell_index_map[relative_shift]])
    return Ret

def build_reciprocal_mat(Hon, Hoff, species, nao_max, latt, nk, k_path, edge_index, nbr_shift):
    """_summary_

    Args:
        Hon (ndarray): (natoms, nao_max, nao_max) 
        Hoff (ndarray): (nedges, nao_max, nao_max)
        species (ndarray): (natoms,)
        latt (ndarray): (3, 3)
        nk: (int)
        k_path (list): [(x,x,x), (x,x,x)]
        nbr_shift (ndarray): (nedges, 3)
    """

    # parse the Atomic Orbital Basis Sets
    basis_definition = np.zeros((99, nao_max))
    # key is the atomic number, value is the index of the occupied orbits.
    if nao_max == 14:
        basis_def = basis_def_14
    elif nao_max == 19:
        basis_def = basis_def_19
    elif nao_max == 26:
            basis_def = basis_def_26
    else:
        raise NotImplementedError
    for k in basis_def.keys():
        basis_definition[k][basis_def[k]] = 1
      
    orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
    orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [natoms*nao_max, natoms*nao_max]

    kpts=kpoints_generator(dim_k=3, lat=latt)
    k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)

    k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
    k_vec = k_vec.reshape(-1,3) # shape (nk, 3)

    natoms = len(species)
    HK = np.zeros((nk, natoms, natoms, nao_max, nao_max), dtype=np.complex64)
    
    na = np.arange(natoms)
    HK[:,na,na,:,:] +=  Hon[None,na,:,:] # shape (nk, natoms, nao_max, nao_max)

    coe = np.exp(2j*np.pi*np.sum(nbr_shift[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, nedges)

    for iedge in range(len(Hoff)):
        # shape (num_k, nao_max, nao_max) += (num_k, 1, 1)*(1, nao_max, nao_max)
        HK[:,edge_index[0, iedge],edge_index[1, iedge]] += coe[:,iedge,None,None] * Hoff[None,iedge,:,:]

    HK = np.swapaxes(HK,2,3) #(nk, natoms, nao_max, natoms, nao_max)
    HK = HK.reshape(nk, natoms*nao_max, natoms*nao_max)

    # mask HK and SK
    #HK = torch.masked_select(HK, orb_mask[idx].repeat(nk,1,1) > 0)
    HK = HK[:, orb_mask > 0]
    norbs = int(math.sqrt(HK.size/nk))
    HK = HK.reshape(nk, norbs, norbs)
    
    return HK

def build_reciprocal_from_sparseMat(H_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (ncells, norbs, norbs)
    """
    
    phase = np.exp(2j*np.pi*np.sum(nbr_shift_of_cell[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, ncells)
    HK = np.einsum('ijk, ni->njk', H_cell, phase) # (nk, norbs, norbs,)
    
    return HK

def build_reciprocal_from_sparseMat2(nabla_H_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (ncells, norbs, norbs, natoms, 3)
    """
    
    phase = np.exp(2j*np.pi*np.sum(nbr_shift_of_cell[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, ncells)
    HK = np.einsum('ijkmo, ni->njkmo', nabla_H_cell, phase) # (nk, norbs, norbs, natoms, 3)
    
    return HK

def build_reciprocal_from_sparseMat3(M_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (ncells, norbs, norbs, 3)
    """
    
    phase = np.exp(2j*np.pi*np.sum(nbr_shift_of_cell[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, ncells)
    MK = 1.0j*np.einsum('ijkl, ni->njkl', M_cell, phase) # (nk, norbs, norbs, 3)
    
    return MK

def transpose_sparse(H, inv_cell_index):
    H_trans = np.zeros_like(H) # shape: (ncells, norbs, norbs)/(ncells, norbs, norbs， natoms, 3)  
    for i in range(len(H)):
        H_trans[inv_cell_index[i]] = np.swapaxes(np.conj(H[i]), axis1=0, axis2=1)
    return H_trans

def build_dense_matrix(H, cell_shift_array, cell_index_map):
    """_summary_

    Args:
        H: shape: (ncells, norbs, norbs)
    """
    ncells = H.shape[0]
    norbs = H.shape[1]
    H_dense = np.zeros((ncells, norbs, ncells, norbs), dtype=H.dtype)
        
    for m, Cm in enumerate(cell_shift_array): # ncells
        for n, Cn in enumerate(cell_shift_array): # ncells
            relative_cell_shift = tuple((Cn - Cm).tolist())
            if relative_cell_shift in cell_index_map:
                relative_cell_mn = cell_index_map[relative_cell_shift] 
                H_dense[m,:,n,:] = H[relative_cell_mn]
    
    return H_dense

def build_dense_matrix2(H, cell_shift_array, cell_index_map):
    """_summary_

    Args:
        H: shape: (ncells, norbs, norbs, natoms, 3)
    """
    ncells = H.shape[0]
    norbs = H.shape[1]
    natoms = H.shape[3]
    H_dense = np.zeros((ncells, norbs, ncells, norbs, natoms, 3), dtype=H.dtype)
        
    for m, Cm in enumerate(cell_shift_array): # ncells
        for n, Cn in enumerate(cell_shift_array): # ncells
            relative_cell_shift = tuple((Cn - Cm).tolist())
            if relative_cell_shift in cell_index_map:
                relative_cell_mn = cell_index_map[relative_cell_shift] 
                H_dense[m,:,n,:,:,:] = H[relative_cell_mn]
    
    return H_dense

def build_sparse_matrix_from_dense_matrix(H, cell_shift_array, cell_index_map):
    """_summary_

    Args:
        H: shape: (ncells, norbs, ncells, norbs)
    """
    ncells = H.shape[0]
    norbs = H.shape[1]
    H_sparse = np.zeros((ncells, norbs, norbs), dtype=H.dtype)
        
    for m, Cm in enumerate(cell_shift_array):
        for n, Cn in enumerate(cell_shift_array):
            relative_cell_shift = tuple((Cn - Cm).tolist())
            if relative_cell_shift in cell_index_map:
                relative_cell_mn = cell_index_map[relative_cell_shift] 
                H_sparse[relative_cell_mn] = H[m,:,n,:]
    
    return H_sparse

def build_sparse_matrix_from_dense_matrix2(H, cell_shift_array, cell_index_map):
    """_summary_

    Args:
        H: shape: (ncells, norbs, ncells, norbs, natoms, 3)
    """
    ncells = H.shape[0]
    norbs = H.shape[1]
    natoms = H.shape[4]
    H_sparse = np.zeros((ncells, norbs, norbs, natoms, 3), dtype=H.dtype)
        
    for m, Cm in enumerate(cell_shift_array):
        for n, Cn in enumerate(cell_shift_array):
            relative_cell_shift = tuple((Cn - Cm).tolist())
            if relative_cell_shift in cell_index_map:
                relative_cell_mn = cell_index_map[relative_cell_shift] 
                H_sparse[relative_cell_mn] = H[m,:,n,:,:,:]
    
    return H_sparse

def save_dict_by_numpy(filename, dict_vale):
    if not (os.path.exists(os.path.dirname(filename))):
        os.mkdir(os.path.dirname(filename))
    np.save(filename, dict_vale)

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = arr.shape.
    This should be a view of the original array.
    """
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)

def do_dot(a, b, out):
    #np.dot(a, b, out)  # does not work. maybe because out is not C-contiguous?
    out[:] = np.dot(a, b)  # less efficient because the output is stored in a temporary array?

def pardot(a, b, nblocks, mblocks, dot_func=do_dot):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))

    out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

    threads = []
    for i in range(nblocks):
        for j in range(mblocks):
            th = threading.Thread(target=dot_func, 
                                  args=(a_blocks[i, 0, :, :], 
                                        b_blocks[0, j, :, :], 
                                        out_blocks[i, j, :, :]))
            th.start()
            threads.append(th)

    for th in threads:
        th.join()

    return out

def expand_sparse_mat(mat, cell_shift_array, cell_shift_array_expand, cell_index_map):
    ncells = len(cell_shift_array)
    ncells_expand = len(cell_shift_array)+len(cell_shift_array_expand)
    norbs = mat.shape[1]
    
    mat_expand = np.zeros((ncells_expand, norbs, ncells_expand, norbs), dtype=mat.dtype)
    # 先用S来初始化ncells部分
    for m, Cm in enumerate(cell_shift_array): # ncells
        for n, Cn in enumerate(cell_shift_array): # ncells
            relative_cell_shift = tuple((Cn - Cm).tolist())
            if relative_cell_shift in cell_index_map:
                relative_cell_mn = cell_index_map[relative_cell_shift] 
                mat_expand[m,:,n,:] = mat[relative_cell_mn]
    
    # 再初始化expand部分
    for m, Cm in enumerate(cell_shift_array_expand): # ncells_expand
        for n, Cn in enumerate(cell_shift_array_expand): # ncells_expand
            relative_cell_shift = tuple((Cn - Cm).tolist())
            if relative_cell_shift in cell_index_map:
                relative_cell_mn = cell_index_map[relative_cell_shift] 
                mat_expand[m+ncells,:,n+ncells,:] = mat[relative_cell_mn]
    
    for m, Cm in enumerate(cell_shift_array): # ncells
        for n, Cn in enumerate(cell_shift_array_expand): # ncells_expand
            relative_cell_shift = tuple((Cn - Cm).tolist())
            if relative_cell_shift in cell_index_map:
                relative_cell_mn = cell_index_map[relative_cell_shift] 
                mat_expand[m,:,n+ncells,:] = mat[relative_cell_mn]
    
    for m, Cm in enumerate(cell_shift_array_expand): # ncells_expand
        for n, Cn in enumerate(cell_shift_array): # ncells
            relative_cell_shift = tuple((Cn - Cm).tolist())
            if relative_cell_shift in cell_index_map:
                relative_cell_mn = cell_index_map[relative_cell_shift] 
                mat_expand[m+ncells,:,n,:] = mat[relative_cell_mn]

def get_K_mesh(latt, g_rad):
    lat_per_inv=np.linalg.inv(latt).T
    pts1 = np.arange(0, g_rad[0])
    pts2 = np.arange(0, g_rad[1])
    pts3 = np.arange(0, g_rad[2])
    grid = np.meshgrid(pts1, pts2, pts3)
    for i in range(3):
        grid[i] = grid[i].ravel()/g_rad[i]
    grid = np.array(grid).T
    return np.tensordot(grid,lat_per_inv,axes=1)
            
    
    
