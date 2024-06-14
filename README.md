# HamEPC
Electron-phonon coupling (EPC) calculator based on [HamGNN](https://github.com/QuantumLab-ZY/HamGNN)

## Introduction
HamEPC is a machine learning workflow that leverages the HamGNN framework to efficiently calculate the electron-phonon coupling (EPC). By utilizing atomic orbital-based Hamiltonian matrices and gradients predicted by HamGNN, HamEPC is able to significantly accelerate EPC calculations compared to traditional density functional perturbation theory (DFPT) methods. HamEPC can be employed to evaluate important materials properties, including the electron-phonon coupling matrix, carrier mobility, and superconducting transition temperature. The script EPC_calculator.py defines `EPC_calculator` class which is used to calculate the epc-related properties. This script also needs HamGNN to predict the Hamiltonian matrix of a system.

## Requirements

The following Python libraries are required to calculate the epc values:
- NumPy
- PyTorch = 1.11.0
- phonopy
- spglib
- mpi4py
- opt_einsum
- pymatgen
- tqdm
- scipy
- yaml
- The C extension numpy_extension is also needed. The installation method is `pip install numpy_extension-0.0.3-cp39-cp39-manylinux1_x86_64.whl`.

## Installation
Run the following command to install HamGNN:
```bash
git clone https://github.com/QuantumLab-ZY/HamEPC.git
cd HamEPC
python setup.py install
```

## Usage
HamEPC supports hybrid parallelization of MPI and OpenMP. Users need to set the number of processes and threads reasonably in order to achieve optimal parallel efficiency and memory utilization.
```
mpirun -np ncores HamEPC --config EPC_input.yaml
```
## Theory

### Smearing Function<a id="gauss_type"></a>

We use some smearing methods to approximate the delta function.

+ `gauss_type >= 0`: Derivative of the [Methfessel-Paxton polynomial](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.40.3616), which can be written as:

    $$ f_{N}\left ( x \right ) = f_{0}\left ( x \right ) + \sum_{n=1}^{N} A_{n}H_{2n-1}\left ( x \right ) e^{-x^{2}} $$

    where

    $$ f_{0}\left ( x \right ) = \frac{1}{2} \left ( 1 - \mathrm{erf}\left ( x  \right )  \right ) $$

    $$ A_{n} = \frac{\left( -1 \right ) ^{n}}{n!4^{n}\sqrt{\pi}} $$

    and $H$ are the Hermite polynomials.

+ `gauss_type = -1`: Derivative of the [Cold smearing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.82.3296), which can be written as:

    $$ \frac{1}{\sqrt{\pi}}{e}^{-(x-\frac{1}{\sqrt{2}})^2}(2-\sqrt{2}x) $$

+ `gauss_type = -99`: Derivative of the Fermi-Dirac function, which can be written as:

    $$ \frac{1}{e^{x} + 1} $$

### Scatteirng Rate in Polar Materials<a id="scatt_split"></a>

For mobility calculation in polar materials, we divide the electron-phonon coupling term into two parts using the method in [Phys. Rev. B 94, 20 (2016)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.201201):

$$ {g}_{mn\nu}\left ( \mathbf{k}, \mathbf{q} \right ) = {g}^{\mathrm{S}}_{mn\nu}\left ( \mathbf{k}, \mathbf{q} \right ) + {g}^{\mathrm{L}}_{mn\nu}\left ( \mathbf{k}, \mathbf{q} \right ) $$

Then, we can split scattering rate into two parts, as:
$$\left [ \frac{1}{\tau} \right ]_{\mathrm{Polar} } = \frac{2\pi}{\hbar} \sum_{\mathbf{q} m\nu } {\left | {g}^{\mathrm{L}}_{mn\nu}\left ( \mathbf{k}, \mathbf{q} \right )  \right | }^{2} F_{mn\nu}\left ( \mathbf{k}, \mathbf{q} \right )$$

$$\left [ \frac{1}{\tau} \right ]_{\mathrm{Remainder} } = \frac{2\pi}{\hbar} \sum_{\mathbf{q} m\nu } \left ( {\left | {g}_{mn\nu}\left ( \mathbf{k}, \mathbf{q} \right )  \right | }^{2} - {\left | {g}^{\mathrm{L}}_{mn\nu}\left ( \mathbf{k}, \mathbf{q} \right )  \right | }^{2} \right )  F_{mn\nu}\left ( \mathbf{k}, \mathbf{q} \right )$$

When `polar_split = 'polar'`, the code will calculate the polar part of the scattering rate, and when `polar_split = 'rmp'`, the code will calculate the remainder part. In addition, when `polar_split = 'none'`, the code will calculate the scattering rate without spliting it into two parts.

### Monte Carlo Sampling<a id="monte_carlo"></a>

To ensure the effective convergence of the integration results on the $\mathbf{q}$ grid, we also employ Monte Carlo integration with importance sampling.

+ Uniform Sampling:

    $\mathbf{q}$ points will be randomly sampled using uniform distribution. Set `MC_sampling = 'uniform'` to use uniform sampling.

+ Cauchy Sampling:<a id="cauchy_sampling"></a>

    $\mathbf{q}$ points will be randomly sampled using cauchy distribution: 
    
    $$ f \left ( x \right ) = \frac{1}{\pi} \frac{\gamma}{x^{2} + \gamma^{2}}$$

    where $\gamma$ is the scale parameter of cauchy distribution. Set `MC_sampling = 'cauchy'` to use Cauchy sampling.

### Band Velocity<a id="velocity"></a>

For `read_momentum = True`, we calculate the band velocity of electrons using: 
$$ \mathbf{v}_{\mathbf{k}n,\alpha} = \frac{1}{\hbar} \frac{d\epsilon_{\mathbf{k}n}}{d\mathbf{k}_\alpha} = \frac{1}{\hbar} \langle n\mathbf{k} | \frac{dH_\mathbf{k}}{dk_\alpha} - \epsilon_{\mathbf{k}n}\frac{dS_\mathbf{k}}{dk_\alpha} | n\mathbf{k} \rangle $$

For `read_momentum = False`, we calculate the band velocity of electrons using:

$$ \mathbf{v}_{\mathbf{k}n,\alpha} = \langle n\mathbf{k} | p_\alpha | n\mathbf{k} \rangle $$

## Explanation of Input Parameters

To use this program, you should first set all the parameters in the basic block and then set the other blocks according to the required functionality.

### `basic`:

+ `cal_mode`: 

    Calculation mode. 

    + default:&nbsp; &nbsp; *MUST SET BY USER*

    + options:

        'band':&nbsp; &nbsp; Calculate the electronic bands of a specific k-path.

        'phonon'&nbsp; &nbsp; Calculate the phonon dispersion of a specific q-path.

        'epc':&nbsp; &nbsp; Calculate the electron-phonon coupling (EPC) elements of a specific k-path.

        'mobility':&nbsp; &nbsp; Mobility calculation.

        'superconduct':&nbsp; &nbsp; Superconductivity calculation.

+ `graph_data_path_uc`:
    
    The path of graph data file.

    + default:&nbsp; &nbsp; *MUST SET BY USER*

    + options:&nbsp; &nbsp; *STRING*

+ `nao_max`:

    The maximum number of atomic orbitals.

    + default:&nbsp; &nbsp; *MUST SET BY USER*

    + options:&nbsp; &nbsp;13,&nbsp; &nbsp;14,&nbsp; &nbsp;19,&nbsp; &nbsp;26

+ `Ham_type`: 

    The hamiltonian type in graph data.

    + default:&nbsp; &nbsp; *MUST SET BY USER*

    + options:
        
        'openmx':&nbsp; &nbsp; The Hamiltonian generated by openmx.
        
        'siesta':&nbsp; &nbsp; The Hamiltonian generated by siesta or Honpas.

+ `outdir`:

    The output directory.

    + default:&nbsp; &nbsp; *MUST SET BY USER*

    + options:&nbsp; &nbsp; *STRING*

### `advanced`:

+ `read_large_grad_mat`:

    If *True*, only three threads will read the gradient matrix simultaneously.

    + default:&nbsp; &nbsp; False

    + options:&nbsp; &nbsp; False,&nbsp; &nbsp;True

+ `split_orbits`: 

    If *True*, the orbit-splited gradient matrix will be read.

    + default:&nbsp; &nbsp; False

    + options:&nbsp; &nbsp; False,&nbsp; &nbsp;True

+ `split_orbits_num_blocks`:

    Some tricks to read large grad_mat file. if `split_orbits = False`, this parameter will be ignored.

    + default:&nbsp; &nbsp; 2

    + options:&nbsp; &nbsp; *POSITIVE INTEGER*
    
+ `soc_switch`:

    If *True*, the calculations will be done with spin-orbit coupling (SOC). And the data must include SOC.

    + default:&nbsp; &nbsp; False

    + options:&nbsp; &nbsp; False,&nbsp; &nbsp;True

### `dispersion`:

+ `high_symmetry_points`:

    The high symmetry points for calculating dispersions.

    + default:&nbsp; &nbsp; *GENERATE ATUOMATICALLY*

    + options:&nbsp; &nbsp; *LIST[LIST[FLOAT]]*,

+ `high_symmetry_labels`:

    The labels of high symmetry points in `high_symmetry_points`. Must set when `high_symmetry_points` is specified.

    + default:&nbsp; &nbsp; *GENERATE ATUOMATICALLY*

    + options:&nbsp; &nbsp; *LIST[STRING]*,

+ `nks_path`:

    The total number of k points on the high symmetry path.

    + default:&nbsp; &nbsp; 200

    + options:&nbsp; &nbsp; *POSITIVE INTEGER*

+ `dispersion_select_index`:

    When `calc_mode = 'band'`, it selects bands indices as '1-3, 4-7';

    When `calc_mode = 'phonon'`, it selects branches indices as '1-2, 4-6';

    When `calc_mode = 'epc'`, it must set as 'band_indice_of_initial_state, band_indice_of_final_state';

    Note that the indice starts from 1.

    + default:&nbsp; &nbsp; *ALL* for `calc_mode = 'band' or 'phonon'`, *MUST SET BY USER* for `calc_mode = 'epc'`

    + options:&nbsp; &nbsp; *STRING*

+ `epc_path_fix_k`:

    When `calc_mode = 'epc'`, it must set to be k point of the initial state, in fractional coordinate.

    + default:&nbsp; &nbsp; []

    + options:&nbsp; &nbsp; *LIST[FLOAT]*

### `phonon`:

+ `supercell_matrix`:

    The supercell matrix used in phonopy calculation.
        
    + default:&nbsp; &nbsp; [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    + options:&nbsp; &nbsp; *LIST[LIST[FLOAT]]*

+ `primitive_matrix`:

    The primitive matrix used in phonopy calculation.

    + default:&nbsp; &nbsp; [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    + options:&nbsp; &nbsp; *LIST[LIST[FLOAT]]*

+ `unitcell_filename`:

    The path of unitcell file used in phonopy calculation.

    + default:&nbsp; &nbsp; "./POSCAR"

    + options:&nbsp; &nbsp; *STRING*

+ `force_sets_filename`:

    The path of FORCE_SETS from phonopy calculation.

    + default:&nbsp; &nbsp; "./FORCE_SETS"

    + options:&nbsp; &nbsp; *STRING*

+ `apply_correction`:

    If *True*, apply the long range correction (LRC).
    
    + default:&nbsp; &nbsp; False

    + options:&nbsp; &nbsp; False,&nbsp; &nbsp;True

+ `q_cut`:

    If `apply_correction = True`, it will give a cutoff that only q points inside will include LRC. The unit of it is the length of the first recoprocal vector.

    + default:&nbsp; &nbsp; 100.0

    + options:&nbsp; &nbsp; *FLOAT*

+ `BECs`:

    The Born effective charge for LRC.

    + default:&nbsp; &nbsp; []

    + options:&nbsp; &nbsp; *LIST[LIST[FLOAT]]*

+ `DL`:

    The electron dielectric constant for LRC.

    + default:&nbsp; &nbsp; []

    + options:&nbsp; &nbsp; *LIST[LIST[FLOAT]]*

### `epc`:

+ `grad_mat_path`:

    The path of gradient matrix file. This file contains the gradients of the Hamiltonian matrix in the basis of real-space atomic orbitals.

    + default:&nbsp; &nbsp; "./grad_mat.npy"

    + options:&nbsp; &nbsp; *STRING*

+ `mat_info_rc_path`:

    The path of mat_info_rc. This file contains supercell infomation and the mapping to the atomic index in the primitive cell.

    + default:&nbsp; &nbsp; "./mat_info_rc.npy"

    + options:&nbsp; &nbsp; *STRING*

+ `cell_range_cut`:

    The cut range for the sum over the supercell index for calculating the EPC matrix.

    + default:&nbsp; &nbsp; [2,2,2]

    + options:&nbsp; &nbsp; *LIST[POSITIVE INTEGER]*

### `transport`:

+ `k_size`:

    The size of k grid for integral over Brillouin Zone (BZ).

    + default:&nbsp; &nbsp; [32,32,32]

    + options:&nbsp; &nbsp; *LIST[POSITIVE INTEGER]*

+ `q_size`:

    The size of q grid for integral over Brillouin Zone (BZ).

    + default:&nbsp; &nbsp; [32,32,32]

    + options:&nbsp; &nbsp; *LIST[POSITIVE INTEGER]*

+ `bands_indices`:

    The band indices related to transport properties, starting from 1.

    + default:&nbsp; &nbsp; []

    + options:&nbsp; &nbsp; *LIST[POSITIVE INTEGER]*

+ `maxarg`:

    Cutoff set to prevent the exponential term of the Gaussian function from being too small.

    + default:&nbsp; &nbsp; 200

    + options:&nbsp; &nbsp; *FLOAT*

+ `fermi_maxiter`:

    The maximum iteration in bisection method, while calculating the fermi level.

    + default:&nbsp; &nbsp; 100

    + options:&nbsp; &nbsp; *POSITIVE INTEGER*

+ `temperature`:

    Temperature of the system, in Klevin.

    + default:&nbsp; &nbsp; 300

    + options:&nbsp; &nbsp; *FLOAT*


+ `phonon_cutoff`:

    Phonon energy cutoff in meV. The phonon with energy less than that cutoff will be ignored.

    + default:&nbsp; &nbsp; 2.0

    + options:&nbsp; &nbsp; *FLOAT*

+ `smeark`:

    Smearing of the delta function for the summation of electric states, in meV.

    + default:&nbsp; &nbsp; 25.0

    + options:&nbsp; &nbsp; *FLOAT*

+ `smearq`:

    Smearing of the delta function for the summation of phonon states, in meV.

    + default:&nbsp; &nbsp; 25.0

    + options:&nbsp; &nbsp; *FLOAT*

+ `gauss_type`:

    The type of gaussian function. For more detail, please see [here](#gauss_type).

    + default:&nbsp; &nbsp; 0

    + options:&nbsp; &nbsp; -99,&nbsp; &nbsp;-1,&nbsp; &nbsp;0,&nbsp; &nbsp;*POSITIVE INTEGER*

+ `e_thr`:

    Energy threshold for match table in meV.

    + default:&nbsp; &nbsp; 75.0

    + options:&nbsp; &nbsp; *FLOAT*

### `mobility`:

+ `read_momentum`:

    If *True*, read the momentum in graph data, and use it to calculate the band velocity. For more detail, please see [here](#velocity).

    + default:&nbsp; &nbsp; False

    + options:&nbsp; &nbsp; False,&nbsp; &nbsp;True

+ `over_cbm`:

    Maximum electron energy that considered in the mobility calculation in eV, and the energy scale is referenced to CBM.

    + default:&nbsp; &nbsp; 0.2

    + options:&nbsp; &nbsp; *FLOAT*    

+ `MC_sampling`:

    The Monte Carlo samling method. For more detail, please see [here](#monte_carlo).

    + default:&nbsp; &nbsp; "none"

    + options:&nbsp; &nbsp; "none",&nbsp; &nbsp;"cauchy",&nbsp; &nbsp;"uniform"

+ `polar_split`:

    Set to `polar` to calculate the polar part of the scattering rate, while set to `rmp` to calculate the remainder part of the scattering rate. `none` for not spliting. For more detail, please see [here](#scatt_split).

    + default:&nbsp; &nbsp; "none"

    + options:&nbsp; &nbsp; "none",&nbsp; &nbsp;"polar",&nbsp; &nbsp;"rmp"

+ `cauchy_scale`:

    The scale parameter of cauchy distribution. For more detail, please see [here](#cauchy_sampling).

    + default:&nbsp; &nbsp; 0.035

    + options:&nbsp; &nbsp; *FLOAT*

+ `sampling_seed`:

    The random seed for Monte Carlo sampling.

    + default:&nbsp; &nbsp; 1

    + options:&nbsp; &nbsp; *POSITIVE INTEGER*

+ `nsamples`:

    The number of sampling q points.

    + default:&nbsp; &nbsp; 1000000

    + options:&nbsp; &nbsp; *POSITIVE INTEGER*

+ `ncarrier`: 

    The carrier concentration in $\mathrm{cm^{-3}}$.

    + default:&nbsp; &nbsp; 1.0E+16

    + options:&nbsp; &nbsp; *FLOAT*

+ `ishole`:

    Need to be *False* in this version, since we can only calculate electron transport properties.

    + default:&nbsp; &nbsp; False

    + options:&nbsp; &nbsp; False

+ `mob_level`:

    The approximation for mobility calculation. Only *"ERTA"* is allowed in this version.

    + default:&nbsp; &nbsp; "ERTA"

    + options:&nbsp; &nbsp; "ERTA"

+ `polar_rate_path`:

    The path of polar-part scattering rate file. If set, the programme will read the scattering rate from this file, and then calculate the mobility. Note that, if `rmp_rate_path` is also set, the programme will read both two files, and sum up the scattering rates.

    + default:&nbsp; &nbsp; ""

    + options:&nbsp; &nbsp; *STRING*

+ `rmp_rate_path`:

    The path of remainder-part scattering rate file. If set, the programme will read the scattering rate from this file, and then calculate the rmp-only mobility. Note that, if `polar_rate_path` is also set, the programme will read both two files, and sum up the scattering rates.

    + default:&nbsp; &nbsp; ""

    + options:&nbsp; &nbsp; *STRING*

### `superconduct`:

+ `mius`:

    The effective Coulomb potential in Allen-Dynes formula. Must be set as [miu_min, miu_max, miu_step].

    + default:&nbsp; &nbsp; [0.05, 0.25, 0.01]

    + options:&nbsp; &nbsp; *LIST[FLOAT]*

+ `omega_range`:

    The phonon energy range, while calculating ${\alpha}^{2}F$ spectrum, in meV. Must be set as [freq_min, freq_max].

    + default:&nbsp; &nbsp; [0, 100]

    + options:&nbsp; &nbsp; *LIST[FLOAT]*

+ `omega_step`:

    The phonon energy step, while calculating ${\alpha}^{2}F$ spectrum, in meV.

    + default:&nbsp; &nbsp; 0.01

    + options:&nbsp; &nbsp; *FLOAT*
