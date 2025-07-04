import os
from typing import Dict, Union, List

import ase
import numpy as np
import torch
from sympy.physics.quantum.cg import CG
from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_load
from mace.calculators import MACECalculator
import time

def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")

__all__ = ["SPaiNNulator"]


class SPaiNNulatorError(Exception):
    """
    SpaiNNulator error class
    """


class ThresholdError(Exception):
    """
    If model threshold exeeded
    """


symbols = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F","Fe","I"]


class SharcCalculator:
    """
    Interface between SHARC and SchNetPack 2.0
    """

    def __init__(
        self,
        atom_types: Union[np.ndarray, torch.Tensor, str] = None,
        modelpath_e: Union[List[str], str] = "",
        modelpath_s: Union[List[str], str] = "",
        cutoff: float = 5,
        device: str = "cuda",
        properties: List[str] = None,
        n_states: Dict[str, int] = None,
        thresholds: Dict[str, float] = None,
        nac_key: str = "nacs",
        soc_key: str = "socs",
        energy_unit: str = "eV",
        distance_unit: str = "Angstrom",
    ):
        """
        Parameters
        ----------

        atom_types: atomic charges or string of atoms
        modelpath: path(s) to trained model(s) or folder(s) with 'best_inference_model'
            for adaptive sampling
        cutoff: cutoff value
        properties: list of properties returned to SHARC
        n_states: dictionary of calculated states
        thresholds: dictionary of threshold values
        
        Examples
        ---------

        You can use this calculator to perform predictions of properties using a 
        trained NN model.

        Here we show, how to predict the energies for a target molecule. First we
        import all necessary modules followed by the creation of an `ase.Atoms` 
        object of the target molecule.

        >>> import os, sys
        >>> import numpy as np
        >>> import ase
        >>> 
        >>> symbols = 'CNHHHH'
        >>> positions = np.array(
        >>>     [[ 0.0000,  0.0000,  0.0000 ],
        >>>      [ 2.4321,  0.0000,  0.0000 ],
        >>>      [-1.0111,  1.7951,  0.0000 ],
        >>>      [ 3.4373,  1.6202, -0.2566 ],
        >>>      [ 3.4373, -1.6202,  0.2566 ],
        >>>      [-1.0111, -1.7951,  0.0000 ]]
        >>> )
        >>> # create ase Atoms object
        >>> target_mol = ase.Atoms(symbols=symbols, positions=positions)

        Next, we define the calculator `NacCalculator` used to predict the energy
        of the target molecule and perform the prediction.

        >>> from schnetpack.transform import MatScipyNeighborList
        >>> from spainn.interface import NacCalculator
        >>> 
        >>> calc = NacCalculator(
        >>>     model_file=os.path.join(os.getcwd(), 'train', 'best_model'),
        >>>     neighbor_list=MatScipyNeighborList(cutoff=10.0)
        >>> )
        >>> target_mol.calc = calc
        >>> # make prediction
        >>> pred = target_mol.get_properties(['energy'])
        """
        distance_units = {"Angstrom": 0.529177249, "Bohr": 1.0}
        energy_units = {"eV": 0.0367493, "Hartree": 1.0}

        self.energy_unit_conversion = energy_units[energy_unit]
        self.distance_unit_conversion = distance_units[distance_unit]

        # Load model and setup molecule
        self.modelpath_e = modelpath_e
        self.modelpath_s = modelpath_s
        self.properties = properties or ["energy", "forces", nac_key, soc_key]
        self.nac_key = nac_key
        self.soc_key = soc_key

        if atom_types is None:
            raise SPaiNNulatorError("atom_types has to be set")

        # Load model and setup molecule
        self.modelpath_e = [modelpath_e] if isinstance(modelpath_e, str) else modelpath_e
        self.modelpath_s = [modelpath_s] if isinstance(modelpath_s, str) else modelpath_s

        if isinstance(atom_types, str):
            atom_types = np.array([26] + [symbols.index(c) for c in atom_types[2:].upper()])

        self.molecule = [
            ase.Atoms(symbols=atom_types) for _ in range(len(self.modelpath_e))
        ]
        self.thresholds = thresholds
        self.atom_types = atom_types
        self.nac_key = nac_key

        # Setup states and matrix masks
        if n_states is None:
            raise SPaiNNulatorError("n_states dict has to be set!")

        all_states={
        'n_singlets': 0, 
        'n_doublets': 0, 
        'n_triplets': 0,
        'n_quartets': 0,
        'n_quintets': 0}
        multiplicities = np.array([1,2,3,4,5])
        self.multiplicities = multiplicities
        for key in n_states:
            if key in all_states:
                all_states[key] = n_states[key]
        self.n_states = all_states
        values = list(all_states.values())

        self.n_total_states = np.sum(multiplicities * values)

        self.n_atoms = len(atom_types)

        self.nac_idx = np.triu_indices(self.n_states["n_singlets"], 1)
        self.dm_idx = np.triu_indices(self.n_states["n_singlets"], 0)
        self.soc_idx = np.triu_indices(self.n_total_states, 1)

        self.last_prediction = None

        self.energy_calcs = []
        self.soc_calcs = []
        
        for idx, (val_e, val_s) in enumerate(zip(self.modelpath_e, self.modelpath_s)):
            self.energy_calcs.append(MACECalculator(model_paths=val_e, device=device))
            self.soc_calcs.append(MACECalculator(model_paths=val_s, device=device))

    def calculate(
        self, sharc_coords: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Calculate properties from new positions.

        If multiple models are used, the average values between
        the two predictions with the lowest NAC MAE will be returned

        Parameters
        ----------

        sharc_coords: Coordinates from SHARC simulation
        """
        spainn_output = []

        for i in range(len(self.modelpath_e)):
            self.molecule[i].set_positions(sharc_coords)
            self.energy_calcs[i].calculate(self.molecule[i])
            self.soc_calcs[i].calculate(self.molecule[i])
            results = self.energy_calcs[i].results
            results["energy"] = results["energy"].squeeze()
            #F = results["forces"]
            # print(F[2])
            # F[:, [1, 4, 7, 10], 0] = 0
            # F[:, [1, 4, 7, 10], 1] = 0
            # F[:, [2, 3, 8, 9], 1] = 0
            # F[:, [2, 3, 8, 9], 2] = 0
            # F[:, [11, 5, 6, 12], 0] = 0
            # F[:, [11, 5, 6, 12], 2] = 0
            # print(F[2])
            # atoms = [1, 3, 5, 2, 4, 6]

            # for a in atoms:
            #     for i in range(F.shape[0]):
            #         vec = F[i, a]
            #         mask = vec != 0
            #         if mask.any():
            #             avg = np.mean(np.abs(vec[mask]))
            #             F[i, a, mask] = np.sign(vec[mask]) * avg
            # print(F[2])
            #time.sleep(1000)

            
            # results["forces"] = F
            results["socs"] = self.soc_calcs[i].results["energy"]
            spainn_output.append(results)

        # Save first prediction for phase tracking
        if self.last_prediction is None:
            self.last_prediction = spainn_output[0]

        if len(self.modelpath_e) == 1:
            for prop in self.properties:
                if prop not in ["energy", "forces"]:
                    spainn_output[0][prop] = self._adjust_phase(
                        self.last_prediction[prop], spainn_output[0][prop]
                    )
            self.last_prediction = spainn_output[0]
            return self.get_qm(spainn_output[0])

        # Adjust phases relative to first model
        for prop in self.properties:
            if prop not in ["energy", "forces"]:
                for idx, val in enumerate(spainn_output):
                    spainn_output[idx][prop] = self._adjust_phase(
                        self.last_prediction[prop], spainn_output[idx][prop]
                    )

        # Check if Thresholds exeeded
        if self.thresholds is not None:
            prop_mae = {
                key: np.mean(np.abs(val - spainn_output[1][key]))
                for (key, val) in spainn_output[0].items()
            }
            below_threshold = all(prop_mae[k] < v for (k, v) in self.thresholds.items())
            if not below_threshold:
                self._write_xyz(sharc_coords)
                raise ThresholdError("Threshold exeeded.")

        # Save last prediction
        self.last_prediction = spainn_output[0]
        return self.get_qm(spainn_output[0])

    def get_qm(self, spainn_output: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Calculate QM string for SHARC with predictions from model.
        """

        states = self.n_total_states
        #to be generalised
        n_singlets = self.n_states["n_singlets"]
        n_triplets = self.n_states["n_triplets"]
        qm_out = {}
        qm_out["h"]=np.zeros((states))
        #expand energies according to multiplicity (sharc format)
        energies=np.array(spainn_output["energy"], dtype=complex)
        expanded_energies = []  
        start_idx = 0
        for i, (state, value_count) in enumerate(self.n_states.items()):
            repeat_count = self.multiplicities[i]
            slice_en = energies[start_idx:start_idx + value_count]
            expanded_slice = np.tile(slice_en, repeat_count)
            expanded_energies.extend(expanded_slice)
            start_idx += value_count

        # Convert energy array to complex diagonal matrix
        qm_out["h"] = np.diag(expanded_energies).tolist()

        #expand forces according to multiplicity
        forces_spinfree = spainn_output["forces"]
        expanded_forces_list = []
        for j in range(len(forces_spinfree)):
            start_idx = 0
            atom_forces = []
            for i, (state, value_count) in enumerate(self.n_states.items()):
                repeat_count = self.multiplicities[i]
                slice_forc = forces_spinfree[j][start_idx:start_idx + value_count]
                expanded_slice = np.tile(slice_forc, (repeat_count,1))
                atom_forces.append(expanded_slice)
                start_idx += value_count
            expanded_forces_list.append(np.vstack(atom_forces))  
        expanded_forces = np.array(expanded_forces_list)
        


        # Reshape force array from [atoms, states, coords] to [states, atoms, coords]
#        qm_out["grad"] = np.einsum("ijk->jik", -spainn_output["forces"]).tolist()
        qm_out["grad"] = np.einsum("ijk->jik", -expanded_forces).tolist()
        if self.nac_key in spainn_output:
            nacs_v = np.einsum("ijk->jik", spainn_output[self.nac_key])
            nacs_m = np.zeros((states, states, self.n_atoms, 3))
 
            if n_triplets == 0:
                nacs_m[self.nac_idx] = nacs_v
                nacs_m -= np.transpose(nacs_m, axes=(1, 0, 2, 3))
            else:
                nacs_singlet = np.zeros((n_singlets, n_singlets, self.n_atoms, 3))
                nacs_singlet[self.nac_idx] = nacs_v[
                    0 : int(n_singlets * (n_singlets - 1) / 2)
                ]
                nacs_singlet -= nacs_singlet.T
 
                nacs_m[0:n_singlets, 0:n_singlets] = nacs_singlet
 
                nacs_trip_sub = np.zeros((n_triplets, n_triplets, self.n_atoms, 3))
                sub_idx = np.triu_indices(n_triplets, 1)
                nacs_trip_sub[sub_idx] = nacs_v[int(n_singlets * (n_singlets - 1) / 2) :]
                nacs_trip_sub -= nacs_trip_sub.T
 
                nacs_trip = np.zeros((3 * n_triplets, 3 * n_triplets, self.n_atoms, 3))
 
                for i in range(3):
                    for j in range(i, 3):
                        nacs_trip[
                            i * n_triplets : (i + 1) * n_triplets,
                            j * n_triplets : (j + 1) * n_triplets,
                        ] = nacs_trip_sub
 
                trip_idx = np.tril_indices(3 * n_triplets)
                nacs_trip[trip_idx] = 0
                nacs_trip -= nacs_trip.T
 
                nacs_m[n_singlets:, n_singlets:] = nacs_trip
            qm_out["nacdr"] = nacs_m.tolist()

        if "dipoles" in self.properties:
            dm_m = np.zeros((states, states, 3), dtype=complex)
            if n_triplets == 0:
                dm_m[self.dm_idx] = spainn_output["dipoles"]
                dm_m += dm_m.T
                dm_m[self.dm_idx] = spainn_output["dipoles"]
            else:
                dm_singlets = np.zeros((n_singlets, n_singlets, 3), dtype=complex)
                dm_singlets[self.dm_idx] = spainn_output["dipoles"][
                    : int(n_singlets * (n_singlets - 1) / 2) + n_singlets
                ]
                dm_singlets += dm_singlets.T
                dm_singlets[self.dm_idx] = spainn_output["dipoles"][
                    : int(n_singlets * (n_singlets - 1) / 2) + n_singlets
                ]

                dm_m[0:n_singlets, 0:n_singlets] = dm_singlets

                dm_triplets = np.zeros((n_triplets, n_triplets, 3), dtype=complex)
                trip_idx = np.triu_indices(n_triplets, 0)
                dm_triplets[trip_idx] = spainn_output["dipoles"][
                    int(n_singlets * (n_singlets - 1) / 2) + n_singlets :
                ]
                dm_triplets += dm_triplets.T
                dm_triplets[trip_idx] = spainn_output["dipoles"][
                    int(n_singlets * (n_singlets - 1) / 2) + n_singlets :
                ]

                for i in range(0, 3):
                    dm_m[
                        n_singlets + i * n_triplets : n_singlets + i * n_triplets
                    ] = dm_triplets

            dm_m = np.einsum("ijk->kij", dm_m)
            qm_out["dm"] = dm_m.tolist()

        if "socs" in self.properties:
             
             #extract H_soc
             soc_m = H_soc(spainn_output["socs"], energies, self.n_states)
             qm_out["h"] = soc_m.tolist()


        return qm_out

    def _check_modelpath(self) -> None:
        """
        Check if valid path(s) given.
        """
        for path in self.modelpath_e:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"'{path}' does not exist!")

    def _adjust_phase(
        self, primary_phase: np.ndarray, secondary_phase: np.ndarray
    ) -> np.ndarray:
        """
        Function to align the phases of the two predictions.
        """

        # Make sure only NACS are transformed
        is_nac = bool(len(primary_phase.shape) > 2)
        if is_nac:
            primary_phase = np.einsum("ijk->jik", primary_phase)
            secondary_phase = np.einsum("ijk->jik", secondary_phase)

        # Adjust phases
        for idx, val in enumerate(secondary_phase):
            if np.vdot(val, primary_phase[idx]) < 0:
                secondary_phase[idx] *= -1
        return np.einsum("ijk->jik", secondary_phase) if is_nac else secondary_phase

    def _write_xyz(self, coords: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Write rejected geometry to xyz file.
        """
        with open("aborted.xyz", "w", encoding="utf-8") as output:
            output.write(f"{self.n_atoms}\n")
            output.write("Rejected geometry\n")
            for idx, val in enumerate(coords):
                output.write(
#                    f"{symbols[self.atom_types[idx]]}\t{val[0]:12.8f}\t{val[1]:12.8f}\t{val[2]:12.8f}\n"
f"{self.atom_types[idx]}\t{val[0]:12.8f}\t{val[1]:12.8f}\t{val[2]:12.8f}\n"
)


#Functions to rebuild H_soc using wigner-eckart theorem
     

def H_soc(flat_rmes, energies, states):

    #reshape rmes values 
    rmes = flat_rmes.reshape(1, int(len(flat_rmes[0])/3), 3)

    #read rmes_idx from calc folder (to improve...)
    rmes_idx = np.loadtxt(os.path.join(os.getcwd(),'RME_idx_NN.txt'))
    rmes_idx = np.array([rmes_idx])

    #extract multiplicities and roos. NB: ORCA ordering [5,4,3,2,1]
    mult = []
    roots = []
    for key_idx, key in enumerate(states):
        if states[key] != 0:
            mult.append(key_idx+1)
            roots.append(states[key])
    mult.reverse() #reverse for ORCA ordering
    roots.reverse() #reverse for ORCA ordering

    #recast real energies. NB: SCHARC ordering [1,2,3,4,5]
    energies = np.array([energies.real])

    #compute Hamiltonian in ORCA format
    H_soc_orca = RMES2H_ORCA(rmes, rmes_idx, energies , mult, roots)


    #convert in SHARC format
    H_soc_sharc_real = orca2sharc_matrix(
            H_soc_orca.real,
            mult,
            roots,
            'soc')
    H_soc_sharc_imag = orca2sharc_matrix(
            H_soc_orca.imag,
            mult,
            roots,
            'soc')

    H_soc_sharc = H_soc_sharc_real + 1j*H_soc_sharc_imag
    
    return np.array(H_soc_sharc)


def RMES2H_ORCA(rmes, rmes_idx, energies , mult, roots):


    index_1 = []
    index_2 = []
    soc_val = []
    au2cm = 2.1947463136314e+5
    cm2au = 1/au2cm


    for element_idx, RME in zip(rmes_idx[0] , rmes[0]):

        #get quantum numbers
        S1 = float(element_idx[0])
        M1 = int(2*S1+1)
        block1 = mult.index(M1)
        root1 = element_idx[2]


        S2 = float(element_idx[1])
        M2 = int(2*S2+1)
        block2 = mult.index(M2)
        root2 = element_idx[3]

        if abs(S1-S2) < 2:
            if S1==S2 and root1==root2:
                idx1, idx2, val = build_soc(block1, block2 , M1, M2, S1, S2, root1, root2, RME)
                index_1.extend(idx1)
                index_2.extend(idx2)
                soc_val.extend(val)
                if M1==M2:
                    idx1, idx2, val = build_soc(block1, block2 , M1, M2, S1, S2, root2, root1, -RME)
                    index_1.extend(idx1)
                    index_2.extend(idx2)
                    soc_val.extend(val)
            else:
                entry_idx = [S1, S2, root1, root2]
                entry_val = [RME[0], RME[1], RME[2]]
                idx1, idx2, val = build_soc(block1, block2 , M1, M2, S1, S2, root1, root2, RME)
                index_1.extend(idx1)
                index_2.extend(idx2)
                soc_val.extend(val)
                if M1==M2:
                    idx1, idx2, val = build_soc(block1, block2 , M1, M2, S1, S2, root2, root1, -RME)
                    index_1.extend(idx1)
                    index_2.extend(idx2)
                    soc_val.extend(val)

    #expand energies over roots
    dim_spinfree = len(energies[0])
    reversed_spinfree_en = list(energies[0][::-1]) #reverse order
    count = 0
    tmp_vec = []
    for block in range(len(roots)):
        tmp_block=[]
        for root in range(roots[block]): #NB: roots is already in the correct QTS format
            tmp_block.append(reversed_spinfree_en[root+count])
        count += roots[block]
        tmp_block.reverse() #reverse each multiplicity block
        for j in range(mult[block]): #expand over Ms
            tmp_vec.append(tmp_block)

    #concatenate elements of tmp_vec
    diag_H_soc = np.array(sum(tmp_vec, []))

    #build diagonal H_soc
    dim = len(diag_H_soc)
    H_SOC = np.matrix(np.zeros((dim, dim), dtype = np.complex128))
    H_SOC += np.diag(diag_H_soc)*au2cm

    #get H_soc indexes
    indexes_HSOC=[]
    for block, m in enumerate(mult):
        S = float((m-1)/2) #multiplicity to spin
        for m_idx in range(m):
            ms = S - m_idx #loop over ms from S to -S
            for state in range(roots[block]):
                indexes_HSOC.append([block, ms, state])

    #build off-diagonal H_soc    
    for i_idx, i in enumerate(indexes_HSOC):
        for j_idx, j in enumerate(indexes_HSOC):
            for soc_idx1, soc_1 in enumerate(index_1):
                if soc_1 == i:
                    if  index_2[soc_idx1] == j:
                        H_SOC[i_idx,j_idx] = soc_val[soc_idx1].real + 1j*soc_val[soc_idx1].imag
                        H_SOC[j_idx,i_idx] = soc_val[soc_idx1].real - 1j*soc_val[soc_idx1].imag




    return H_SOC*cm2au

def build_soc(block1, block2, M1, M2, S1,S2,root1,root2,RME):

    index_1 =[]
    index_2 =[]
    soc_val = []

    if M1==M2:
        L0 = 0+1j*RME[2]/2
        LP = RME[1]/2/np.sqrt(2) + 1j*RME[0]/2/np.sqrt(2)
        for idx1 in range(M1):
            ms1 = S1-idx1
            for idx2 in range(M2):
                ms2 = S2-idx2
                if ms1 == ms2:
                    fac = float(CG(S2,ms2,1,0,S1,ms1).doit())*(np.sqrt(S1*(S1+1))/S1)
                    soc_element = fac*L0
                    index_1.append([block1, ms1, root1])
                    index_2.append([block2, ms2, root2])
                    soc_val.append(soc_element)

                elif  ms2 == ms1-1 :
                    fac = -float(CG(S2,ms2,1,1,S1,ms1).doit())*(np.sqrt(S1*(S1+1))/S1)
                    soc_element = fac*LP
                    index_1.append([block1, ms1, root1])
                    index_2.append([block2, ms2, root2])
                    soc_val.append(soc_element)


    elif M1!=M2:
        L0 = 0+1j*RME[2]
        LM = RME[1]/np.sqrt(2) -1j*RME[0]/np.sqrt(2)
        LP = RME[1]/np.sqrt(2) +1j*RME[0]/np.sqrt(2)
        for idx1 in range(M1):
            ms1 = S1-idx1
            for idx2 in range(M2):
                ms2 = S2-idx2
                if ms1 == ms2:
                    fac = -float(CG(S2,ms2,1,0,S1,ms1).doit())/np.sqrt(2)
                    soc_element = fac*L0
                    index_1.append([block1, ms1, root1])
                    index_2.append([block2, ms2, root2])
                    soc_val.append(soc_element)

                elif ms1 == ms2-1:
                    fac = float(CG(S2,ms2,1,-1,S1,ms1).doit())/np.sqrt(2)
                    soc_element = fac*LM
                    index_1.append([block1, ms1, root1])
                    index_2.append([block2, ms2, root2])
                    soc_val.append(soc_element)

                elif ms1 == ms2+1:
                    fac = float(CG(S2,ms2,1,1,S1,ms1).doit())/np.sqrt(2)
                    soc_element = fac*LP
                    index_1.append([block1, ms1, root1])
                    index_2.append([block2, ms2, root2])
                    soc_val.append(soc_element)

    
    return index_1, index_2, soc_val


def orca2sharc_matrix(matrix, M, R, representation):

    matrix = np.array(matrix)

    #Transform in H_soc representation
    if representation == 'soc':

        #for each ms, the roots have to be reversed
        end = 0
        for block, (multiplicity, roots) in enumerate(zip(M, R)):
            for i in range(multiplicity):
                  start = end  #block*multiplicity*roots + end + i*roots
                  end   = start + roots
                  indices = list(range(start, end))
                  reversed_indices = indices[::-1]
                  # Swap rows and columns based on the range
                  matrix[indices, :] = matrix[reversed_indices, :]
                  matrix[:, indices] = matrix[:, reversed_indices]


        #reverse the matrix:
        #matrix[::-1] is the matrix with all rows reversed.
        #for each row, row[::-1] reverse the columns
        new_matrix = [row[::-1] for row in matrix[::-1]]
        new_matrix = np.array(new_matrix)


    elif representation == 'spinfree':

        #for each ms, the roots have to be reversed
        end = 0
        for block, roots in enumerate(R):
                  start = end  #block*roots + end + i*roots
                  end   = start + roots
                  indices = list(range(start, end))
                  reversed_indices = indices[::-1]
                  # Swap rows and columns based on the range
                  matrix[indices, :] = matrix[reversed_indices, :]
                  matrix[:, indices] = matrix[:, reversed_indices]


        #reverse the matrix:
        #matrix[::-1] is the matrix with all rows reversed.
        #for each row, row[::-1] reverse the columns
        new_matrix = [row[::-1] for row in matrix[::-1]]
        new_matrix = np.array(new_matrix)

    return new_matrix