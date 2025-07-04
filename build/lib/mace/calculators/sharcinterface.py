from typing import Dict, List, Union

from sharc.pysharc.interface import SHARC_INTERFACE

from mace.calculators.sharc_calculator import SharcCalculator


class SHARC_NN(SHARC_INTERFACE):
    """
    Class for SHARC NN
    """
    # Name of the interface
    interface = 'NN'
    # store atom ids
    save_atids = True
    # store atom names
    save_atnames = True
    # accepted units:  0 : Bohr, 1 : Angstrom
    iunit = 0
    # not supported keys
    not_supported = ['nacdt', 'dmdr']

    def __init__(
        self,
        modelpath_e,
        modelpath_s,
        atoms: Union[List[int], str] = None,
        **kwargs
        ):
        print(modelpath_e)
        print(modelpath_s)
        self.spainn_init = SharcCalculator(atom_types=atoms, device="cuda", modelpath_e=modelpath_e, modelpath_s=modelpath_s, **kwargs)

    def initial_setup(self, **kwargs):
        pass

    def do_qm_job(self, tasks, Crd):
        result = self.spainn_init.calculate(Crd)
        return result

    def final_print(self):
        self.sharc_writeQMin()

    def readParameter(self, param,  *args, **kwargs):
        pass
