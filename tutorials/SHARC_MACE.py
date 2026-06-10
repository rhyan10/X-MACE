#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2019 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************

# IMPORTS
import datetime
import numpy as np
import torch
# internal
from SHARC_FAST import SHARC_FAST
from utils import *
from io import TextIOWrapper
from mace.calculators.sharc_calculator import SharcCalculator

authors = 'Rhyan Barrett'
version = '4.0'
versiondate = datetime.datetime(2023, 7, 15)
description = ""
name = "MACE"

changelogstring = ''

class SHARC_MACE(SHARC_FAST):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.interface_name = "MACE"
        # Add template keys

        QMin = self.QMin
        QMin.template.types={
                "model_file":str,
                "cutoff": float,
                "properties": list,
                "device": str,
                "energy_unit": str,
                "distance_unit": str,
                "paddingstates": bool,
                }
        QMin.template.data={
                "model_file": None,
                "cutoff": 5.0,
                "properties": ["energy", "forces"],
                "device": "cpu",
                "energy_unit": "eV",
                "distance_unit": "Ang",
                "paddingstates": False,
                }

        self.spainnulator = None

    @staticmethod
    def version() -> str:
        return version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return versiondate

    @staticmethod
    def changelogstring() -> str:
        return changelogstring

    @staticmethod
    def authors() -> str:
        return authors

    @staticmethod
    def name() -> str:
        return name

    @staticmethod
    def description() -> str:
        return description

    @staticmethod
    def about() -> str:
        return f"{name}\n{description}"

    def get_features(self, KEYSTROKES: TextIOWrapper = None) -> set:
        return {
            "h",
            "soc",
            "dm",
            "grad",
            "nacdr",
            "point_charges",
        }

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'MACE interface setup':^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        if os.path.isfile("MACE.template"):
            self.log.info("Found MACE.template in current directory")
            if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                self._template_file = "MACE.template"
        else:
            self.log.info("Specify a path to a MACE template file.")
            while not os.path.isfile(template_file := question("Template path:", str, KEYSTROKES=KEYSTROKES)):
                self.log.info(f"File {template_file} does not exist!")
            self._template_file = template_file

        if question("Do you have a MACE.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            while not os.path.isfile(
                resources_file := question("Specify path to MACE.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
            ):
                self.log.info(f"File {resources_file} does not exist!")
            self._resources_file = resources_file
        else:
            self.log.info(f"{'MACE resource usage':-^60}\n")
            self.setupINFOS["modelpath"] = question("Specify path to MACE model: ", str, KEYSTROKES=KEYSTROKES)
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self._resources_file:
            with open(os.path.join(dir_path, "MACE.resources"), "w", encoding="utf-8") as file:
                if "modelpath" in self.setupINFOS:
                    file.write(f"modelpath {self.setupINFOS['modelpath']}\n")
                else:
                    self.log.error("Modelpath not specified!")
                    raise ValueError
        else:
            create_file(expand_path(self._resources_file), os.path.join(dir_path, "MACE.resources"))
        create_file(expand_path(self._template_file), os.path.join(dir_path, "MACE.template"))

    def read_resources(self, resources_filename="MACE.resources"):
        super().read_resources(resources_filename)

    def read_template(self, template_filename='MACE.template'):
        '''reads the template file
        has to be called after setup_mol!'''
        super().read_template(template_filename)

    def setup_interface(self):
        super().setup_interface()
        self.models = SharcCalculator(
            atom_types=self.QMin.molecule["elements"],
            model_path=self.QMin.template["model_file"],
            device=self.QMin.template["device"],
            energy_unit=self.QMin.template["energy_unit"],
            distance_unit=self.QMin.template["distance_unit"],
            n_states={"n_singlets": self.QMin.molecule["states"][0], "n_triplets": 0},
            properties=self.QMin.template["properties"]
        )

    def create_restart_files(self):
        pass

    def run(self):
        NN_out = self.models.calculate(self.QMin.coords["coords"])

        #'dm', 'nacdr', 'h', 'grad', 'dydf', 'pc_grad'
        self.log.debug(NN_out.keys())
        for key in NN_out.keys():
            match key:

                case "h":
                    self.log.debug(key)
                    self.QMout.h = np.array(NN_out["h"])
                case "grad":
                    self.log.debug(key)
                    self.QMout.grad = np.array(NN_out["grad"])
                case "dydf":
                    pc_grads = self.get_pc_grad(NN_out["dydf"])
                    self.QMout.grad += np.sum(pc_grads,axis=2)*(-1)
                    self.QMout.grad_pc = np.sum(pc_grads,axis=1)#*(-1)
                    self.log.debug(key)
                case "dm":
                    self.QMout.dm = np.array(NN_out["dm"])
                case "nacdr":
                    self.QMout.nacdr = np.array(NN_out["nacdr"])
                    self.log.debug(key)
                case "socdr":
                    self.QMout.socdr = np.array(NN_out["socdr"])
                case _:
                    self.log.warning(key," is not implemented")

    def getQMout(self):
        # everything is already
        self.QMout.states = self.QMin.molecule['states']
        self.QMout.nstates = self.QMin.molecule['nstates']
        self.QMout.nmstates = self.QMin.molecule['nmstates']
        self.QMout.natom = self.QMin.molecule['natom']
        self.QMout.npc = self.QMin.molecule['npc']
        self.QMout.point_charges = False
        return self.QMout


if __name__ == "__main__":
    SHARC_MACE().main()
