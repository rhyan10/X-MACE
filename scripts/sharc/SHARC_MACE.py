#!/usr/bin/env python3
# ******************************************
#
#    SHARC 4 interface for excited-state MACE models (X-MACE)
#
#    Install: copy or link this file into $SHARC/bin and put MACE.template
#    (and optionally MACE.resources with "modelpath <file>") into the QM directory.
#    The MACE package has to be importable from the Python environment SHARC uses.
#
#    This interface is based on the SHARC_FAST template of the SHARC Program Suite
#    (University of Vienna, GPL v3).
#
# ******************************************

import datetime
import os
import shutil
from io import TextIOWrapper

import numpy as np
from SHARC_FAST import SHARC_FAST  # pylint: disable=import-error
from utils import expand_path, link, question  # pylint: disable=import-error

from mace.calculators.sharc_calculator import SharcCalculator

authors = "Rhyan Barrett"
version = "4.0"
versiondate = datetime.datetime(2025, 11, 1)
description = "MACE excited-state machine-learning potentials (X-MACE)"
name = "MACE"
changelogstring = ""


class SHARC_MACE(SHARC_FAST):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.interface_name = "MACE"
        QMin = self.QMin
        QMin.template.types = {
            "model_file": str,
            "properties": list,
            "device": str,
            "energy_unit": str,
            "distance_unit": str,
        }
        QMin.template.data = {
            "model_file": None,
            "properties": ["energy", "forces"],
            "device": "cpu",
            "energy_unit": "eV",
            "distance_unit": "Ang",
        }
        if hasattr(QMin.resources, "types"):
            QMin.resources.types["modelpath"] = str
            QMin.resources.data["modelpath"] = None
        self.calculator = None

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
        return {"h", "soc", "dm", "grad", "nacdr", "phases"}

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"||{'MACE interface setup':^76}||")
        self.log.info("=" * 80)
        if os.path.isfile("MACE.template"):
            self.log.info("Found MACE.template in current directory")
            if question(
                "Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True
            ):
                self._template_file = "MACE.template"
        else:
            self.log.info("Specify a path to a MACE template file.")
            while not os.path.isfile(
                template_file := question("Template path:", str, KEYSTROKES=KEYSTROKES)
            ):
                self.log.info(f"File {template_file} does not exist!")
            self._template_file = template_file
        if question(
            "Do you have a MACE.resources file?",
            bool,
            KEYSTROKES=KEYSTROKES,
            autocomplete=False,
            default=False,
        ):
            while not os.path.isfile(
                resources_file := question(
                    "Specify path to MACE.resources",
                    str,
                    KEYSTROKES=KEYSTROKES,
                    autocomplete=True,
                )
            ):
                self.log.info(f"File {resources_file} does not exist!")
            self._resources_file = resources_file
        else:
            self.setupINFOS["modelpath"] = question(
                "Specify path to MACE model: ", str, KEYSTROKES=KEYSTROKES
            )
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self._resources_file:
            with open(
                os.path.join(dir_path, "MACE.resources"), "w", encoding="utf-8"
            ) as file:
                if "modelpath" in self.setupINFOS:
                    file.write(f"modelpath {self.setupINFOS['modelpath']}\n")
                else:
                    self.log.error("Modelpath not specified!")
                    raise ValueError("Modelpath not specified")
        else:
            create_file(
                expand_path(self._resources_file),
                os.path.join(dir_path, "MACE.resources"),
            )
        create_file(
            expand_path(self._template_file), os.path.join(dir_path, "MACE.template")
        )

    def read_resources(self, resources_filename="MACE.resources"):
        if os.path.isfile(resources_filename):
            super().read_resources(resources_filename)

    def read_template(self, template_filename="MACE.template"):
        super().read_template(template_filename)

    def setup_interface(self):
        super().setup_interface()
        states = list(self.QMin.molecule["states"])
        if len(states) > 1 and states[1] > 0:
            raise ValueError(
                "SHARC_MACE supports singlets and triplets only (no doublets)"
            )
        if len(states) > 3 and any(s > 0 for s in states[3:]):
            raise ValueError("SHARC_MACE supports singlets and triplets only")
        model_file = self.QMin.template["model_file"]
        if model_file is None:
            try:
                model_file = self.QMin.resources["modelpath"]
            except (KeyError, TypeError):
                model_file = None
        if model_file is None:
            raise ValueError(
                "Give model_file in MACE.template or modelpath in MACE.resources"
            )
        self.calculator = SharcCalculator(
            atom_types=self.QMin.molecule["elements"],
            model_paths=model_file,
            device=self.QMin.template["device"],
            energy_unit=self.QMin.template["energy_unit"],
            distance_unit=self.QMin.template["distance_unit"],
            n_states={
                "n_singlets": states[0],
                "n_triplets": states[2] if len(states) > 2 else 0,
            },
            properties=self.QMin.template["properties"],
        )

    def create_restart_files(self):
        pass

    def _requested(self, key: str) -> bool:
        try:
            return bool(self.QMin.requests[key])
        except (KeyError, TypeError):
            return False

    def run(self):
        qm = self.calculator.calculate(self.QMin.coords["coords"])
        h = np.array(qm["h"])
        if not self._requested("soc"):
            h = np.diag(np.diag(h))
        self.QMout.h = h
        if self._requested("grad"):
            self.QMout.grad = np.array(qm["grad"])
        if self._requested("nacdr"):
            if "nacdr" not in qm:
                raise ValueError(
                    "nacdr requested: add nacs (or smooth_nacs) to the properties"
                )
            self.QMout.nacdr = np.array(qm["nacdr"])
        if self._requested("dm"):
            if "dm" in qm:
                self.QMout.dm = np.array(qm["dm"])
            else:
                self.log.warning(
                    "dm requested but the model has no dipoles: zeros returned"
                )
                nm = self.QMin.molecule["nmstates"]
                self.QMout.dm = np.zeros((3, nm, nm))
        if self._requested("phases"):
            # couplings are kept phase-continuous by SharcCalculator
            self.QMout.phases = np.ones(self.QMin.molecule["nmstates"], dtype=complex)

    def getQMout(self):
        self.QMout.states = self.QMin.molecule["states"]
        self.QMout.nstates = self.QMin.molecule["nstates"]
        self.QMout.nmstates = self.QMin.molecule["nmstates"]
        self.QMout.natom = self.QMin.molecule["natom"]
        self.QMout.npc = self.QMin.molecule["npc"]
        self.QMout.point_charges = False
        return self.QMout


if __name__ == "__main__":
    SHARC_MACE().main()
