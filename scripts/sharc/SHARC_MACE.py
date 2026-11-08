#!/usr/bin/env python3
# ******************************************
#
#    SHARC 4 interface for excited-state MACE models (X-MACE)
#
#    Install: copy or link this file into $SHARC/bin (SHARC >= 4.0) and put
#    MACE.template (and optionally MACE.resources with "modelpath <file>") into the
#    QM directory. The MACE package has to be importable from the Python environment
#    SHARC uses. Works with sharc.x (QM/runQM.sh calling SHARC_MACE.py QM.in) and with
#    pysharc ($SHARC/driver.py -i MACE input).
#
#    Any multiplicities SHARC supports (singlets, doublets, triplets, quartets,
#    quintets, ...) can be used when the model was trained with --states; SOCs come
#    from models trained with --soc_mode=spin_orbit (Wigner-Eckart) or, for singlets and
#    triplets, from the X-MACE upper-triangle layout.
#
#    This interface follows the SHARC_FAST template of the SHARC Program Suite
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

from mace.calculators.sharc_calculator import HARTREE_TO_EV, SharcCalculator

AUTHORS = "Rhyan Barrett"
VERSION = "4.1"
VERSIONDATE = datetime.datetime(2026, 11, 8)
NAME = "MACE"
DESCRIPTION = "     FAST interface for excited-state MACE models (X-MACE)"
CHANGELOGSTRING = """
4.1  any multiplicities (doublets, quartets, quintets, ...), Wigner-Eckart SOCs,
     SHARC 4.1 API, phase continuity across non-persistent calls
"""

HARTREE_TO_INV_CM = 219474.6313632
SOC_UNITS_TO_HARTREE = {
    "Hartree": 1.0,
    "eV": 1.0 / HARTREE_TO_EV,
    "cm-1": 1.0 / HARTREE_TO_INV_CM,
}

ALL_FEATURES = {"h", "soc", "dm", "grad", "nacdr", "phases"}

TEMPLATE_DEFAULTS = {
    "model_file": None,
    "properties": ["energy", "forces"],
    "device": "cpu",
    "energy_unit": "eV",
    "distance_unit": "Ang",
    "soc_unit": None,  # default: same unit as the energies
}
TEMPLATE_TYPES = {
    "model_file": str,
    "properties": list,
    "device": str,
    "energy_unit": str,
    "distance_unit": str,
    "soc_unit": str,
}


class SHARC_MACE(SHARC_FAST):
    """SHARC interface for ExcitedMACE / AutoencoderExcitedMACE models."""

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.QMin.resources.update({"modelpath": None})
        self.QMin.resources.types.update({"modelpath": str})
        self.QMin.template.update(dict(TEMPLATE_DEFAULTS))
        self.QMin.template.types.update(dict(TEMPLATE_TYPES))
        self.calculator = None
        self._qm = None
        self._resources_file = None
        self._template_file = None
        self._soc_warned = False

    @staticmethod
    def version() -> str:
        return SHARC_MACE._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_MACE._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_MACE._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_MACE._authors

    @staticmethod
    def name() -> str:
        return SHARC_MACE._name

    @staticmethod
    def description() -> str:
        return SHARC_MACE._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_MACE._name}\n{SHARC_MACE._description}"

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        return set(ALL_FEATURES)

    # ----- setup (setup_traj.py) ------------------------------------------------------
    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"||{'MACE interface setup':^76}||")
        self.log.info("=" * 80)
        if os.path.isfile("MACE.template") and question(
            "Use the MACE.template in the current directory?",
            bool,
            KEYSTROKES=KEYSTROKES,
            default=True,
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
                "Specify path to MACE model (leave empty to use model_file of the template): ",
                str,
                KEYSTROKES=KEYSTROKES,
                default="",
            )
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        create_file = link if INFOS.get("link_files") else shutil.copy
        if self._resources_file:
            create_file(
                expand_path(self._resources_file),
                os.path.join(dir_path, "MACE.resources"),
            )
        elif self.setupINFOS.get("modelpath"):
            with open(
                os.path.join(dir_path, "MACE.resources"), "w", encoding="utf-8"
            ) as file:
                file.write(f"modelpath {self.setupINFOS['modelpath']}\n")
        create_file(
            expand_path(self._template_file), os.path.join(dir_path, "MACE.template")
        )

    # ----- initialisation ----------------------------------------------------------------
    def read_resources(self, resources_file="MACE.resources", kw_whitelist=None):
        if not os.path.isfile(resources_file):
            self.log.debug(f"{resources_file} not found, using the template only")
            self._read_resources = True
            return
        super().read_resources(resources_file, kw_whitelist)

    def read_template(self, template_file="MACE.template", kw_whitelist=None):
        super().read_template(template_file, kw_whitelist)

    def _model_file(self) -> str:
        model_file = self.QMin.template["model_file"] or self.QMin.resources["modelpath"]
        if not model_file:
            raise ValueError(
                "Give model_file in MACE.template or modelpath in MACE.resources"
            )
        return expand_path(model_file)

    def setup_interface(self):
        super().setup_interface()
        template = self.QMin.template
        soc_unit = template["soc_unit"]
        soc_to_au = None
        if soc_unit is not None:
            if soc_unit not in SOC_UNITS_TO_HARTREE:
                raise ValueError(
                    f"soc_unit must be one of {list(SOC_UNITS_TO_HARTREE)}, got {soc_unit}"
                )
            soc_to_au = SOC_UNITS_TO_HARTREE[soc_unit]
        self.calculator = SharcCalculator(
            atom_types=self.QMin.molecule["elements"],
            model_paths=self._model_file(),
            device=template["device"],
            energy_unit=template["energy_unit"],
            distance_unit=template["distance_unit"],
            states=list(self.QMin.molecule["states"]),
            properties=template["properties"],
            soc_to_au=soc_to_au,
        )
        self.log.info(
            f"MACE model {self._model_file()} ({self.calculator.mode}), SHARC states "
            f"{list(self.QMin.molecule['states'])}, properties {self.calculator.properties}"
        )

    # ----- phase continuity between separate calls (sharc.x) ----------------------------
    def _phase_file(self, step: int) -> str:
        return os.path.join(self.QMin.save["savedir"], f"mace_phases.{step}")

    def _load_phases(self):
        if self.persistent or self.calculator._previous:
            return
        step = self.QMin.save["step"]
        for previous in (step - 1, step):
            path = self._phase_file(previous)
            if previous >= 0 and os.path.isfile(path):
                with open(path, "rb") as file:
                    stored = np.load(file)
                    self.calculator._previous = {k: stored[k] for k in stored.files}
                return

    def _save_phases(self):
        if self.persistent or not self.calculator._previous:
            return
        with open(self._phase_file(self.QMin.save["step"]), "wb") as file:
            np.savez(file, **self.calculator._previous)

    # ----- runtime -----------------------------------------------------------------------
    def _requested(self, key: str) -> bool:
        try:
            return bool(self.QMin.requests[key])
        except (KeyError, TypeError):
            return False

    def run(self):
        self._load_phases()
        self._qm = self.calculator.calculate(self.QMin.coords["coords"])
        self._save_phases()
        self.QMout["runtime"] = self.clock.measuretime(False)

    def getQMout(self):
        requests = {
            key for key, value in self.QMin.requests.items() if value and key != "retain"
        }
        self.QMout.allocate(
            states=self.QMin.molecule["states"],
            natom=self.QMin.molecule["natom"],
            npc=self.QMin.molecule["npc"],
            requests=requests,
        )
        qm = self._qm
        nmstates = self.QMin.molecule["nmstates"]
        if self._requested("h") or self._requested("soc"):
            h = np.array(qm["h"], dtype=complex)
            if not self._requested("soc"):
                h = np.diag(np.diag(h))
            elif self.calculator.soc_source is None and not self._soc_warned:
                self._soc_warned = True
                self.log.warning(
                    "SOCs requested, but the model gives none for these states (add socs "
                    "to the properties of MACE.template?): spin-orbit couplings are zero"
                )
            self.QMout["h"] = h
        if self._requested("grad"):
            self.QMout["grad"] = np.array(qm["grad"])
        if self._requested("nacdr"):
            if "nacdr" not in qm:
                raise ValueError(
                    "nacdr requested: add nacs (or smooth_nacs) to the properties in "
                    "MACE.template"
                )
            self.QMout["nacdr"] = np.array(qm["nacdr"])
        if self._requested("dm"):
            if "dm" in qm:
                self.QMout["dm"] = np.array(qm["dm"])
            else:
                self.log.debug("dm requested but no dipoles predicted: zeros returned")
        if self._requested("phases"):
            # couplings are kept phase-continuous by SharcCalculator
            self.QMout["phases"] = np.ones(nmstates, dtype=complex)
        self.QMout["runtime"] = self.clock.measuretime(False)
        return self.QMout

    def create_restart_files(self):
        super().create_restart_files()


if __name__ == "__main__":
    SHARC_MACE().main()
