"""The spellings of the excited-state (X-MACE) outputs, on all surfaces.

``ExcitedMACE`` / ``AutoencoderExcitedMACE`` (mace/modules/excited.py) and
``ExcitedMACECalculator`` (mace/calculators/excited.py) predict several
electronic states at once, so most of their outputs have a state or state-pair
axis whose extent is set by the model's heads:

    model forward   energy [B, S], forces [N, S, 3], nacs [N, P, 3],
                    socs [B, L], soc_vectors [B, Q, 3], dipoles [B, D, 3],
                    ground_energy [B] (the backbone's energy),
                    state_charges [N, S], state_electrostatic_energy [B, S],
                    invariant_vals / encoded_energy [B, K] and
                    decoded_energy [B, S] (autoencoder)
    calculator      energies [S], forces_states [N, S, 3], nacs and
                    smooth_nacs [N, P, 3], socs [L], soc_vectors [Q, 3],
                    soc_matrix [nm, nm] (complex), dipoles [D, 3], and the
                    committee spreads energies_var, forces_states_var

The extents are graph arrays / per-atom matrices, so the kinds below leave them
free. Two spellings collide with the ground-state vocabulary and are *not*
re-registered here: the excited calculator's ``energies`` is the per-state
energy vector (the ground-state calculator's is per atom), and the excited
models' ``energy`` / ``forces`` carry a state axis. A golden of an excited-state
model must therefore compare those through its own channel names (as the X-MACE
tests in tests/excited do), not through the ground-state channels.
"""

from __future__ import annotations

from tests.golden import harness

_GRAPH = harness.GRAPH_ARRAY
_ATOMS = harness.PER_ATOM_MATRIX

for _name, _kind, _unit in (
    ("forces_states", _ATOMS, "eV/Ang"),
    ("forces_states_var", _ATOMS, "eV^2/Ang^2"),
    ("energies_var", _GRAPH, "eV^2"),
    ("nacs", _ATOMS, "eV/Ang"),  # smooth NACs (NAC * dE) on the model surface
    ("smooth_nacs", _ATOMS, "eV/Ang"),
    ("socs", _GRAPH, "label unit"),
    ("soc_vectors", _GRAPH, "label unit"),
    ("soc_matrix", _GRAPH, "label unit"),
    ("dipoles", _GRAPH, "label unit"),
    ("ground_energy", harness.GRAPH_SCALAR, "eV"),
    ("state_charges", _ATOMS, "e"),
    ("state_electrostatic_energy", _GRAPH, "eV"),
    ("decoded_energy", _GRAPH, "eV"),
):
    harness.register_channel(_name, _kind, _unit)

for _name, _reason in (
    (
        "invariant_vals",
        "the autoencoder's latent vector: an internal coordinate of the decoder, "
        "defined only up to the decoder's parametrisation; the energies it "
        "decodes to are pinned instead",
    ),
    (
        "encoded_energy",
        "the encoder's image of the reference energies, a training target of the "
        "autoencoder rather than a prediction (absent without reference energies)",
    ),
):
    harness.ignore_key(_name, _reason)
