from typing import Sequence, Tuple
import jax.numpy as jnp


ANGSTROM_TO_BOHR = 1.8897259886


ELEMENT_CHARGES = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
}


def parse_molecule(
    molecule: Sequence[Tuple[str, Tuple[float, float, float]]],
    units: str = "bohr",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    atoms = []
    charges = []

    scale = ANGSTROM_TO_BOHR if units == "angstrom" else 1.0

    for element, coords in molecule:
        charges.append(ELEMENT_CHARGES[element])
        atoms.append([c * scale for c in coords])

    return jnp.array(atoms), jnp.array(charges)


def get_spin_config(
    charges: jnp.ndarray, spin_polarized: bool = False
) -> Tuple[int, int]:
    total_electrons = int(jnp.sum(charges))

    if spin_polarized:
        n_up = total_electrons
        n_down = 0
    else:
        n_up = (total_electrons + 1) // 2
        n_down = total_electrons // 2

    return (n_up, n_down)
