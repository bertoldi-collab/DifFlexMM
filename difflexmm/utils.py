import pickle
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Union

import jax.numpy as jnp
import numpy as np


class SolutionData(NamedTuple):
    """Solution data.
    This data structure is useful to lump together the solution fields and the geometrical parameters of the system.

    Attrs:
        block_centroids (jnp.ndarray): Array of shape (n_blocks, 2) representing Reference centroids of the blocks.
        centroid_node_vectors (jnp.ndarray): Array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes (effectively defining the shape of the blocks).
        bond_connectivity (jnp.ndarray): Array of shape (n_bonds, 2) representing the connectivity of the bonds.
        timepoints (jnp.ndarray): Array of shape (n_timepoints,) representing the timepoints at which the solution was saved.
        fields (jnp.ndarray): Array of shape (n_timepoints, 2, n_blocks, 3) representing the solution fields (displacements and rotations) at each timepoint. The axis are (timepoint, displacement/velocity, block, x/y/theta).
    """

    block_centroids: Any
    centroid_node_vectors: Any
    bond_connectivity: Any
    timepoints: Any
    fields: Any


class EigenmodeData(NamedTuple):
    """Eigenmode data.
    This data structure is useful to lump together the eigenvalues/eigenmodes and the geometrical parameters of the system.

    Attrs:
        block_centroids (jnp.ndarray): Array of shape (n_blocks, 2) representing Reference centroids of the blocks.
        centroid_node_vectors (jnp.ndarray): Array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes (effectively defining the shape of the blocks).
        eigenvalues (jnp.ndarray): Array of shape (n_modes,) representing the eigenvalues.
        fields (jnp.ndarray): Array of shape (n_modes, 2, n_blocks, 3) representing the eigenmodes (displacements and rotations). The axis are (mode, displacement, block, x/y/theta).
    """

    block_centroids: Any
    centroid_node_vectors: Any
    eigenvalues: Any
    fields: Any


SolutionType = Union[SolutionData, EigenmodeData]


class GeometricalParams(NamedTuple):
    """Geometrical parameters of the system.

    These parameters are typically computed from a given geometry class defined in `difflexmm.geometry`.

    Attrs:
        block_centroids (jnp.ndarray): Centroid coordinates of the blocks.
        centroid_node_vectors (jnp.ndarray): Array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes (effectively defining the shape of the blocks).
    """

    block_centroids: Any
    centroid_node_vectors: Any


class LigamentParams(NamedTuple):
    """Parameters for the bonds modeled as finite-length ligaments.

    These are meant to be used with the ligament energy functions defined in `difflexmm.energy`.

    Attrs:
        k_stretch (jnp.ndarray): Either a scalar or an array of shape (n_bonds,) representing the stretch stiffness of each bond.
        k_shear (jnp.ndarray): Either a scalar or an array of shape (n_bonds,) representing the shear stiffness of each bond.
        k_rot (jnp.ndarray): Either a scalar or an array of shape (n_bonds,) representing the rotational stiffness of each bond.
        reference_bond_vectors (jnp.ndarray): Array of shape (n_bonds, 2) representing the reference configuration of the bond (length matters). These are typically computed from a given geometry class.
    """

    k_stretch: Any
    k_shear: Any
    k_rot: Any
    reference_vector: Any


class StretchingTorsionalSpringParams(NamedTuple):
    """Parameters for the bonds modeled as zero-length springs accounting for stretching and bending.

    These are meant to be used with the `stretching_torsional_spring_energy` functions defined in `difflexmm.energy`.

    Attrs:
        k_stretch (jnp.ndarray): Either a scalar or an array of shape (n_bonds,) representing the stretch stiffness of each bond.
        k_rot (jnp.ndarray): Either a scalar or an array of shape (n_bonds,) representing the rotational stiffness of each bond.
    """

    k_stretch: Any
    k_rot: Any


BondParams = Union[LigamentParams, StretchingTorsionalSpringParams]


class ContactParams(NamedTuple):
    """Contact parameters for the simplified contact model.

    See `energy.contact_energy` for details.
    Note: If distance-based contact is used the min_angle and cutoff_angle are interpreted as distances.

    Attrs:
        min_angle (jnp.ndarray, optional): Lower bound for the angle between the blocks.
        cutoff_angle (jnp.ndarray, optional): Cutoff for the contact energy.
        k_contact (float, optional): Initial stiffness of the contact.
    """

    min_angle: Any
    cutoff_angle: Any
    k_contact: Any


class MagneticParams(NamedTuple):
    """Magnetic parameters of the system.

    These are meant to be used with the magnetic energy functions defined in `difflexmm.energy`.

    Attrs:
        dipole_angles (jnp.ndarray): Array of shape (n_dipoles, 2) representing the initial (reference) angles (in_plane_angle, pitch) of each dipole.
        dipole_strengths (jnp.ndarray): Either a scalar or an array of shape (n_dipoles,) representing the magnitude of the magnetic moment of each dipole.
    """

    dipole_angles: Any
    dipole_strengths: Any


class MechanicalParams(NamedTuple):
    """Mechanical parameters of the system.

    Attrs:
        bond_params (BondParams): NamedTuple defining the bond parameters.
        density (jnp.ndarray): Density of the blocks, either a scalar or an array of shape (n_blocks,).
        inertia (jnp.ndarray, optional): Array of shape (n_blocks, 3) defining the inertia of the blocks. If None, the inertia is computed from the geometry and density. Defaults to None.
        contact_params (ContactParams, optional): NamedTuple defining the contact parameters. Defaults to None.
    """

    bond_params: BondParams
    density: Any
    inertia: Optional[Any] = None
    damping: Any = 0.
    contact_params: Optional[ContactParams] = None


class ControlParams(NamedTuple):
    """Control parameters for the dynamic solver.
    The control parameters are used to define the geometry, the mechanical properties, loading parameters, etc.
    This data structure is meant to help with the construction of the mapping: design variables -> geometry, mechanical properties, etc. -> dynamic solver.

    Attrs:
        geometrical_params (GeometricalParams): NamedTuple defining the geometrical parameters.
        mechanical_params (MechanicalParams): NamedTuple defining the mechanical parameters.
        magnetic_params (MagneticParams): NamedTuple defining the magnetic parameters.
        loading_params (Dict[str, Any]): Loading parameters to be passed to loading functions. Default: {}.
        constraint_params (Dict[str, Any]): Constraint parameters to be passed to constraint_DOFs_fn. Default: {}.
    """

    geometrical_params: GeometricalParams  # centroids and centroid_node_vectors
    mechanical_params: MechanicalParams  # bond params, mass density, damping
    # dipole angles, dipole moments
    magnetic_params: Optional[MagneticParams] = None
    loading_params: Dict = dict()
    constraint_params: Dict = dict()


def save_data(path_or_filename: Union[str, Path], data: object):
    """Saves data to a file via `pickle`.

    Args:
        path_or_filename (Union[str, Path]): Path or filename of the outputfile e.g. output.dat.
        data (object): Object to be saved.
    """

    path = Path(path_or_filename)
    # Make sure parents directories exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(data, file)
        print("Data saved at " + str(path))


def load_data(path_or_filename: Union[str, Path]):
    """Loads data object via `pickle`.

    Args:
        path_or_filename (Union[str, Path]): Path or filename of the data file e.g. output.dat.

    Returns:
        object: The data object.
    """

    with open(path_or_filename, "rb") as file:
        data = pickle.load(file)

        if isinstance(data, (SolutionData, EigenmodeData)):
            # Cast arrays to jax arrays
            class_type = type(data)
            return class_type(*(jnp.array(attr) if isinstance(attr, np.ndarray) else attr for attr in data))

        return data


def is_scalar(x):
    """
    Check if x is a scalar. Note: if x has no attribute shape is assumed to a scalar.
    """
    # NOTE: this is needed because jnp.isscalar sucks.

    if jnp.array(x).shape == ():
        return True
    else:
        return False
