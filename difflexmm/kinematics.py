"""
The `kinematics` module implements the block-to-node rigid body kinematics.
"""

from typing import Callable, Dict

import jax.numpy as jnp
from jax import vmap

from difflexmm.geometry import DOFsInfo, Geometry, rotation_matrix


def _block_to_node_displacement(block_displacement: jnp.ndarray, centroid_node_vectors: jnp.ndarray):
    """Computes displacement of node belonging to a block that is displaced by `block_displacement` where centroid_node_vectors is the vector connecting the centroid of the block to the node.

    Args:
        block_displacement (ndarray): array of shape (3,) representing the in-plane displacement of the block's centroid (first two positions) and the block's rotation (last position).
        centroid_node_vectors (ndarray): array of shape (2,) representing the vectors connecting the centroid of the block to the node.

    Returns:
        ndarray: array of shape (3,) representing the displacement of the node and the rotation of the block.
    """

    block_centroid_displacement = block_displacement[:2]
    block_rotation = block_displacement[2]

    node_displacement = block_centroid_displacement + \
        jnp.dot(rotation_matrix(block_rotation) -
                jnp.eye(2), centroid_node_vectors)

    return jnp.concatenate([node_displacement, jnp.array([block_rotation]).flatten()])


# Vectorize over array of nodes per block first and then over array of blocks
block_to_node_kinematics = vmap(
    vmap(_block_to_node_displacement, in_axes=(None, 0)), in_axes=(0, 0))


def build_constrained_kinematics(geometry: Geometry, constrained_block_DOF_pairs: jnp.ndarray, constrained_DOFs_fn: Callable = lambda t, **kwargs: 0):
    """Defines a constrained kinematics of the blocks.

    Args:
        geometry (Geometry): Geometry of the structure.
        constrained_block_DOF_pairs (jnp.ndarray): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id].
        constrained_DOFs_fn (Callable, optional): Constraint function defining how the DOFs are driven over time. Output shape should either be scalar or match (len(constrained_block_DOF_pairs),). Valid signature: `constrained_DOFs_fn(t, **kwargs) -> ndarray`. Defaults to lambda t: 0.

    Returns:
        Callable: Constraint function mapping the free DOFs and time to the displacement of all the blocks. The signature is `constrained_kinematics(free_DOFs, t, constraint_params)`.
    """

    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(
        geometry.n_blocks, constrained_block_DOF_pairs)

    def constrained_kinematics(free_DOFs: jnp.ndarray, t, constraint_params: Dict = dict()):
        """Constrained kinematics of the blocks.

        Args:
            free_DOFs (jnp.ndarray): Array of shape (n_free_DOFs,) representing the free DOFs.
            t (float): Time parameter for time-dependent constraints.
            constraint_params (Dict, optional): Dictionary of kwargs to be passed to the `constrained_DOFs_fn`. Defaults to dict().

        Returns:
            jnp.ndarray: Array of shape (n_blocks, 3) representing the DOFs of all the blocks.
        """

        all_DOFs = jnp.zeros((len(all_DOF_ids),))
        # Assign imposed displacements along the constrained DOFs
        if len(constrained_DOF_ids) != 0:
            all_DOFs = all_DOFs.at[constrained_DOF_ids].set(
                constrained_DOFs_fn(t, **constraint_params)
            )
        # Simply assign the free_DOFs along the free DOFs (this acts as the identity operator)
        all_DOFs = all_DOFs.at[free_DOF_ids].set(
            free_DOFs
        )

        return all_DOFs.reshape((geometry.n_blocks, 3))

    return constrained_kinematics


def block_to_dipole_configuration(block_displacements: jnp.ndarray, block_centroids: jnp.ndarray, dipole_angles: jnp.ndarray, magnetic_block_ids: jnp.ndarray):
    """Computes configuration of the dipoles from the displacements of the blocks.

    Args:
        block_displacements (ndarray): array of shape (n_blocks, 3) representing the in-plane displacement of the block's centroid (first two positions) and the block's rotation (last position).
        block_centroids (ndarray): array of shape (n_blocks, 2) representing the centroid of the blocks in the reference configuration.
        dipole_angles (ndarray): array of shape (n_dipoles, 2) representing the reference in-plane and pitch angle of the dipoles.
        magnetic_block_ids (ndarray): array of shape (n_dipoles,) representing the block ids of the blocks holding magnets.

    Returns:
        ndarray: array of shape (n_dipoles, 4) representing the current configuration of the dipoles.
    """

    dipole_locations = block_centroids[magnetic_block_ids] + \
        block_displacements[magnetic_block_ids, :2]
    current_dipole_inplane_angles = block_displacements[magnetic_block_ids, 2] + \
        dipole_angles[:, 0]  # in-plane angle of the dipole
    # out-of-plane angle of the dipole (pitch)
    current_dipole_outofplane_angles = dipole_angles[:, 1]
    return jnp.column_stack((dipole_locations, current_dipole_inplane_angles, current_dipole_outofplane_angles))
