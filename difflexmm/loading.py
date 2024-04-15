"""
The `loading` module implements loading and boundary conditions data structures.
"""

from typing import Callable, Dict

import jax.numpy as jnp

from difflexmm.geometry import DOFsInfo, Geometry


def build_loading(
        geometry: Geometry,
        loaded_block_DOF_pairs: jnp.ndarray,
        loading_fn: Callable,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([])):
    """Defines the loading function.

    Args:
        geometry (Geometry): geometry.
        loaded_block_DOF_pairs (jnp.ndarray): array of shape (Any, 2) where each row defines a pair of [block_id, DOF_id] where DOF_id is either 0, 1, or 2
        loading_fn (Callable): Loading function. Output shape should either be scalar or match (len(loaded_block_DOF_pairs),).
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        Callable: vector loading function evaluating to `loading_fn` for the DOFs defined by `loaded_block_DOF_pairs` and 0 otherwise.
    """

    # loaded DOF ids based on global numeration
    loaded_DOF_ids = jnp.array(
        [block_id * 3 + DOF_id for block_id, DOF_id in loaded_block_DOF_pairs])
    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, _, all_DOF_ids = DOFsInfo(
        geometry.n_blocks, constrained_block_DOF_pairs)

    def global_loading_fn(state, t, loading_params: Dict):

        loading_vector = jnp.zeros((len(all_DOF_ids),))
        loading_vector = loading_vector.at[loaded_DOF_ids].set(
            loading_fn(state, t, **loading_params)
        )
        # Reduce loading vector to the free DOFs
        loading_vector = loading_vector[free_DOF_ids]

        return loading_vector

    return global_loading_fn


def build_node_loading(
        geometry: Geometry,
        loaded_block_node_DOF_triples: jnp.ndarray,
        loading_fn: Callable,
        centroid_node_vectors: jnp.ndarray,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([])):
    """
    docstring
    """

    # TODO: Implement nodal loading function in one of the following ways:
    #   - Compute virtual power and let jax take the gradient with respect to virtual velocity.
    #   - Find the appropriate way to vectorize something like (A_n)^T . F_n where A_n is the gradient of n node displacement with respect to block DOFs and F_n the nodal loading.
    # In both cases, be sure to constrained the resulting loading vector to the freeDOFs using constraints info.

    # node_displacements = block_to_node_kinematics(
    #     block_displacement,
    #     centroid_node_vectors
    # )


def build_viscous_damping(
        geometry: Geometry,
        damped_blocks: jnp.ndarray,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([])):
    """Defines viscous damping forces.

    Args:
        geometry (Geometry): geometry.
        damped_blocks (jnp.ndarray): array of shape (n_damped_blocks,) collecting the block ids of the damped blocks.
        damping_values (jnp.ndarray): array of shape (n_damped_blocks, 3) collecting the damping values for each block and DOF.
        constrained_block_DOF_pairs (jnp.ndarray): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        Callable: function evaluating the viscous damping forces.
    """

    damped_DOF_ids = jnp.concatenate(
        [jnp.arange(block_id * 3, (block_id + 1) * 3) for block_id in damped_blocks])
    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, _, all_DOF_ids = DOFsInfo(
        geometry.n_blocks, constrained_block_DOF_pairs)

    # This is to ensure correct shape of loading vector when damping is either a scalar or an array of shape (n_damped_blocks, 3)
    reshaping_array = jnp.ones((len(damped_blocks), 3))

    def loading_fn(state, t, damping: jnp.ndarray):
        _, velocity = state
        loading_vector = jnp.zeros((len(all_DOF_ids),))
        loading_vector = loading_vector.at[damped_DOF_ids].set(
            (damping * reshaping_array).reshape(damped_DOF_ids.shape)
        )
        loading_vector = loading_vector[free_DOF_ids]

        return -loading_vector * velocity

    return loading_fn
