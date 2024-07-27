"""
The `energy` module implements the energy functional for the whole structure.
"""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import vmap
from jax_md import smap

from difflexmm.geometry import compute_edge_angles, rotation_matrix
from difflexmm.kinematics import block_to_node_kinematics
from difflexmm.utils import ControlParams


def vdot(v1, v2):
    """Vectorized dot product based on *.

    Args:
        v1 (jnp.ndarray): Array of shape (Any, Any).
        v2 (jnp.ndarray): Array having the same shape as v1 or (v1.shape[1],).

    Returns:
        jnp.ndarray: row-wise dot product between v1 and v2
    """

    return jnp.sum(v1 * v2, axis=-1)


def simple_spring_energy(nodal_DOFs: Tuple[jnp.ndarray, jnp.ndarray], reference_vector: jnp.ndarray = jnp.array([1., 0.]), k_stretch=1.):
    """Computes the energy of a simple linear spring connecting two nodes.

    Args:
        nodal_DOFs (Tuple[ndarray, ndarray]): tuple of arrays of shape (Any, 3) representing the DOFs of the nodes connected by the ligament.
        reference_vector (jnp.ndarray, optional): array of shape (2,) or (Any, 2) representing the reference configuration of the ligament. Defaults to jnp.array([1., 0.]).
        k_stretch (float, optional): linear stretching stiffness. Defaults to 1..

    Returns:
        float: strain energy.
    """

    DOFs1, DOFs2 = nodal_DOFs
    dU = DOFs2[:, :2] - DOFs1[:, :2]
    l = jnp.linalg.norm(dU + reference_vector, axis=-1)
    l0 = jnp.linalg.norm(reference_vector, axis=-1)
    axial_strain = l / l0 - 1

    return k_stretch * (axial_strain*l0)**2 / 2


def stretching_torsional_spring_energy(nodal_DOFs: Tuple[jnp.ndarray, jnp.ndarray], k_stretch=1., k_rot=1.):
    """Computes the energy of a zero-length spring connecting two coincident nodes accounting for stretching and bending energies.

    Args:
        nodal_DOFs (Tuple[ndarray, ndarray]): tuple of arrays of shape (Any, 3) representing the DOFs of the nodes connected by the ligament.
        k_stretch (float, optional): linear stretching stiffness. Defaults to 1..
        k_rot (float, optional): linear rotational stiffness. Defaults to 1..

    Returns:
        float: strain energy.
    """

    DOFs1, DOFs2 = nodal_DOFs
    dU = DOFs2[:, :2] - DOFs1[:, :2]
    dRot = DOFs2[:, 2] - DOFs1[:, 2]

    return k_stretch * vdot(dU, dU) / 2 + k_rot * dRot**2 / 2


def ligament_strains_linearized(DOFs1: jnp.ndarray, DOFs2: jnp.ndarray, reference_vector: jnp.ndarray = jnp.array([1., 0.])):
    """Computes linearized strain measures of an elastic ligament i.e. axial, shear, and flexural strains.

    The axial strain is defined as dU.v0/v0^2.
    The shear strain is defined as (theta1+theta2)/2 - v0✕dU/v0^2.
    The rotational strain is defined as theta2-theta1.

    Note: These strains are based on the linearized beam theory.

    Args:
        DOFs1 (jnp.ndarray): array of shape (Any, 3) representing the DOFs of the first node connected by the ligament.
        DOFs2 (jnp.ndarray): array of shape (Any, 3) representing the DOFs of the second node connected by the ligament.
        reference_vector (jnp.ndarray, optional): array of shape (2,) or (Any, 2) representing the reference configuration of the ligament. Defaults to jnp.array([1., 0.]).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: axial, shear, and rotational strains.
    """

    dU = DOFs2[:, :2] - DOFs1[:, :2]
    dRot = DOFs2[:, 2] - DOFs1[:, 2]

    axial_strain = vdot(dU, reference_vector) / \
        jnp.linalg.norm(reference_vector, axis=-1)**2
    shear_strain = jnp.cross(reference_vector, dU, axis=-1) / jnp.linalg.norm(reference_vector, axis=-1)**2 \
        - (DOFs2[:, 2] + DOFs1[:, 2])/2

    return axial_strain, shear_strain, dRot


def ligament_energy_linearized(nodal_DOFs: Tuple[jnp.ndarray, jnp.ndarray], reference_vector: jnp.ndarray = jnp.array([1., 0.]), k_stretch=1., k_shear=1., k_rot=1.):
    """Computes the strain energy of an elastic ligament using linearized strain measures (suitable for moderate global rotations).

    Args:
        nodal_DOFs (Tuple[ndarray, ndarray]): tuple of arrays of shape (Any, 3) representing the DOFs of the nodes connected by the ligament.
        reference_vector (ndarray, optional): array of shape (2,) or (Any, 2) representing the reference bond geometry (length matters). Defaults to jnp.array([1., 0.]).
        k_stretch (float, optional): linear stretching stiffness. Defaults to 1..
        k_shear (float, optional): linear shearing stiffness. Defaults to 1..
        k_rot (float, optional): linear rotational stiffness. Defaults to 1..

    Returns:
        float: strain energy.
    """

    axial_strain, shear_strain, dRot = ligament_strains_linearized(
        *nodal_DOFs, reference_vector=reference_vector)
    l0 = jnp.linalg.norm(reference_vector, axis=-1)

    return k_stretch * (axial_strain*l0)**2 / 2 + k_shear * (shear_strain*l0)**2 / 2 + k_rot * dRot**2 / 2


def ligament_strains(DOFs1: jnp.ndarray, DOFs2: jnp.ndarray, reference_vector: jnp.ndarray = jnp.array([1., 0.])):
    """Computes the nonlinear strain measures of an elastic ligament i.e. axial, shear, and flexural strains.

    The axial strain is defined as (L-L0)/L0.
    The shear strain is defined as current_bond_angle-reference_bond_pushed_angle where reference_bond_pushed_angle is the reference rotated by (theta1+theta2)/2.
    Note: the shear strain is assumed to be between -pi and pi.
    The rotational strain is defined as theta2-theta1.

    Note: These strains are based on beam theory (e.g. see https://static-content.springer.com/esm/art%3A10.1038%2Fnphys4269/MediaObjects/41567_2018_BFnphys4269_MOESM1_ESM.pdf).

    Args:
        DOFs1 (jnp.ndarray): array of shape (Any, 3) representing the DOFs of the first node connected by the ligament.
        DOFs2 (jnp.ndarray): array of shape (Any, 3) representing the DOFs of the second node connected by the ligament.
        reference_vector (jnp.ndarray, optional): array of shape (2,) or (Any, 2) representing the reference configuration of the ligament. Defaults to jnp.array([1., 0.]).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: axial, shear, and rotational strain measures.
    """

    dU = DOFs2[:, :2] - DOFs1[:, :2]
    dRot = DOFs2[:, 2] - DOFs1[:, 2]
    mean_rot = (DOFs2[:, 2] + DOFs1[:, 2])/2
    current_bond_vector = dU + reference_vector
    current_bond_angle = jnp.arctan2(
        *(jnp.roll(current_bond_vector, 1, axis=-1)).T)
    reference_bond_pushed = vmap(lambda angle, reference_v: jnp.dot(
        rotation_matrix(angle), reference_v), in_axes=(0, 0))(mean_rot, jnp.ones((len(DOFs1), 2))*reference_vector)
    reference_bond_pushed_angle = jnp.arctan2(
        *(jnp.roll(reference_bond_pushed, 1, axis=-1)).T)

    axial_strain = (vdot(current_bond_vector, current_bond_vector) /
                    vdot(reference_vector, reference_vector))**0.5 - 1
    shear_strain = jnp.mod(
        current_bond_angle - reference_bond_pushed_angle + jnp.pi, 2*jnp.pi) - jnp.pi

    return axial_strain, shear_strain, dRot


def ligament_energy(nodal_DOFs: Tuple[jnp.ndarray, jnp.ndarray], reference_vector: jnp.ndarray = jnp.array([1., 0.]), k_stretch=1., k_shear=1., k_rot=1.):
    """Computes the strain energy of an elastic ligament using nonlinear strain measures (suitable for arbitrarily large rotations).

    Args:
        nodal_DOFs (Tuple[ndarray, ndarray]): tuple of arrays of shape (Any, 3) representing the DOFs of the nodes connected by the ligament.
        reference_vector (ndarray, optional): array of shape (2,) or (Any, 2) representing the reference configuration of the bond (length matters). Defaults to jnp.array([1., 0.]).
        k_stretch (float, optional): linear stretching stiffness. Defaults to 1..
        k_shear (float, optional): linear shearing stiffness. Defaults to 1..
        k_rot (float, optional): linear rotational stiffness. Defaults to 1..

    Returns:
        float: strain energy.
    """

    axial_strain, shear_strain, dRot = ligament_strains(
        *nodal_DOFs, reference_vector=reference_vector)
    l0 = jnp.linalg.norm(reference_vector, axis=-1)

    return k_stretch * (axial_strain*l0)**2 / 2 + k_shear * (shear_strain*l0)**2 / 2 + k_rot * dRot**2 / 2


def strain_energy_bond(bond_connectivity: jnp.ndarray, bond_energy_fn: Callable = ligament_energy_linearized):
    """Maps energy functional of a single bond to a set of bonds defined by `bond_connectivity`.

    Args:
        bond_connectivity (ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.
        bond_energy_fn (Callable): energy functional of a single bond. Defaults to `energy.ligament_energy_linearized`.

    Returns:
        Callable: strain energy vectorized over the set of bonds defined by `bond_connectivity`.
    """

    return smap.bond(
        bond_energy_fn,  # Single bond energy
        # This pattern is needed because smap.bond is not vmapping kwargs in the strain function (workaround: strain measures are computed inside bond energy).
        lambda Ua, Ub, **kwargs: (Ua, Ub),
        static_bonds=bond_connectivity,
        static_bond_types=None
        # It can take any additional parameters to be passed to the single bond energy function
    )


# Contact energy between adjacent edges
# NOTE: This is a simplified way to handle contact. The energy is just based on the angle between blocks connected by a bond.
# NOTE: This is also not based on general data structures for defining edges (see geometry.compute_edge_angles).

def void_angles(current_block_nodes: jnp.ndarray, bond_connectivity: jnp.ndarray):
    """Computes angles between blocks connected by the bonds.

    Args:
        current_block_nodes (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the current position of the blocks.
        bond_connectivity (jnp.ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.

    Returns:
        jnp.ndarray: array of shape (2*n_bonds,) defining the void angles.
    """

    angles = vmap(lambda bond: compute_edge_angles(
        current_block_nodes, bond))(bond_connectivity)
    void_angles = jnp.array(angles)[:2].ravel()

    return void_angles


def point_to_edge_distance(point: jnp.ndarray, edge: jnp.ndarray):
    """Computes the distance between a point and an edge.

    Args:
        point (jnp.ndarray): array of shape (2,) defining the point.
        edge (jnp.ndarray): array of shape (2, 2) defining the edge.

    Returns:
        jnp.ndarray: distance between the point and the edge.
    """

    x0 = edge[0]
    x1 = edge[1]
    t = jnp.dot(point-x0, x1-x0)/jnp.dot(x1-x0, x1-x0)
    x_distance_to_e = jnp.where(
        (t >= 0) & (t <= 1),
        # Projected point is on the edge
        jnp.sum((point-x0)**2 - (t*(x1-x0))**2)**0.5,
        jnp.where(
            # Projected point is outside the edge
            t < 0,
            # Distance to first point
            jnp.sum((point-x0)**2)**0.5,
            # Distance to second point
            jnp.sum((point-x1)**2)**0.5
        )
    )
    return x_distance_to_e


# Contact model based edge-to-edge distances
def edges_distance(edge_1: jnp.ndarray, edge_2: jnp.ndarray):
    """Computes the distance between two edges.

    Args:
        edge_1 (jnp.ndarray): array of shape (2, 2) defining the first edge.
        edge_2 (jnp.ndarray): array of shape (2, 2) defining the second edge.

    Returns:
        jnp.ndarray: scalar distance between the two edges.
    """

    # Compute the distance projecting second edge on the first edge
    e2_onto_e1_distance = vmap(
        point_to_edge_distance, in_axes=(0, None))(edge_2, edge_1)
    # Compute the distance projecting first edge on the second edge
    e1_onto_e2_distance = vmap(
        point_to_edge_distance, in_axes=(0, None))(edge_1, edge_2)
    # Return the minimum distance
    distances = jnp.concatenate((e2_onto_e1_distance, e1_onto_e2_distance))

    return jnp.min(distances)


# Vectorized version of edges_distance (vectorized over arrays of edges)
edges_distance_mapped = vmap(edges_distance, in_axes=(0, 0))


def build_void_edge_distance(bond_connectivity: jnp.ndarray):
    """Builds a function that computes the distance between edges connected by the bonds.

    Args:
        bond_connectivity (jnp.ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.

    Returns:
        Callable: function that computes all the pairwise distances between edges connected by the bonds.
    """

    def void_edge_distance(current_block_nodes: jnp.ndarray):
        """Computes the distance between edges connected by the bonds.

        Args:
            current_block_nodes (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the current position of the blocks.

        Returns:
            jnp.ndarray: array of shape (2*n_bonds,) defining the distances between edges connected by the bonds.
        """

        _, n_nodes_per_block, _ = current_block_nodes.shape
        nodes_1_id = bond_connectivity[:, 0]
        nodes_2_id = bond_connectivity[:, 1]
        pts1 = current_block_nodes[nodes_1_id //
                                   n_nodes_per_block, nodes_1_id % n_nodes_per_block]
        pts1_prev = current_block_nodes[nodes_1_id //
                                        n_nodes_per_block, (nodes_1_id-1) % n_nodes_per_block]
        pts1_next = current_block_nodes[nodes_1_id //
                                        n_nodes_per_block, (nodes_1_id+1) % n_nodes_per_block]

        pts2 = current_block_nodes[nodes_2_id //
                                   n_nodes_per_block, nodes_2_id % n_nodes_per_block]
        pts2_prev = current_block_nodes[nodes_2_id //
                                        n_nodes_per_block, (nodes_2_id-1) % n_nodes_per_block]
        pts2_next = current_block_nodes[nodes_2_id //
                                        n_nodes_per_block, (nodes_2_id+1) % n_nodes_per_block]

        # Distance between edges on one side of the bond
        void_distances1 = edges_distance_mapped(
            jnp.concatenate((pts1[:, None], pts1_next[:, None]), axis=1),
            jnp.concatenate((pts2[:, None], pts2_prev[:, None]), axis=1)
        )
        # Distance between edges on the other side of the bond
        void_distances2 = edges_distance_mapped(
            jnp.concatenate((pts1[:, None], pts1_prev[:, None]), axis=1),
            jnp.concatenate((pts2[:, None], pts2_next[:, None]), axis=1)
        )

        return jnp.concatenate((void_distances1, void_distances2))

    return void_edge_distance


def contact_energy(current_void_angles: jnp.ndarray, min_angle: jnp.ndarray = jnp.array(0.), cutoff_angle: jnp.ndarray = jnp.array(2.0*jnp.pi/180), k_contact=1.0):
    """Computes the contact energy between connected blocks.

    This is a simplified way to handle contact. The energy is just based on the angle between blocks connected by a bond.

    Args:
        current_void_angles (jnp.ndarray): array of shape (2*n_bonds,) defining the angles between connected blocks.
        min_angle (jnp.ndarray, optional): lower bound for the angle between the blocks. Defaults to jnp.array(0.).
        cutoff_angle (jnp.ndarray, optional): cutoff for the contact energy. Defaults to jnp.array(2.0*jnp.pi/180).
        k_contact (float, optional): initial stiffness of the contact. Defaults to 1.0.

    Returns:
        float: contact energy
    """
    # Current contact energy is of the kind ~1/x with a C^1 cutoff.
    # min_angle is an asymptote for the energy. This is to make sure that min_angle cannot be overcome.
    x = (current_void_angles-cutoff_angle)/(cutoff_angle-min_angle)
    energy = jnp.where(
        # This means that the blocks are not in contact as we assume that min_angle is the minimum angle between the blocks
        current_void_angles < min_angle,
        0,
        jnp.where(
            current_void_angles < cutoff_angle,
            k_contact/4 * (cutoff_angle-min_angle)**2 * \
            ((x+1)**-1 - (x-1)**-1 - 2),
            0
        )
    )
    return energy


def build_contact_energy(bond_connectivity: jnp.ndarray, angle_based=True):
    """Defines the energy functional for simulating contact between connected blocks.

    Args:
        bond_connectivity (jnp.ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.
        angle_based (bool, optional): whether to use the angle-based contact energy or the distance-based one. Defaults to True (angle-based). Angle-based is more cheaper but less accurate for complex geometries.

    Returns:
        Callable: contact energy functional as a function of the DOFs of the blocks and the `control_params`.
    """

    void_edge_distance_fn = build_void_edge_distance(bond_connectivity)

    def void_angle_fn(current_block_nodes): return void_angles(
        current_block_nodes, bond_connectivity)
    distance_fn = void_angle_fn if angle_based else void_edge_distance_fn

    def contact_energy_fn(block_displacement: jnp.ndarray, control_params: ControlParams):
        """Computes the contact energy between connected blocks.

        Args:
            block_displacement (jnp.ndarray): array of shape (n_blocks, 3) collecting the displacements (first two positions) and rotations (last position) of all the blocks.
            centroid_node_vectors (ndarray): array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes.
            control_params (ControlParams): contains the contact params in control_params.mechanical_params.contact_params.

        Returns:
            float: Total contact energy.
        """

        block_centroids = control_params.geometrical_params.block_centroids
        centroid_node_vectors = control_params.geometrical_params.centroid_node_vectors
        contact_params = control_params.mechanical_params.contact_params

        node_displacements = jnp.array(
            block_to_node_kinematics(
                block_displacement,
                centroid_node_vectors
            )
        )[:, :, :2]
        current_block_nodes = block_centroids[:, None] + \
            centroid_node_vectors + node_displacements
        return jnp.sum(contact_energy(current_void_angles=distance_fn(current_block_nodes), **contact_params._asdict()))

    return contact_energy_fn


def build_strain_energy(bond_connectivity: jnp.ndarray, bond_energy_fn: Callable = ligament_energy_linearized):
    """Defines the strain energy functional of the system.

    Args:
        bond_connectivity (ndarray): array of shape (n_bonds, 2) where each row [n1, n2] defines a bond connecting nodes n1 and n2.
        bond_energy_fn (Callable): energy functional of a single bond. Defaults to `energy.ligament_energy_linearized`.

    Returns:
        Callable: function evaluating the strain energy of the system from the DOFs of the blocks and the `control_params`.
    """

    # Build vectorized bond energy using smap.bond
    strain_energy_bonds = strain_energy_bond(
        bond_connectivity=bond_connectivity, bond_energy_fn=bond_energy_fn)

    def strain_energy_fn(block_displacement: jnp.ndarray, control_params: ControlParams):
        """Computes total strain energy by summing over all bonds.

        Args:
            block_displacement (ndarray): array of shape (n_blocks, 3) collecting the displacements (first two positions) and rotations (last position) of all the blocks.
            control_params (ControlParams): contains the geometrical params in control_params.geometrical_params, as well as the bond params in control_params.mechanical_params.bond_params.

        Returns:
            float: Total strain energy.
        """

        centroid_node_vectors = control_params.geometrical_params.centroid_node_vectors
        bond_params = control_params.mechanical_params.bond_params

        n_blocks, n_nodes_per_block, _ = centroid_node_vectors.shape
        node_displacements = block_to_node_kinematics(
            block_displacement,
            centroid_node_vectors
        )
        node_displacements = node_displacements.reshape(
            (n_blocks * n_nodes_per_block, 3))

        return strain_energy_bonds(node_displacements, **bond_params._asdict())

    return strain_energy_fn


def combine_block_energies(*energy_fns: Callable):
    """Combines multiple energy functions into a single function with signature (block_displacement, control_params) -> energy.

    Args:
        *energy_fns (Callable): energy functions with signature (block_displacement, control_params) -> energy.

    Returns:
        Callable: energy function with signature (block_displacement, control_params) -> energy.
    """

    def combined_energy_fn(block_displacement: jnp.ndarray, control_params: ControlParams):
        # NOTE: Maybe there is a better way of doing this using a scan/loop. See https://github.com/google/jax/issues/673#issuecomment-894955037.
        # But, a for loop should be fine as the number of energy functions is small, so unrolling the loop should not be a problem.
        energy = jnp.array(0.)
        for energy_fn in energy_fns:
            energy += energy_fn(block_displacement, control_params)
        return energy

    return combined_energy_fn


def constrain_energy(energy_fn: Callable, constrained_kinematics: Callable):
    """Defines a constrained version of `energy_fn` according to `constrained_kinematics`.

    Args:
        energy_fn (Callable): Energy functional to be constrained.
        constrained_kinematics (Callable): Constraint function mapping the free DOFs and time to the displacement of all the blocks. Normally, this is the output of `kineamtics.build_constrained_kinematics`.

    Returns:
        Callable: Constrained energy functional with signature (free_dofs, time, control_params) -> energy.
    """

    def constrained_energy_fn(free_DOFs, t, control_params: ControlParams):
        return energy_fn(
            constrained_kinematics(
                free_DOFs, t, control_params.constraint_params),
            control_params
        )

    return constrained_energy_fn


def kinetic_energy(block_velocity, inertia):
    """
    Computes the kinetic energy of the blocks.
    """

    return jnp.sum(inertia * block_velocity**2 / 2)


def angular_momentum(block_position, block_velocity, inertia, reference_point=jnp.array([0., 0.])):
    """
    Computes the angular momentum of the blocks.

    Args:
        block_position (ndarray): array of shape (n_blocks, 2) representing the position of the blocks.
        block_velocity (ndarray): array of shape (n_blocks, 3) representing the velocity of the blocks.
        inertia (ndarray): array of shape (n_blocks, 3) representing the inertia of the blocks.
        reference_point (ndarray, optional): array of shape (2,) representing the reference point for computing the angular momentum. Defaults to jnp.array([0., 0.]).

    Returns:
        ndarray: array of shape (n_blocks,) representing the angular momentum of the blocks.
    """

    momentum_centroids = jnp.cross(block_position[:, :2] - reference_point,
                                   block_velocity[:, :2] * inertia[:, :2], axis=-1)
    momentum_rotations = block_velocity[:, 2] * inertia[:, 2]
    return momentum_centroids + momentum_rotations


def compute_ligament_strains(block_displacement, centroid_node_vectors, bond_connectivity, reference_bond_vectors):
    node_displacements = block_to_node_kinematics(
        block_displacement,
        centroid_node_vectors
    ).reshape(-1, 3)
    return ligament_strains(node_displacements[bond_connectivity[:, 0]],
                            node_displacements[bond_connectivity[:, 1]],
                            reference_vector=reference_bond_vectors)


compute_ligament_strains_history = vmap(
    compute_ligament_strains, in_axes=(0, None, None, None))
