"""
The `geometry` module implements some geometries.
"""

# NOTE: This module acts as a sort of kitchen for geometric design spaces.


from typing import Callable, Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap


# Utility functions


def rotation_matrix(angle):
    """
    docstring
    """

    return jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                      [jnp.sin(angle), jnp.cos(angle)]])


def current_coordinates(vertices, centroids, angles, displacements):
    """
    Computes the deformed configuration coordinates.
    """

    def _current_coordinates(v, Q, c, d):
        return (Q @ v.T).T + c + d

    rotations = vmap(rotation_matrix)(angles)
    current_coordinates_v = vmap(_current_coordinates, in_axes=(0, 0, 0, 0))  # Vectorize over blocks
    return current_coordinates_v(vertices, rotations, centroids, displacements)


def get_point_ids_in_bounding_box(points: jnp.ndarray, bounding_box: jnp.ndarray):
    """Returns the indices of the points that lie within the bounding box.

    Args:
        points (jnp.ndarray): array of shape (n_points, 2) collecting the coordinates of the points.
        bounding_box (jnp.ndarray): array of shape (2, 2) collecting the coordinates of the bounding box. The first row collects the coordinates of the bottom-left corner and the second row collects the coordinates of the top-right corner.

    Returns:
        jnp.ndarray: array of shape (n_points_in_bounding_box,) collecting the indices of the points that lie within the bounding box.
    """

    return jnp.where(
        (points[:, 0] >= bounding_box[0, 0]) & (points[:, 0] <= bounding_box[1, 0]) &
        (points[:, 1] >= bounding_box[0, 1]) & (points[:, 1] <= bounding_box[1, 1])
    )[0]


def get_point_ids_in_circle(points: jnp.ndarray, center: jnp.ndarray, radius: float):
    """Returns the indices of the points that lie within the circle.

    Args:
        points (jnp.ndarray): array of shape (n_points, 2) collecting the coordinates of the points.
        center (jnp.ndarray): array of shape (2,) collecting the coordinates of the center of the circle.
        radius (float): radius of the circle.

    Returns:
        jnp.ndarray: array of shape (n_points_in_circle,) collecting the indices of the points that lie within the circle.
    """

    return jnp.where(jnp.linalg.norm(points - center, axis=1) <= radius)[0]


def polygon_area(vertices: jnp.ndarray):
    """Computes area of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (jnp.ndarray): array of shape (n_vertices, 2).

    Returns:
        float: Area of the polygon.
    """

    v1 = jnp.roll(vertices, shift=1, axis=0)
    v2 = vertices

    return jnp.abs(jnp.sum(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2)


def polygon_centroid(vertices: jnp.ndarray):
    """Computes centroid of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (jnp.ndarray): array of shape (n_vertices, 2).

    Returns:
        jnp.ndarray: Centroid of the polygon.
    """

    area = polygon_area(vertices)
    v1 = jnp.roll(vertices, shift=1, axis=0)
    v2 = vertices
    x_plus_y = v1 + v2
    v_cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    return jnp.array([
        jnp.sum(x_plus_y[:, 0] * v_cross),
        jnp.sum(x_plus_y[:, 1] * v_cross)
    ]) / (6 * area)


def polygon_polar_moment(vertices: jnp.ndarray):
    """Computes polar moment of area of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (jnp.ndarray): array of shape (n_vertices, 2).

    Returns:
        float: Polar moment of area of the polygon.
    """

    centroid = polygon_centroid(vertices)
    v1 = jnp.roll(vertices, shift=1, axis=0) - centroid
    v2 = vertices - centroid

    return jnp.abs(
        jnp.sum((v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) * (
            v1[:, 0]**2 + v1[:, 0] * v2[:, 0] + v2[:, 0]**2 + v1[:, 1]**2 + v1[:, 1] * v2[:, 1] + v2[:, 1]**2
        )) / 12
    )


@vmap
def polygons_geometric_properties(vertices: jnp.ndarray):
    """Computes area, centroid, and polar moment of area of an array of polygons defined by `vertices`.

    Args:
        vertices (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: centroid, area, and polar moment of area of the polygons.
    """

    return polygon_centroid(vertices), polygon_area(vertices), polygon_polar_moment(vertices)


@jit
def compute_inertia(vertices: jnp.ndarray, density: Union[jnp.ndarray, float]):
    """Computes inertia of a set of blocks.

    Args:
        vertices (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2).
        density (Union[jnp.ndarray, float]): either a scalar or an array of shape (n_blocks, ) defining the mass density.

    Returns:
        jnp.ndarray: array of shape (n_blocks, 3) collecting the translational and rotational inertia of the blocks.
    """

    _, areas, area_moments = polygons_geometric_properties(vertices)
    translational_inertia = density * areas
    rotational_inertia = density * area_moments

    return jnp.column_stack((translational_inertia, translational_inertia, rotational_inertia))


def DOFsInfo(n_blocks: int, constrained_block_DOF_pairs: jnp.ndarray):
    """Computes arrays defining the free, constrained, and all DOFs.

    Args:
        n_blocks (int): Number of blocks in the geometry (i.e. geometry.n_blocks)
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: arrays defining the free, constrained, and all DOFs.
    """

    constrained_DOF_ids = jnp.array([block_id * 3 + DOF_id for block_id, DOF_id in constrained_block_DOF_pairs])
    all_DOF_ids = jnp.arange(n_blocks * 3)
    free_DOF_ids = jnp.array([dof for dof in all_DOF_ids if dof not in constrained_DOF_ids])

    return free_DOF_ids, constrained_DOF_ids, all_DOF_ids


def compute_edge_unit_vectors(current_block_nodes: jnp.ndarray, node_id: int):
    """Computes unit vectors from bond node to the two closest nodes of the same block.

    Args:
        current_block_coordinates (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the position of all the blocks' vertices.
        node_id (int): global node index.

    Returns:
        Tuple[jnp.array, jnp.array]: void and block angles.
    """

    _, n_sides, _ = current_block_nodes.shape

    node = current_block_nodes[node_id // n_sides, node_id % n_sides]

    unit_vector_1 = current_block_nodes[node_id // n_sides, (node_id+1) % n_sides] - node
    unit_vector_1 = unit_vector_1/jnp.linalg.norm(unit_vector_1)

    unit_vector_2 = current_block_nodes[node_id // n_sides, (node_id-1) % n_sides] - node
    unit_vector_2 = unit_vector_2/jnp.linalg.norm(unit_vector_2)

    return unit_vector_1, unit_vector_2


def compute_edge_lengths(centroid_node_vectors: jnp.ndarray):
    """Computes edge lengths of the blocks.

    Args:
        centroid_node_vectors (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the position of all the blocks' vertices relative to the centroids.

    Returns:
        jnp.ndarray: array of shape (n_blocks, n_nodes_per_block) collecting the edge lengths of the blocks.
    """

    return jnp.linalg.norm(
        jnp.roll(centroid_node_vectors, 1, axis=1) - centroid_node_vectors,
        axis=2
    )


def angle_between_unit_vectors(u1, u2):
    """Computes the signed angle between two unit vectors using arctan2.

    Args:
        u1 (jnp.ndarray): array of shape (2, ) defining the first unit vector.
        u2 (jnp.ndarray): array of shape (2, ) defining the second unit vector.

    Returns:
        float: Signed angle measured from u1 to u2 (positive counter-clockwise). Result is in the range [-pi, pi].
    """
    return jnp.arctan2(u1[0] * u2[1] - u1[1] * u2[0], u1[0] * u2[0] + u1[1] * u2[1])


def compute_edge_angles(current_block_nodes: jnp.ndarray, nodes: Tuple[int, int]):
    """Computes the two block and two void angles.

    Args:
        current_block_coordinates (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the position of all the blocks' vertices.
        nodes (Tuple[int, int]): tuple of node indices connected by a bond.

    Returns:
        Tuple[float, float, float, float]: void and block angles.
    """

    block_1_node_1, block_1_node_2 = compute_edge_unit_vectors(current_block_nodes, nodes[0])
    block_2_node_1, block_2_node_2 = compute_edge_unit_vectors(current_block_nodes, nodes[1])

    void_angle_1 = angle_between_unit_vectors(block_2_node_2, block_1_node_1)
    void_angle_2 = angle_between_unit_vectors(block_1_node_2, block_2_node_1)
    block_angle_1 = angle_between_unit_vectors(block_1_node_1, block_1_node_2)
    block_angle_2 = angle_between_unit_vectors(block_2_node_1, block_2_node_2)

    return void_angle_1, void_angle_2, block_angle_1, block_angle_2


def compute_xy_limits(points: jnp.ndarray):
    """Computes the the pair xlim, ylim for the given set of points.

    Args:
        points (jnp.ndarray): array of shape (n, 2)

    Returns:
        jnp.ndarray: array of xlim, ylim
    """

    return jnp.array([points.min(axis=0), points.max(axis=0)]).T


# Geometry classes


class Geometry:
    """
    Template class for defining geometric data.
    """

    n_blocks: int
    n_nodes: int
    block_centroids: Callable
    centroid_node_vectors: Callable
    bond_connectivity: Callable
    reference_bond_vectors: Callable

    def compute_geometry(self):
        """Any geometric class must implement the definition of the following data structures:
        - `block_centroids`: (ndarray): array of shape (n_blocks, 2) defining the centroid of each block.
        - `centroid_node_vectors` (ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the vectors connecting the centroid of the block to each node.
        - `bond_connectivity` (ndarray): array of shape (n_bonds, 2) defining the pair of nodes connected by bonds i.e. each row is of the form [node1, node2].
        - `reference_bond_vectors` (ndarray): array of shape (n_bonds, 2) defining the reference configuration of the bonds.

        Raises:
            NotImplementedError: `compute_geometry` must define `centroid_node_vectors`, `bond_connectivity`, and `reference_bond_vectors`.
        """
        raise NotImplementedError("Child classes should implement this method.")

    def get_reference_geometry(self, *args):
        """
        Computes reference configuration of all the nodes.
        """

        try:
            centroid_node_vectors = self.centroid_node_vectors(*args)
        except AttributeError as err:
            self.compute_geometry()
            centroid_node_vectors = self.centroid_node_vectors(*args)

        centroids = self.block_centroids(*args)

        return vmap(lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0))(centroid_node_vectors, centroids)

    def get_xy_limits(self, *args):
        """
        Computes reference coonfiguration xy limits.
        """

        vertices = self.get_reference_geometry(*args).reshape((self.n_nodes, 2))
        return compute_xy_limits(vertices)

    def get_parametrization(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Returns the set of functions parameterizing the geometry.

        Returns:
            Tuple[Callable, Callable, Callable, Callable]: parameterizing functions: block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors.
        """

        self.compute_geometry()

        return self.block_centroids, self.centroid_node_vectors, self.bond_connectivity, self.reference_bond_vectors


class LatticeGeometry(Geometry):
    """
    docstring
    """

    def __init__(self, n1_cells: int, n2_cells: int, n_bpc: int, direct_basis: jnp.ndarray = jnp.eye(2)):
        """Lattice geometry (not necessarily periodic) composed of unit cells arranged in a parallelepiped array.

        Args:
            n1_cells (int): Number of cells along the x direction.
            n2_cells (int): Number of cells along the y direction.
            n_bpc (int): Number of blocks per cell.
            direct_basis (jnp.ndarray, optional): Direct basis of the tesselation. Defaults to jnp.eye(2).
        """

        self.n1_cells = n1_cells
        self.n2_cells = n2_cells
        self.n_bpc = n_bpc
        self.n_cells = self.n1_cells * self.n2_cells
        self.n_blocks = self.n_cells * self.n_bpc
        self.direct_basis = direct_basis


class RotatedSquareGeometry(LatticeGeometry):
    """
    Rotated square geometry.
    """

    def __init__(self, n1_cells: int, n2_cells: int, spacing: float = 1., bond_length: float = 0.1):
        """
        Creates a rotated square lattice geometry.
        """

        super().__init__(n1_cells=n1_cells, n2_cells=n2_cells, n_bpc=4, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = 2 * self.n1_cells
        self.n2_blocks = 2 * self.n2_cells
        self.n_npb = 4
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between `angle` and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def _centroid_node_vectors(angle, n1: int, n2: int):
            v0 = (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) * \
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)])
            return vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

        def centroid_node_vectors(angle):
            """
            Computes the vectors connecting the centroid of the block to each node.
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))

            return vmap(_centroid_node_vectors, in_axes=(None, 0, 0))(angle, n1s, n2s)

        def block_centroids(angle):
            """
            Computes blocks' centroid.
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2] for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2] for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            horizontal_bonds = jnp.full(((self.n1_blocks - 1) * self.n2_blocks, 2),
                                        self.bond_length*jnp.array([1., 0.]))
            vertical_bonds = jnp.full(((self.n2_blocks - 1) * self.n1_blocks, 2),
                                      self.bond_length*jnp.array([0., 1.]))

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, initial_angle):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(initial_angle)


class KagomePeriodicGeometry(LatticeGeometry):
    """
    Kagome periodic geometry.
    """
    # [block_numeration]([cell_numeration])
    #
    #                    2(5)
    #                  /     \
    #                 /       \
    #   2(2) --- 1(1) 0(3) --- 1(4)
    #    \       /
    #     \     /
    #      0(0)

    def __init__(self, n1_cells: int, n2_cells: int, direct_basis=jnp.array([[1., 0.], [jnp.cos(jnp.pi / 3), jnp.sin(jnp.pi / 3)]]), bond_length: float = 0.1):
        """
        Creates a kagome lattice geometry.
        """

        super().__init__(n1_cells=n1_cells, n2_cells=n2_cells, n_bpc=2, direct_basis=direct_basis)
        self.bond_length = bond_length
        self.n_npb = 3
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between `shifts` and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        # Reference vectors for the bonds at the vertices of the triangles
        reference_vector_internal_bond = self.bond_length * jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)])
        reference_vector_boundary_bond_1 = self.bond_length * jnp.array([0., -1.])
        reference_vector_boundary_bond_2 = self.bond_length * jnp.array([-jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)])

        def reference_node_vectors(shifts: jnp.ndarray = jnp.zeros((3, 2))):
            # regular kagome
            a1, a2 = self.direct_basis
            block_1 = jnp.array([a1 / 2, a1 / 2 + a2 / 2, a2 / 2]) - \
                0.5*jnp.array([reference_vector_boundary_bond_1,
                              reference_vector_internal_bond,
                              reference_vector_boundary_bond_2])  # make space for bonds' length
            block_1 -= polygon_centroid(block_1)
            block_2 = vmap(lambda v: jnp.dot(rotation_matrix(-jnp.pi / 3), v))(block_1)
            # apply shifts
            block_1 += shifts
            block_2 += shifts[jnp.array([1, 2, 0])]
            # return a single cell
            return jnp.array([block_1, block_2])

        def centroid_node_vectors(shifts: jnp.ndarray = jnp.zeros((3, 2))):
            """
            Computes the vectors connecting the centroid of the block to each node.
            """

            # Compute the shifts wrt to the regular kagome
            reference_vectors = reference_node_vectors(shifts)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            cell = vmap(lambda block_nodes, shift: block_nodes - shift,
                        in_axes=(0, 0))(reference_vectors, centroid_shifts)
            return jnp.tile(cell, (self.n_cells, 1, 1))

        def block_centroids(shifts: jnp.ndarray = jnp.zeros((3, 2))):
            """
            Computes blocks' centroid.
            """

            # Compute centroids of the regular kagome.
            a1, a2 = self.direct_basis
            block_1 = polygon_centroid(jnp.array([a1 / 2, a1 / 2 + a2 / 2, a2 / 2]))
            block_2 = polygon_centroid(jnp.array([a1 / 2 + a2 / 2, a1 + a2 / 2, a1 / 2 + a2]))
            # Compute the shifts wrt to the regular kagome
            reference_vectors = reference_node_vectors(shifts)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells))
            n1s, n2s = n1s.reshape((self.n_cells,)), n2s.reshape((self.n_cells,))
            return jnp.concatenate(
                vmap(
                    lambda n1, n2: jnp.array([block_1, block_2]) + centroid_shifts + n1 * a1 + n2 * a2, in_axes=(0, 0)
                )(n1s, n2s)
            )

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)

        def translate_internal_bond(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + (n2 * self.n1_cells + n1) * n_npc

        def translate_boundary_bond1(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + jnp.array([((n2 + 1) * self.n1_cells + n1) * n_npc, (n2 * self.n1_cells + n1) * n_npc])

        def translate_boundary_bond2(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + jnp.array([(n2 * self.n1_cells + n1 + 1) * n_npc, (n2 * self.n1_cells + n1) * n_npc])

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            internal_connectivity = jnp.array([[1, 3]])
            boundary_connectivity1 = jnp.array([[0, 5]])
            boundary_connectivity2 = jnp.array([[2, 4]])
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells))
            n1s, n2s = n1s.reshape((self.n_cells,)), n2s.reshape((self.n_cells,))
            internal_bonds = jnp.concatenate(
                vmap(translate_internal_bond, in_axes=(None, 0, 0))(internal_connectivity, n1s, n2s)
            )
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells - 1))
            n1s = n1s.reshape((self.n1_cells * (self.n2_cells - 1),))
            n2s = n2s.reshape((self.n1_cells * (self.n2_cells - 1),))
            boundary_bonds1 = jnp.concatenate(
                vmap(translate_boundary_bond1, in_axes=(None, 0, 0))(boundary_connectivity1, n1s, n2s)
            )
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells - 1), jnp.arange(self.n2_cells))
            n1s = n1s.reshape(((self.n1_cells - 1) * self.n2_cells,))
            n2s = n2s.reshape(((self.n1_cells - 1) * self.n2_cells,))
            boundary_bonds2 = jnp.concatenate(
                vmap(translate_boundary_bond2, in_axes=(None, 0, 0))(boundary_connectivity2, n1s, n2s)
            )

            return jnp.concatenate([internal_bonds, boundary_bonds1, boundary_bonds2])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            internal_bonds = jnp.full(
                (self.n_cells, 2),
                reference_vector_internal_bond
            )
            boundary_bonds_1 = jnp.full(
                (self.n1_cells * (self.n2_cells - 1), 2),
                reference_vector_boundary_bond_1
            )
            boundary_bonds_2 = jnp.full(
                ((self.n1_cells - 1) * self.n2_cells, 2),
                reference_vector_boundary_bond_2
            )

            return jnp.concatenate([internal_bonds, boundary_bonds_1, boundary_bonds_2])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, shifts: jnp.ndarray = jnp.zeros((3, 2))):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(shifts)


class KagomeGeometry(LatticeGeometry):
    """
    Non-periodic Kagome geometry.
    """
    # [block_numeration]([cell_numeration])
    #
    #                    2(5)
    #                  /     \
    #                 /       \
    #   2(2) --- 1(1) 0(3) --- 1(4)
    #    \       /
    #     \     /
    #      0(0)

    def __init__(self, n1_cells: int, n2_cells: int, direct_basis=jnp.array([[1., 0.], [jnp.cos(jnp.pi / 3), jnp.sin(jnp.pi / 3)]]), bond_length: float = 0.1):
        """
        Creates a kagome lattice geometry.
        """

        super().__init__(n1_cells=n1_cells, n2_cells=n2_cells, n_bpc=2, direct_basis=direct_basis)
        self.bond_length = bond_length
        self.n_npb = 3
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between `shifts` and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        # Reference vectors for the bonds at the vertices of the triangles
        reference_vector_internal_bond = self.bond_length * jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)])
        reference_vector_boundary_bond_1 = self.bond_length * jnp.array([0., -1.])
        reference_vector_boundary_bond_2 = self.bond_length * jnp.array([-jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)])

        def _reference_node_vectors_cell_blocks(shift_1_1, shift_1_2, shift_2_1, shift_2_2, shift_3):
            """
            Computes the reference node vectors for the a single cell (2 blocks).
            Reference node vectors are the vectors from the bottom left corner of the cell to the nodes.
            Each shift here is a point in 2d space.
            shift_1_1: shift of the node (2)
            shift_1_2: shift of the node (4)
            shift_2_1: shift of the node (0)
            shift_2_2: shift of the node (5)
            shift_3: shift of the node (1)==(3)
            """
            a1, a2 = self.direct_basis
            block_1 = jnp.array([a1 / 2, a1 / 2 + a2 / 2, a2 / 2]) - \
                0.5*jnp.array([reference_vector_boundary_bond_1,
                              reference_vector_internal_bond,
                              reference_vector_boundary_bond_2])  # make space for bonds' length
            block_2 = jnp.array([a1 / 2 + a2 / 2, a1 + a2 / 2, a1 / 2 + a2]) + \
                0.5*jnp.array([reference_vector_internal_bond,
                               reference_vector_boundary_bond_2,
                               reference_vector_boundary_bond_1])  # make space for bonds' length
            # apply shifts
            block_1 += jnp.array([shift_2_1, shift_3, shift_1_1])
            block_2 += jnp.array([shift_3, shift_1_2, shift_2_2])
            # return a single cell
            return jnp.array([block_1, block_2])

        _reference_node_vectors_cell_blocks_mapped = vmap(
            vmap(_reference_node_vectors_cell_blocks, in_axes=(0, 0, 0, 0, 0)), in_axes=(0, 0, 0, 0, 0))

        def reference_node_vectors(
                shifts_1: jnp.ndarray = jnp.zeros((self.n1_cells+1, self.n2_cells, 2)),
                shifts_2: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells+1, 2)),
                shifts_3: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells, 2)),):

            reference_vectors = _reference_node_vectors_cell_blocks_mapped(
                shifts_1[:-1, :, :], shifts_1[1:, :, :], shifts_2[:, :-1, :], shifts_2[:, 1:, :], shifts_3)  # [n1_cells, n2_cells, bpc=2, npb=3, 2]
            # Transpose the first two axis to make sure that the reshaping reflect the row-wise numeration of the blocks.
            reference_vectors = jnp.transpose(
                reference_vectors,
                (1, 0, 2, 3, 4))  # [n2_cells, n1_cells, npb=3, bpc=2, 2]
            return reference_vectors.reshape((self.n_blocks, self.n_npb, 2))

        def centroid_node_vectors(
                shifts_1: jnp.ndarray = jnp.zeros((self.n1_cells+1, self.n2_cells, 2)),
                shifts_2: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells+1, 2)),
                shifts_3: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells, 2)),):
            """
            Computes the vectors connecting the centroid of the block to each node.
            """

            # Compute the shifts wrt to the regular kagome
            reference_vectors = reference_node_vectors(shifts_1, shifts_2, shifts_3)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        def reference_points():
            """
            Computes reference points of the blocks (i.e. positions on regular grid).
            """
            a1, a2 = self.direct_basis
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells))
            n1s, n2s = n1s.reshape((self.n_cells,)), n2s.reshape((self.n_cells,))
            cell_points = vmap(lambda n1, n2: n1 * a1 + n2 * a2, in_axes=(0, 0))(n1s, n2s)  # [n_cells, 2]
            return jnp.repeat(cell_points, self.n_bpc, axis=0)  # [n_blocks, 2]

        def block_centroids(
                shifts_1: jnp.ndarray = jnp.zeros((self.n1_cells+1, self.n2_cells, 2)),
                shifts_2: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells+1, 2)),
                shifts_3: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells, 2)),):
            """
            Computes blocks' centroid.
            """

            # Compute the shifts wrt the regular kagome
            reference_vectors = reference_node_vectors(shifts_1, shifts_2, shifts_3)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute blocks' centroid
            return reference_points() + centroid_shifts

        self.centroid_node_vectors = centroid_node_vectors
        self.block_centroids = block_centroids

        def translate_internal_bond(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + (n2 * self.n1_cells + n1) * n_npc

        def translate_boundary_bond1(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + jnp.array([((n2 + 1) * self.n1_cells + n1) * n_npc, (n2 * self.n1_cells + n1) * n_npc])

        def translate_boundary_bond2(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + jnp.array([(n2 * self.n1_cells + n1 + 1) * n_npc, (n2 * self.n1_cells + n1) * n_npc])

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            internal_connectivity = jnp.array([[1, 3]])
            boundary_connectivity1 = jnp.array([[0, 5]])
            boundary_connectivity2 = jnp.array([[2, 4]])
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells))
            n1s, n2s = n1s.reshape((self.n_cells,)), n2s.reshape((self.n_cells,))
            internal_bonds = jnp.concatenate(
                vmap(translate_internal_bond, in_axes=(None, 0, 0))(internal_connectivity, n1s, n2s)
            )
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells - 1))
            n1s = n1s.reshape((self.n1_cells * (self.n2_cells - 1),))
            n2s = n2s.reshape((self.n1_cells * (self.n2_cells - 1),))
            boundary_bonds1 = jnp.concatenate(
                vmap(translate_boundary_bond1, in_axes=(None, 0, 0))(boundary_connectivity1, n1s, n2s)
            )
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells - 1), jnp.arange(self.n2_cells))
            n1s = n1s.reshape(((self.n1_cells - 1) * self.n2_cells,))
            n2s = n2s.reshape(((self.n1_cells - 1) * self.n2_cells,))
            boundary_bonds2 = jnp.concatenate(
                vmap(translate_boundary_bond2, in_axes=(None, 0, 0))(boundary_connectivity2, n1s, n2s)
            )

            return jnp.concatenate([internal_bonds, boundary_bonds1, boundary_bonds2])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            internal_bonds = jnp.full(
                (self.n_cells, 2),
                reference_vector_internal_bond
            )
            boundary_bonds_1 = jnp.full(
                (self.n1_cells * (self.n2_cells - 1), 2),
                reference_vector_boundary_bond_1
            )
            boundary_bonds_2 = jnp.full(
                ((self.n1_cells - 1) * self.n2_cells, 2),
                reference_vector_boundary_bond_2
            )

            return jnp.concatenate([internal_bonds, boundary_bonds_1, boundary_bonds_2])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(
            self,
            shifts_1: jnp.ndarray,
            shifts_2: jnp.ndarray,
            shifts_3: jnp.ndarray,):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(shifts_1, shifts_2, shifts_3)


class QuadGeometry(LatticeGeometry):
    """
    Aperiodic lattice made of quadrangles with finite-length bonds.
    """

    def __init__(self, n1_blocks: int, n2_blocks: int, spacing: float = 1.0, bond_length: float = 0.1):
        """
        Creates a non-periodic lattice made of quadrangles with finite-length bonds.
        """

        super().__init__(n1_cells=n1_blocks, n2_cells=n2_blocks, n_bpc=1, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = self.n1_cells
        self.n2_blocks = self.n2_cells
        self.n_npb = 4
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between (`horizontal_shift`, `vertical_shift`) and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def reference_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the reference point (square grid) of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            v0 = (self.spacing - self.bond_length) / 2 * jnp.array([1., 0.])
            v0s = vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

            def _reference_node_vectors_block(n1_block, n2_block):
                return v0s + jnp.array([
                    horizontal_shift[n1_block+1, n2_block],
                    vertical_shift[n1_block, n2_block+1],
                    horizontal_shift[n1_block, n2_block],
                    vertical_shift[n1_block, n2_block],
                ])

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))

            return vmap(_reference_node_vectors_block, in_axes=(0, 0))(n1s, n2s)

        def centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the centroid of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            # Compute the shifts wrt to the square grid
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        def reference_points():
            """
            Computes reference points of the blocks (i.e. positions on the square grid).
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        def block_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """
            Computes blocks' centroid.
            """

            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute blocks' centroid
            return reference_points() + centroid_shifts

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2] for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2] for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            horizontal_bonds = jnp.full(((self.n1_blocks - 1) * self.n2_blocks, 2),
                                        self.bond_length*jnp.array([1., 0.]))
            vertical_bonds = jnp.full(((self.n2_blocks - 1) * self.n1_blocks, 2),
                                      self.bond_length*jnp.array([0., 1.]))

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(horizontal_shift, vertical_shift)

    def get_design_from_rotated_square(self, angle):
        """Get horizontal and vertical shifts corresponding to a rotated square geometry with the given angle.

        Args:
            angle (float): Angle of the rotated square geometry.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple of horizontal and vertical shifts.
        """

        horizontal_shifts = jnp.array([[
            (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
            jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
            jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            for n2 in range(self.n2_blocks)] for n1 in range(self.n1_blocks+1)])
        vertical_shifts = jnp.array([[
            jnp.dot(
                rotation_matrix(jnp.pi/2),
                (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
                jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            )
            for n2 in range(self.n2_blocks+1)] for n1 in range(self.n1_blocks)])

        return horizontal_shifts, vertical_shifts
