from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import shapely.geometry as shape_geom
import shapely.ops as shape_ops
from difflexmm.geometry import QuadGeometry, RotatedSquareGeometry, compute_xy_limits
from difflexmm.plotting import generate_polygons
from jax import vmap

from jax.config import config
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Polygon, Rectangle

config.update("jax_enable_x64", True)  # enable float64 type


linewidth = 0.003*72  # "hairline" thickness in points
block_color = "black"
grip_color = "blue"
slot_color = "red"
chamfer_color = "blue"
shim_color = "black"
hole_color = "green"


def generate_chamfer_lines(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        reference_bond_vectors: jnp.ndarray,
        chamfer_depth: float):
    """
    Generates chamfer lines orthogonal to the reference bond vectors at a certain depth from the connected vertices.
    """
    # NOTE: This is somewhat deprecated.

    n_blocks, n_block_nodes, _ = centroid_node_vectors.shape
    block_nodes = vmap(lambda block_nodes, centroid: block_nodes + centroid,
                       in_axes=(0, 0))(centroid_node_vectors, block_centroids)

    def reference_director_from_node_id(node_id):
        spring_id, _ = jnp.where(bond_connectivity == node_id)
        if len(spring_id) == 0:
            return
        return reference_bond_vectors[spring_id[0]] / jnp.linalg.norm(reference_bond_vectors[spring_id[0]])

    chamfer_lines_data = []
    for block_id, block in enumerate(block_nodes):
        for node_local_id, node in enumerate(block):
            node_id = block_id*n_block_nodes + node_local_id
            director = reference_director_from_node_id(node_id)
            if director != None:
                prev_node, next_node = block[node_local_id -
                                             1], block[jnp.mod(node_local_id+1, len(block))]
                min_cos_1 = min(jnp.abs(jnp.dot((prev_node-node), director)),
                                jnp.abs(jnp.dot(director, (prev_node-node)))) / jnp.linalg.norm((prev_node-node))
                min_cos_2 = min(jnp.abs(jnp.dot((next_node-node), director)),
                                jnp.abs(jnp.dot(director, (next_node-node)))) / jnp.linalg.norm((next_node-node))
                chamfer_point_1 = node + \
                    (prev_node-node) / jnp.linalg.norm(prev_node-node) * \
                    chamfer_depth / min_cos_1
                chamfer_point_2 = node + \
                    (next_node-node) / jnp.linalg.norm(next_node-node) * \
                    chamfer_depth / min_cos_2
                chamfer_lines_data.append(
                    jnp.array([chamfer_point_1, chamfer_point_2]))

    chamfer_lines_data = jnp.array(chamfer_lines_data)

    return LineCollection(chamfer_lines_data, color=chamfer_color, linewidth=linewidth)


def generate_slot_lines_straight(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        reference_bond_vectors: jnp.ndarray,
        slot_size: float):
    """
    docstring
    """

    n_blocks, n_block_nodes, _ = centroid_node_vectors.shape
    nodes = vmap(lambda block_nodes, centroid: block_nodes + centroid,
                 in_axes=(0, 0))(centroid_node_vectors, block_centroids).reshape((n_blocks*n_block_nodes, 2))

    def slot_line(bond_nodes, reference_bond_vector):
        return bond_nodes + jnp.array([-slot_size*reference_bond_vector, slot_size*reference_bond_vector])/jnp.linalg.norm(reference_bond_vector)

    slot_lines_data = vmap(slot_line, in_axes=(0, 0))(
        nodes[bond_connectivity], reference_bond_vectors)

    return LineCollection(slot_lines_data, color=slot_color, linewidth=linewidth)


def generate_slot_lines_centroid(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        slot_size: float):
    """
    docstring
    """

    n_blocks, n_block_nodes, _ = centroid_node_vectors.shape
    nodes = vmap(lambda block_nodes, centroid: block_nodes + centroid,
                 in_axes=(0, 0))(centroid_node_vectors, block_centroids).reshape((n_blocks*n_block_nodes, 2))

    def slot_line(bond_nodes, centroid_node_vectors_bond):
        end_points = bond_nodes - slot_size * (centroid_node_vectors_bond.T /
                                               jnp.linalg.norm(centroid_node_vectors_bond, axis=-1)).T
        return jnp.array([end_points[0], bond_nodes[0], bond_nodes[1], end_points[1]])

    centroid_node_vectors_flattened = centroid_node_vectors.reshape(
        (n_blocks*n_block_nodes, 2))
    slot_lines_data = vmap(slot_line, in_axes=(0, 0))(
        nodes[bond_connectivity],
        centroid_node_vectors_flattened[bond_connectivity]
    )

    return LineCollection(slot_lines_data, color=slot_color, linewidth=linewidth)


def generate_slot_lines_bisectrix(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        slot_size: float):
    """
    docstring
    """

    n_blocks, n_block_nodes, _ = centroid_node_vectors.shape
    nodes = vmap(lambda block_nodes, centroid: block_nodes + centroid,
                 in_axes=(0, 0))(centroid_node_vectors, block_centroids).reshape((n_blocks*n_block_nodes, 2))

    edge_vectors = (jnp.roll(centroid_node_vectors, 1, axis=1) -
                    centroid_node_vectors).reshape((n_blocks*n_block_nodes, 2))
    edge_unit_vectors = (
        edge_vectors.T / jnp.linalg.norm(edge_vectors, axis=-1)).T
    edge_unit_vectors = edge_unit_vectors.reshape((n_blocks, n_block_nodes, 2))

    def bisectrix_unit_vector(u1, u2):
        cross = jnp.cross(u1, u2)
        return jnp.where(
            cross == 0,
            # rotate u2 by 90 degrees, as u1 and u2 are parallel
            jnp.array([-u2[1], u2[0]]),
            (u1 + u2) / jnp.linalg.norm(u1 + u2) * jnp.sign(cross)
        )

    edge_unit_vectors_bonds_1 = vmap(bisectrix_unit_vector, in_axes=(0, 0))(
        -edge_unit_vectors[bond_connectivity[:, 0] // n_block_nodes,
                           bond_connectivity[:, 0] % n_block_nodes].reshape(-1, 2),
        edge_unit_vectors[bond_connectivity[:, 0] // n_block_nodes,
                          (bond_connectivity[:, 0]+1) % n_block_nodes].reshape(-1, 2)
    )
    edge_unit_vectors_bonds_2 = vmap(bisectrix_unit_vector, in_axes=(0, 0))(
        -edge_unit_vectors[bond_connectivity[:, 1] // n_block_nodes,
                           bond_connectivity[:, 1] % n_block_nodes].reshape(-1, 2),
        edge_unit_vectors[bond_connectivity[:, 1] // n_block_nodes,
                          (bond_connectivity[:, 1]+1) % n_block_nodes].reshape(-1, 2)
    )
    edge_unit_vectors_bonds = vmap(lambda u1, u2: jnp.array([u1, u2]), in_axes=(0, 0))(
        edge_unit_vectors_bonds_1,
        edge_unit_vectors_bonds_2
    )

    def slot_line(bond_nodes, edge_unit_vectors_bond):
        end_points = bond_nodes + slot_size * edge_unit_vectors_bond
        return jnp.array([end_points[0], bond_nodes[0], bond_nodes[1], end_points[1]])

    slot_lines_data = vmap(slot_line, in_axes=(0, 0))(
        nodes[bond_connectivity],
        edge_unit_vectors_bonds
    )

    return LineCollection(slot_lines_data, color=slot_color, linewidth=linewidth)


def generate_slot_lines(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        reference_bond_vectors: jnp.ndarray,
        slot_size: float,
        slot_type: Literal["straight", "centroid", "bisectrix"]):
    """
    docstring
    """

    if slot_type == "straight":
        lines = generate_slot_lines_straight(block_centroids, centroid_node_vectors,
                                             bond_connectivity, reference_bond_vectors, slot_size)
    elif slot_type == "centroid":
        lines = generate_slot_lines_centroid(block_centroids, centroid_node_vectors,
                                             bond_connectivity, slot_size)
    elif slot_type == "bisectrix":
        lines = generate_slot_lines_bisectrix(block_centroids, centroid_node_vectors,
                                              bond_connectivity, slot_size)

    return lines


def generate_continuous_bond_lines_centroid(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        reference_bond_vectors: jnp.ndarray,
        n1_blocks: int,
        n2_blocks: int,
        offset_type: Literal["ratio", "constant"],
        offset_size: float = 0.,
        path_orientation: Literal["column", "row"] = "column",):
    """
    docstring
    """

    n_blocks, n_block_nodes, _ = centroid_node_vectors.shape
    nodes = vmap(lambda block_nodes, centroid: block_nodes + centroid,
                 in_axes=(0, 0))(centroid_node_vectors, block_centroids).reshape((n_blocks*n_block_nodes, 2))
    centroid_node_vectors_flattened = centroid_node_vectors.reshape(
        (n_blocks*n_block_nodes, 2))

    if offset_type == "ratio":
        centroid_node_vectors_shifted = (
            1-offset_size) * centroid_node_vectors_flattened
    else:
        centroid_node_vectors_shifted = centroid_node_vectors_flattened - offset_size * (centroid_node_vectors_flattened.T /
                                                                                         jnp.linalg.norm(centroid_node_vectors_flattened, axis=-1)).T

    def bond_line(bond_nodes, centroid_node_vectors_bond):
        end_points = bond_nodes - centroid_node_vectors_bond
        return jnp.array([end_points[0], bond_nodes[0], bond_nodes[1], end_points[1]])

    bond_lines_data = vmap(bond_line, in_axes=(0, 0))(
        nodes[bond_connectivity],
        (centroid_node_vectors_flattened -
         centroid_node_vectors_shifted)[bond_connectivity]
    )

    block_nodes_shifted = vmap(lambda block_nodes, centroid: block_nodes + centroid,
                               in_axes=(0, 0))(centroid_node_vectors_shifted.reshape((n_blocks, n_block_nodes, 2)), block_centroids)
    if path_orientation == "column":
        internal_segments_connectivity = [[[0, 1], [2, 3]] if n1 % 2 == 0 else [
            [0, 3], [1, 2]] for n1 in range(n1_blocks) for n2 in range(n2_blocks)]
    elif path_orientation == "row":
        internal_segments_connectivity = [[[0, 1], [2, 3]] if n2 % 2 == 0 else [
            [0, 3], [1, 2]] for n1 in range(n1_blocks) for n2 in range(n2_blocks)]
    else:
        raise ValueError("Wrong path orientation!")

    internal_segments = jnp.concatenate([block[jnp.array(ic)]
                                        for block, ic in zip(block_nodes_shifted, internal_segments_connectivity)])
    all_bond_lines = list(bond_lines_data) + list(internal_segments)
    # Dealing with boundaries
    block_nodes = nodes.reshape((n_blocks, n_block_nodes, 2))
    outstanding_segment_vertical = 3*reference_bond_vectors[-1]
    outstanding_segment_horizontal = 3*reference_bond_vectors[0]
    bottom_lines = jnp.array([
        [block_shifted[3], block[3], block[3]-outstanding_segment_vertical]
        for block, block_shifted in zip(block_nodes[:n1_blocks], block_nodes_shifted[:n1_blocks])
    ])
    top_lines = jnp.array([
        [block_shifted[1], block[1], block[1]+outstanding_segment_vertical]
        for block, block_shifted in zip(block_nodes[-n1_blocks:], block_nodes_shifted[-n1_blocks:])
    ])
    left_lines = jnp.array([
        [block_shifted[2], block[2], block[2]-outstanding_segment_horizontal]
        for block, block_shifted in zip(block_nodes[::n1_blocks], block_nodes_shifted[::n1_blocks])
    ])
    right_lines = jnp.array([
        [block_shifted[0], block[0], block[0]+outstanding_segment_horizontal]
        for block, block_shifted in zip(block_nodes[n1_blocks-1::n1_blocks], block_nodes_shifted[n1_blocks-1::n1_blocks])
    ])
    all_bond_lines = all_bond_lines + \
        list(bottom_lines)+list(top_lines)+list(left_lines)+list(right_lines)
    bottom_closing_lines = bottom_lines[:, -1][1:-1].reshape(-1, 2, 2)
    top_closing_lines = top_lines[:, -1].reshape(-1, 2, 2)
    left_closing_lines = left_lines[:, -1].reshape(-1, 2, 2)
    right_closing_lines = right_lines[:, -1][1:-1].reshape(-1, 2, 2)
    corner_closing_lines = [bottom_lines[-1, -1],
                            jnp.array([right_lines[0, -1][0], bottom_lines[-1, -1][1]]), right_lines[0, -1]]
    all_bond_lines = all_bond_lines + list(bottom_closing_lines) + list(top_closing_lines) + \
        list(left_closing_lines) + \
        list(right_closing_lines) + [corner_closing_lines]

    # left_auxiliary_rectangle = [bottom_lines[-1,-1], jnp.array([right_lines[0,-1][0],bottom_lines[-1,-1][1]]), right_lines[0,-1]]
    return LineCollection(all_bond_lines, color=slot_color, linewidth=linewidth)


def generate_block_lines(block_centroids: jnp.ndarray, centroid_node_vectors: jnp.ndarray):
    """
    docstring
    """

    return PatchCollection(
        generate_polygons(block_centroids, centroid_node_vectors),
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )


def generate_central_holes(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        hole_size: float,
        hole_type: Literal["circle", "square"] = "circle",
        orientation=0.):
    """
    docstring
    """

    _orientation = orientation*jnp.ones(len(block_centroids))

    holes = []
    if hole_type == "circle":
        for i in range(len(block_centroids)):
            holes.append(Circle((block_centroids[i, 0],
                                block_centroids[i, 1]), radius=hole_size/2))
    elif hole_type == "square":
        for i in range(len(block_centroids)):
            holes.append(
                Rectangle((block_centroids[i, 0]-hole_size/2, block_centroids[i, 1]-hole_size/2), width=hole_size, height=hole_size,
                          angle=_orientation[i]*180/jnp.pi, rotation_point="center")
            )
    else:
        return ValueError("Wrong shape!")

    patches_holes = PatchCollection(holes, facecolor=(0., 0., 0., 0.),
                                    edgecolor=hole_color, linewidth=linewidth)

    return patches_holes


def generate_blocks_cut_drawing(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        reference_bond_vectors: jnp.ndarray,
        slot_size: float,
        out_file: str,
        slot_type: Literal["straight", "centroid", "bisectrix"] = "straight",
        chamfer_depth: Optional[float] = None,
        block_hole_size: Optional[float] = None,
        block_hole_type: Literal["circle", "square"] = "circle",
        block_hole_orientation=0.):
    """
    docstring
    """

    fig, axes = plt.subplots()

    block_lines = generate_block_lines(block_centroids, centroid_node_vectors)
    slot_lines = generate_slot_lines(
        block_centroids,
        centroid_node_vectors,
        bond_connectivity,
        reference_bond_vectors,
        slot_size,
        slot_type,
    )
    axes.add_collection(block_lines)
    axes.add_collection(slot_lines)

    if chamfer_depth is not None:
        chamfer_lines = generate_chamfer_lines(
            block_centroids,
            centroid_node_vectors,
            bond_connectivity,
            reference_bond_vectors,
            chamfer_depth
        )
        axes.add_collection(chamfer_lines)

    if block_hole_size is not None:
        central_holes = generate_central_holes(
            block_centroids,
            centroid_node_vectors,
            block_hole_size,
            hole_type=block_hole_type,
            orientation=block_hole_orientation
        )
        axes.add_collection(central_holes)

    axes.autoscale()
    axes.axis("equal")
    axes.axis("off")

    out_path = Path(out_file)
    # Make sure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, transparent=True)
    plt.close(fig)
    print("Saved at " + str(out_path))


def generate_blocks_continuous_bond_lines_drawing(
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        reference_bond_vectors: jnp.ndarray,
        n1_blocks: int,
        n2_blocks: int,
        out_file: str,
        block_hole_size: Optional[float] = None,
        block_hole_type: Literal["circle", "square"] = "circle",
        block_hole_orientation=0.,
        offset_type: Literal["ratio", "constant"] = "constant",
        offset_size: float = 0.,
        path_orientation: Literal["column", "row"] = "column",):
    """
    docstring
    """

    fig, axes = plt.subplots()

    block_lines = generate_block_lines(block_centroids, centroid_node_vectors)
    bond_lines = generate_continuous_bond_lines_centroid(
        block_centroids,
        centroid_node_vectors,
        bond_connectivity,
        reference_bond_vectors,
        n1_blocks,
        n2_blocks,
        offset_type,
        offset_size,
        path_orientation=path_orientation,
    )
    axes.add_collection(block_lines)
    axes.add_collection(bond_lines)

    if block_hole_size is not None:
        central_holes = generate_central_holes(
            block_centroids,
            centroid_node_vectors,
            block_hole_size,
            hole_type=block_hole_type,
            orientation=block_hole_orientation
        )
        axes.add_collection(central_holes)

    axes.autoscale()
    axes.axis("equal")
    axes.axis("off")

    out_path = Path(out_file)
    # Make sure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, transparent=True)
    plt.close(fig)
    print("Saved at " + str(out_path))


def generate_shim_lines(n_shims: int, length: float, width: float, hole_size: float, n_rows: int = 1):
    """
    docstring
    """

    rectangles = []
    circles = []
    cols = [n_shims//n_rows]*(n_rows-1) + [n_shims//n_rows + n_shims % n_rows]
    for i in range(n_rows):
        for j in range(cols[i]):
            rectangles.append(
                Rectangle((0. + j*1.1*width, 0. + i*1.1*length), width, length))
            circles.append(
                Circle((0.25*width + j*1.1*width, 0.1*length +
                       i*1.1*length), radius=hole_size/2)
            )  # bottom hole
            circles.append(
                Circle((0.75*width + j*1.1*width, 0.1*length +
                       i*1.1*length), radius=hole_size/2)
            )  # bottom hole
            circles.append(
                Circle((0.25*width + j*1.1*width, (1-0.1) *
                       length + i*1.1*length), radius=hole_size/2)
            )  # top hole
            circles.append(
                Circle((0.75*width + j*1.1*width, (1-0.1) *
                       length + i*1.1*length), radius=hole_size/2)
            )  # top hole

    patches_boundary = PatchCollection(rectangles, facecolor=(0., 0., 0., 0.),
                                       edgecolor=shim_color, linewidth=linewidth)
    patches_holes = PatchCollection(circles, facecolor=(0., 0., 0., 0.),
                                    edgecolor=hole_color, linewidth=linewidth)

    return patches_boundary, patches_holes


def generate_shims_cut_drawing(n_shims: int, length: float, width: float, hole_size: float, out_file: str, n_rows: int = 1):
    """
    docstring
    """

    patches_boundary, patches_holes = generate_shim_lines(
        n_shims, length, width, hole_size, n_rows)
    fig, axes = plt.subplots()
    axes.add_collection(patches_boundary)
    axes.add_collection(patches_holes)
    axes.autoscale()
    axes.axis("equal")
    axes.axis("off")

    out_path = Path(out_file)
    # Make sure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, transparent=True)
    plt.close(fig)
    print("Saved at " + str(out_path))


def generate_grip_lines(geometry: RotatedSquareGeometry, angle, hinge_length, grip_width, grip_lateral_spacing, hole_size: Optional[float] = None):
    """
    Generates lines for grips
    """
    # NOTE: This only works for RotatedSquareGeometry!

    xlim, ylim = geometry.get_xy_limits(angle)
    e1, e2 = jnp.eye(2)
    grip_top_top_right_corner = jnp.array([xlim[1], ylim[1]]) + hinge_length*e2
    grip_top_top_left_corner = jnp.array(
        [xlim[0], ylim[1]]) + hinge_length*e2 - grip_lateral_spacing*e1
    grip_top_bottom_left_corner = jnp.array(
        [xlim[0], ylim[0]]) - grip_lateral_spacing*e1
    grip_top = Polygon(
        jnp.array([
            grip_top_top_right_corner,
            grip_top_top_right_corner + grip_width*e2,
            grip_top_top_left_corner - grip_width*e1 + grip_width*e2,
            grip_top_bottom_left_corner-grip_width*e1,
            grip_top_bottom_left_corner,
            grip_top_top_left_corner
        ]),
    )
    grip_bottom_top_right_corner = jnp.array(
        [xlim[1], ylim[1]]) + grip_lateral_spacing*e1
    grip_bottom_bottom_right_corner = jnp.array(
        [xlim[1], ylim[0]]) - hinge_length*e2 + grip_lateral_spacing*e1
    grip_bottom_bottom_left_corner = jnp.array(
        [xlim[0], ylim[0]]) - hinge_length*e2
    grip_bottom = Polygon(
        jnp.array([
            grip_bottom_top_right_corner,
            grip_bottom_bottom_right_corner,
            grip_bottom_bottom_left_corner,
            grip_bottom_bottom_left_corner - grip_width*e2,
            grip_bottom_bottom_right_corner + grip_width*e1 - grip_width*e2,
            grip_bottom_top_right_corner + grip_width*e1
        ]),
    )
    patches_grips = PatchCollection(
        [grip_top, grip_bottom],
        facecolor=(0., 0., 0., 0.),
        edgecolor=grip_color,
        linewidth=linewidth
    )
    # Add holes for bolts
    # Holes spacing: |<- ~10mm ->O<- 40mm ->O<- ~10mm ->|
    if hole_size is not None:
        holes = []
        holes.append(
            Circle(
                (xlim.mean() + 20., ylim[1] + hinge_length + 0.6*grip_width), radius=hole_size/2)
        )  # Top
        holes.append(Circle(
            (xlim.mean() - 20., ylim[1] + hinge_length + 0.6*grip_width), radius=hole_size/2))
        holes.append(
            Circle(
                (xlim.mean() + 20., ylim[0] - hinge_length - 0.6*grip_width), radius=hole_size/2)
        )  # Bottom
        holes.append(
            Circle(
                (xlim.mean() - 20., ylim[0] - hinge_length - 0.6*grip_width), radius=hole_size/2)
        )
        holes.append(
            Circle((xlim[0] - grip_lateral_spacing - 0.5 *
                   grip_width, ylim.mean() + 20.), radius=hole_size/2)
        )  # Left
        holes.append(
            Circle((xlim[0] - grip_lateral_spacing - 0.5 *
                   grip_width, ylim.mean() - 20.), radius=hole_size/2)
        )
        holes.append(
            Circle((xlim[1] + grip_lateral_spacing + 0.5 *
                   grip_width, ylim.mean() + 20.), radius=hole_size/2)
        )  # Right
        holes.append(
            Circle((xlim[1] + grip_lateral_spacing + 0.5 *
                   grip_width, ylim.mean() - 20.), radius=hole_size/2)
        )

        return patches_grips, PatchCollection(
            holes,
            facecolor=(0., 0., 0., 0.),
            edgecolor=hole_color,
            linewidth=linewidth
        )
    else:
        return patches_grips,


def generate_grippable_sample_drawing(
        geometry: RotatedSquareGeometry,
        angle,
        hinge_length,
        slot_size,
        grip_width,
        grip_lateral_spacing,
        out_file,
        block_hole_size: Optional[float] = None,
        block_hole_type: Literal["circle", "square"] = "circle",
        block_hole_orientation=0.,
        grip_hole_size: Optional[float] = None,
        slot_type: Literal["straight", "centroid", "bisectrix"] = "straight"):
    """
    Generates sample lines with top/bottom grip pieces.
    """
    # NOTE: This only works for RotatedSquareGeometry!

    block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = geometry.get_parametrization()

    fig, axes = plt.subplots()

    block_lines = generate_block_lines(
        block_centroids(angle), centroid_node_vectors(angle))
    slot_lines = generate_slot_lines(
        block_centroids(angle),
        centroid_node_vectors(angle),
        bond_connectivity(),
        reference_bond_vectors(),
        slot_size,
        slot_type
    )

    grips = generate_grip_lines(
        geometry,
        angle,
        hinge_length,
        grip_width,
        grip_lateral_spacing,
        hole_size=grip_hole_size
    )
    axes.add_collection(block_lines)
    axes.add_collection(slot_lines)

    # Add grip lines
    for patch in grips:
        axes.add_collection(patch)
    top_slot_lines = jnp.array(
        slot_lines.get_segments()
    )[-2*geometry.n1_blocks:-geometry.n1_blocks] + 2*geometry.spacing*jnp.array([0., 1.])
    grip_slot_lines = jnp.concatenate(
        [top_slot_lines, top_slot_lines - geometry.n2_blocks *
            geometry.spacing*jnp.array([0., 1.])]
    )
    axes.add_collection(LineCollection(
        grip_slot_lines, color=slot_color, linewidth=linewidth))

    if block_hole_size is not None:
        central_holes = generate_central_holes(
            block_centroids(angle),
            centroid_node_vectors(angle),
            block_hole_size,
            hole_type=block_hole_type,
            orientation=block_hole_orientation
        )
        axes.add_collection(central_holes)

    axes.autoscale()
    axes.axis("equal")
    axes.axis("off")

    out_path = Path(out_file)
    # Make sure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, transparent=True)
    plt.close(fig)
    print("Saved at " + str(out_path))


def generate_clamped_sample_drawing(
        geometry: Union[RotatedSquareGeometry, QuadGeometry],
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        reference_bond_vectors: jnp.ndarray,
        slot_size: float,
        n_blocks_clamped_corners: int,
        attachment_size: Tuple[float, float],
        attachment_hole_size: float,
        out_file,
        hole_size: Optional[float] = None,
        slot_type: Literal["straight", "centroid", "bisectrix"] = "straight"):
    """
    docstring
    """
    # NOTE: This is meant for RotatedSquareGeometry and QuadGeometry

    fig, axes = plt.subplots()

    clamped_block_ids_bl = jnp.concatenate([
        jnp.arange(0, n_blocks_clamped_corners),
        jnp.array(
            [0+i*geometry.n1_blocks for i in range(1, n_blocks_clamped_corners)])
    ])
    clamped_block_ids_br = jnp.concatenate([
        jnp.arange(geometry.n1_blocks-n_blocks_clamped_corners,
                   geometry.n1_blocks),
        jnp.array([(i+1)*geometry.n1_blocks -
                  1 for i in range(1, n_blocks_clamped_corners)])
    ])
    clamped_block_ids_tr = jnp.concatenate([
        jnp.arange(geometry.n_blocks-n_blocks_clamped_corners,
                   geometry.n_blocks),
        jnp.array([geometry.n_blocks-i*geometry.n1_blocks -
                  1 for i in range(1, n_blocks_clamped_corners)])
    ])
    clamped_block_ids_tl = jnp.concatenate([
        jnp.arange(geometry.n_blocks-geometry.n1_blocks, geometry.n_blocks -
                   geometry.n1_blocks+n_blocks_clamped_corners),
        jnp.array([geometry.n_blocks-geometry.n1_blocks-i *
                   geometry.n1_blocks for i in range(1, n_blocks_clamped_corners)])
    ])
    block_ids_clamped_corners = jnp.concatenate([
        clamped_block_ids_bl, clamped_block_ids_br, clamped_block_ids_tr, clamped_block_ids_tl
    ])
    block_ids_inner = jnp.setdiff1d(jnp.arange(
        geometry.n_blocks), block_ids_clamped_corners)
    bond_ids_corners = jnp.concatenate([
        jnp.arange(0, n_blocks_clamped_corners-1),
        jnp.arange(geometry.n1_blocks-n_blocks_clamped_corners,
                   geometry.n1_blocks-1),
        jnp.arange((geometry.n1_blocks-1)*geometry.n2_blocks-geometry.n1_blocks+1,
                   (geometry.n1_blocks-1)*geometry.n2_blocks-geometry.n1_blocks+n_blocks_clamped_corners),
        jnp.arange((geometry.n1_blocks-1)*geometry.n2_blocks-n_blocks_clamped_corners+1,
                   (geometry.n1_blocks-1)*geometry.n2_blocks),  # horizontal corner bonds
        jnp.array(
            [i*geometry.n1_blocks for i in jnp.arange(n_blocks_clamped_corners-1)]
        ) + (geometry.n1_blocks-1)*geometry.n2_blocks,
        jnp.array(
            [geometry.n1_blocks-1+i * \
                geometry.n1_blocks for i in jnp.arange(n_blocks_clamped_corners-1)]
        ) + (geometry.n1_blocks-1)*geometry.n2_blocks,
        jnp.array(
            [geometry.n1_blocks*(geometry.n2_blocks-1)-geometry.n1_blocks*(n_blocks_clamped_corners-1) + \
             i*geometry.n1_blocks for i in jnp.arange(n_blocks_clamped_corners-1)]
        ) + (geometry.n1_blocks-1)*geometry.n2_blocks,
        jnp.array(
            [geometry.n1_blocks*(geometry.n2_blocks-1)-1-geometry.n1_blocks*(n_blocks_clamped_corners-2) + \
             i*geometry.n1_blocks for i in jnp.arange(n_blocks_clamped_corners-1)]
        ) + (geometry.n1_blocks-1)*geometry.n2_blocks,  # vertical corner bonds
    ])
    bond_ids_inner = jnp.setdiff1d(jnp.arange(
        len(bond_connectivity)), bond_ids_corners)

    block_lines_inner = generate_block_lines(
        block_centroids[block_ids_inner], centroid_node_vectors[block_ids_inner])
    slot_lines = generate_slot_lines(
        block_centroids,
        centroid_node_vectors,
        bond_connectivity[bond_ids_inner],
        reference_bond_vectors[bond_ids_inner],
        slot_size,
        slot_type,
    )
    axes.add_collection(block_lines_inner)
    axes.add_collection(slot_lines)

    # Attachment holes
    hole_shift_x = (25.4 - jnp.mod((geometry.n1_blocks-1) *
                    geometry.spacing, 25.4))/2  # Based on 1" breadboard
    hole_shift_y = (25.4 - jnp.mod((geometry.n2_blocks-1) *
                    geometry.spacing, 25.4))/2  # Based on 1" breadboard
    hole_position_bl = (-hole_shift_x, -hole_shift_y)
    hole_position_br = ((geometry.n1_blocks-1) *
                        geometry.spacing + hole_shift_x, -hole_shift_y)
    hole_position_tl = (-hole_shift_x, (geometry.n2_blocks-1)
                        * geometry.spacing+hole_shift_y)
    hole_position_tr = ((geometry.n1_blocks-1)*geometry.spacing+hole_shift_x,
                        (geometry.n2_blocks-1)*geometry.spacing+hole_shift_y)

    for center in [hole_position_bl, hole_position_br, hole_position_tl, hole_position_tr]:
        axes.add_patch(
            Circle(
                center, radius=attachment_hole_size/2,
                facecolor=(0., 0., 0., 0.),
                edgecolor=hole_color,
                linewidth=linewidth
            )
        )

    # Attachments at corners
    leg_width = 0.5*geometry.spacing
    # bottom left
    blocks_bl = vmap(
        lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0)
    )(centroid_node_vectors[clamped_block_ids_bl], block_centroids[clamped_block_ids_bl])
    xlim_bl, ylim_bl = compute_xy_limits(blocks_bl.reshape(
        (2*n_blocks_clamped_corners-1)*geometry.n_npb, 2))
    attachment_bl_points = jnp.array(
        shape_ops.unary_union([
            *(shape_geom.Polygon(block) for block in blocks_bl),
            shape_geom.box(xlim_bl[0], ylim_bl[0], xlim_bl[1] -
                           0.25*geometry.spacing, ylim_bl[0]+leg_width),
            shape_geom.box(xlim_bl[0], ylim_bl[0], xlim_bl[0] +
                           leg_width, ylim_bl[1]-0.25*geometry.spacing),
            shape_geom.box(
                hole_position_bl[0]-attachment_size[0]/2,
                hole_position_bl[1]-attachment_size[1]/2,
                hole_position_bl[0]+attachment_size[0]/2,
                hole_position_bl[1]+attachment_size[1]/2
            )
        ]).exterior.xy
    ).T
    attachment_bl = Polygon(
        attachment_bl_points,
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )

    # bottom right
    blocks_br = vmap(
        lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0)
    )(centroid_node_vectors[clamped_block_ids_br], block_centroids[clamped_block_ids_br])
    xlim_br, ylim_br = compute_xy_limits(blocks_br.reshape(
        (2*n_blocks_clamped_corners-1)*geometry.n_npb, 2))
    attachment_br_points = jnp.array(
        shape_ops.unary_union([
            *(shape_geom.Polygon(block) for block in blocks_br),
            shape_geom.box(xlim_br[0]+0.25*geometry.spacing,
                           ylim_br[0], xlim_br[1], ylim_br[0] + leg_width),
            shape_geom.box(xlim_br[1], ylim_br[0], xlim_br[1] -
                           leg_width, ylim_br[1]-0.25*geometry.spacing),
            shape_geom.box(
                hole_position_br[0]-attachment_size[0]/2,
                hole_position_br[1]-attachment_size[1]/2,
                hole_position_br[0]+attachment_size[0]/2,
                hole_position_br[1]+attachment_size[1]/2
            )
        ]).exterior.xy
    ).T
    attachment_br = Polygon(
        attachment_br_points,
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )

    # top left
    blocks_tl = vmap(
        lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0)
    )(centroid_node_vectors[clamped_block_ids_tl], block_centroids[clamped_block_ids_tl])
    xlim_tl, ylim_tl = compute_xy_limits(blocks_tl.reshape(
        (2*n_blocks_clamped_corners-1)*geometry.n_npb, 2))
    attachment_tl_points = jnp.array(
        shape_ops.unary_union([
            *(shape_geom.Polygon(block) for block in blocks_tl),
            shape_geom.box(xlim_tl[0], ylim_tl[1], xlim_tl[1] -
                           0.25*geometry.spacing, ylim_tl[1] - leg_width),
            shape_geom.box(xlim_tl[0], ylim_tl[1], xlim_tl[0] +
                           leg_width, ylim_tl[0]+0.25*geometry.spacing),
            shape_geom.box(
                hole_position_tl[0]-attachment_size[0]/2,
                hole_position_tl[1]-attachment_size[1]/2,
                hole_position_tl[0]+attachment_size[0]/2,
                hole_position_tl[1]+attachment_size[1]/2
            )
        ]).exterior.xy
    ).T
    attachment_tl = Polygon(
        attachment_tl_points,
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )

    # top right
    blocks_tr = vmap(
        lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0)
    )(centroid_node_vectors[clamped_block_ids_tr], block_centroids[clamped_block_ids_tr])
    xlim_tr, ylim_tr = compute_xy_limits(blocks_tr.reshape(
        (2*n_blocks_clamped_corners-1)*geometry.n_npb, 2))
    attachment_tr_points = jnp.array(
        shape_ops.unary_union([
            *(shape_geom.Polygon(block) for block in blocks_tr),
            shape_geom.box(xlim_tr[1], ylim_tr[1], xlim_tr[0] +
                           0.25*geometry.spacing, ylim_tr[1] - leg_width),
            shape_geom.box(xlim_tr[1], ylim_tr[1], xlim_tr[1] -
                           leg_width, ylim_tr[0]+0.25*geometry.spacing),
            shape_geom.box(
                hole_position_tr[0]-attachment_size[0]/2,
                hole_position_tr[1]-attachment_size[1]/2,
                hole_position_tr[0]+attachment_size[0]/2,
                hole_position_tr[1]+attachment_size[1]/2
            )
        ]).exterior.xy
    ).T
    attachment_tr = Polygon(
        attachment_tr_points,
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )

    for patch in [attachment_bl, attachment_br, attachment_tl, attachment_tr]:
        axes.add_patch(patch)

    # NOTE: Check design is contained in one 24x12" acrylic plate
    xlim, ylim = compute_xy_limits(
        jnp.concatenate([attachment_bl_points, attachment_br_points,
                        attachment_tl_points, attachment_tr_points])
    )
    assert xlim[1]-xlim[0] < 25.4*24, "Design is wider than 24 inches!"
    assert ylim[1]-ylim[0] < 25.4*12, "Design is taller than 12 inches!"

    axes.autoscale()
    axes.axis("equal")
    axes.axis("off")

    out_path = Path(out_file)
    # Make sure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, transparent=True)
    plt.close(fig)
    print("Saved at " + str(out_path))


def generate_clamped_sample_continuous_bond_drawing(
        geometry: Union[RotatedSquareGeometry, QuadGeometry],
        block_centroids: jnp.ndarray,
        centroid_node_vectors: jnp.ndarray,
        bond_connectivity: jnp.ndarray,
        reference_bond_vectors: jnp.ndarray,
        n_blocks_clamped_corners: int,
        attachment_size: Tuple[float, float],
        attachment_hole_size: float,
        out_file,
        offset_type: Literal["ratio", "constant"],
        offset_size: float = 0.,
        path_orientation: Literal["column", "row"] = "column",):
    """
    docstring
    """
    # NOTE: This is meant for RotatedSquareGeometry and QuadGeometry

    fig, axes = plt.subplots()

    clamped_block_ids_bl = jnp.concatenate([
        jnp.arange(0, n_blocks_clamped_corners),
        jnp.array(
            [0+i*geometry.n1_blocks for i in range(1, n_blocks_clamped_corners)])
    ]).astype(jnp.int64)
    clamped_block_ids_br = jnp.concatenate([
        jnp.arange(geometry.n1_blocks-n_blocks_clamped_corners,
                   geometry.n1_blocks),
        jnp.array([(i+1)*geometry.n1_blocks -
                  1 for i in range(1, n_blocks_clamped_corners)])
    ]).astype(jnp.int64)
    clamped_block_ids_tr = jnp.concatenate([
        jnp.arange(geometry.n_blocks-n_blocks_clamped_corners,
                   geometry.n_blocks),
        jnp.array([geometry.n_blocks-i*geometry.n1_blocks -
                  1 for i in range(1, n_blocks_clamped_corners)])
    ]).astype(jnp.int64)
    clamped_block_ids_tl = jnp.concatenate([
        jnp.arange(geometry.n_blocks-geometry.n1_blocks, geometry.n_blocks -
                   geometry.n1_blocks+n_blocks_clamped_corners),
        jnp.array([geometry.n_blocks-geometry.n1_blocks-i *
                   geometry.n1_blocks for i in range(1, n_blocks_clamped_corners)])
    ]).astype(jnp.int64)
    block_ids_clamped_corners = jnp.concatenate([
        clamped_block_ids_bl, clamped_block_ids_br, clamped_block_ids_tr, clamped_block_ids_tl
    ])
    block_ids_inner = jnp.setdiff1d(jnp.arange(
        geometry.n_blocks), block_ids_clamped_corners)
    bond_ids_corners = jnp.concatenate([
        jnp.arange(0, n_blocks_clamped_corners-1),
        jnp.arange(geometry.n1_blocks-n_blocks_clamped_corners,
                   geometry.n1_blocks-1),
        jnp.arange((geometry.n1_blocks-1)*geometry.n2_blocks-geometry.n1_blocks+1,
                   (geometry.n1_blocks-1)*geometry.n2_blocks-geometry.n1_blocks+n_blocks_clamped_corners),
        jnp.arange((geometry.n1_blocks-1)*geometry.n2_blocks-n_blocks_clamped_corners+1,
                   (geometry.n1_blocks-1)*geometry.n2_blocks),  # horizontal corner bonds
        jnp.array(
            [i*geometry.n1_blocks for i in jnp.arange(n_blocks_clamped_corners-1)]
        ) + (geometry.n1_blocks-1)*geometry.n2_blocks,
        jnp.array(
            [geometry.n1_blocks-1+i * \
                geometry.n1_blocks for i in jnp.arange(n_blocks_clamped_corners-1)]
        ) + (geometry.n1_blocks-1)*geometry.n2_blocks,
        jnp.array(
            [geometry.n1_blocks*(geometry.n2_blocks-1)-geometry.n1_blocks*(n_blocks_clamped_corners-1) + \
             i*geometry.n1_blocks for i in jnp.arange(n_blocks_clamped_corners-1)]
        ) + (geometry.n1_blocks-1)*geometry.n2_blocks,
        jnp.array(
            [geometry.n1_blocks*(geometry.n2_blocks-1)-1-geometry.n1_blocks*(n_blocks_clamped_corners-2) + \
             i*geometry.n1_blocks for i in jnp.arange(n_blocks_clamped_corners-1)]
        ) + (geometry.n1_blocks-1)*geometry.n2_blocks,  # vertical corner bonds
    ])
    bond_ids_inner = jnp.setdiff1d(jnp.arange(
        len(bond_connectivity)), bond_ids_corners)

    block_lines_inner = generate_block_lines(
        block_centroids[block_ids_inner], centroid_node_vectors[block_ids_inner])
    bond_lines = generate_continuous_bond_lines_centroid(
        block_centroids,
        centroid_node_vectors,
        bond_connectivity,
        reference_bond_vectors,
        geometry.n1_blocks,
        geometry.n2_blocks,
        offset_type,
        offset_size,
        path_orientation=path_orientation,
    )
    axes.add_collection(bond_lines)
    axes.add_collection(block_lines_inner)

    # Attachment holes
    hole_shift_x = (25.4 - jnp.mod((geometry.n1_blocks-1) *
                    geometry.spacing, 25.4))/2  # Based on 1" breadboard
    hole_shift_y = (25.4 - jnp.mod((geometry.n2_blocks-1) *
                    geometry.spacing, 25.4))/2  # Based on 1" breadboard
    hole_position_bl = (-hole_shift_x, -hole_shift_y)
    hole_position_br = ((geometry.n1_blocks-1) *
                        geometry.spacing + hole_shift_x, -hole_shift_y)
    hole_position_tl = (-hole_shift_x, (geometry.n2_blocks-1)
                        * geometry.spacing+hole_shift_y)
    hole_position_tr = ((geometry.n1_blocks-1)*geometry.spacing+hole_shift_x,
                        (geometry.n2_blocks-1)*geometry.spacing+hole_shift_y)

    for center in [hole_position_bl, hole_position_br, hole_position_tl, hole_position_tr]:
        axes.add_patch(
            Circle(
                center, radius=attachment_hole_size/2,
                facecolor=(0., 0., 0., 0.),
                edgecolor=hole_color,
                linewidth=linewidth
            )
        )

    # Attachments at corners
    leg_width = 0.5*geometry.spacing
    # bottom left
    blocks_bl = vmap(
        lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0)
    )(centroid_node_vectors[clamped_block_ids_bl], block_centroids[clamped_block_ids_bl])
    xlim_bl, ylim_bl = compute_xy_limits(blocks_bl.reshape(
        (2*n_blocks_clamped_corners-1)*geometry.n_npb, 2))
    attachment_bl_points = jnp.array(
        shape_ops.unary_union([
            *(shape_geom.Polygon(block) for block in blocks_bl),
            shape_geom.box(xlim_bl[0], ylim_bl[0], xlim_bl[1] -
                           0.25*geometry.spacing, ylim_bl[0]+leg_width),
            shape_geom.box(xlim_bl[0], ylim_bl[0], xlim_bl[0] +
                           leg_width, ylim_bl[1]-0.25*geometry.spacing),
            shape_geom.box(
                hole_position_bl[0]-attachment_size[0]/2,
                hole_position_bl[1]-attachment_size[1]/2,
                hole_position_bl[0]+attachment_size[0]/2,
                hole_position_bl[1]+attachment_size[1]/2
            )
        ]).exterior.xy
    ).T
    attachment_bl = Polygon(
        attachment_bl_points,
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )

    # bottom right
    blocks_br = vmap(
        lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0)
    )(centroid_node_vectors[clamped_block_ids_br], block_centroids[clamped_block_ids_br])
    xlim_br, ylim_br = compute_xy_limits(blocks_br.reshape(
        (2*n_blocks_clamped_corners-1)*geometry.n_npb, 2))
    attachment_br_points = jnp.array(
        shape_ops.unary_union([
            *(shape_geom.Polygon(block) for block in blocks_br),
            shape_geom.box(xlim_br[0]+0.25*geometry.spacing,
                           ylim_br[0], xlim_br[1], ylim_br[0] + leg_width),
            shape_geom.box(xlim_br[1], ylim_br[0], xlim_br[1] -
                           leg_width, ylim_br[1]-0.25*geometry.spacing),
            shape_geom.box(
                hole_position_br[0]-attachment_size[0]/2,
                hole_position_br[1]-attachment_size[1]/2,
                hole_position_br[0]+attachment_size[0]/2,
                hole_position_br[1]+attachment_size[1]/2
            )
        ]).exterior.xy
    ).T
    attachment_br = Polygon(
        attachment_br_points,
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )

    # top left
    blocks_tl = vmap(
        lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0)
    )(centroid_node_vectors[clamped_block_ids_tl], block_centroids[clamped_block_ids_tl])
    xlim_tl, ylim_tl = compute_xy_limits(blocks_tl.reshape(
        (2*n_blocks_clamped_corners-1)*geometry.n_npb, 2))
    attachment_tl_points = jnp.array(
        shape_ops.unary_union([
            *(shape_geom.Polygon(block) for block in blocks_tl),
            shape_geom.box(xlim_tl[0], ylim_tl[1], xlim_tl[1] -
                           0.25*geometry.spacing, ylim_tl[1] - leg_width),
            shape_geom.box(xlim_tl[0], ylim_tl[1], xlim_tl[0] +
                           leg_width, ylim_tl[0]+0.25*geometry.spacing),
            shape_geom.box(
                hole_position_tl[0]-attachment_size[0]/2,
                hole_position_tl[1]-attachment_size[1]/2,
                hole_position_tl[0]+attachment_size[0]/2,
                hole_position_tl[1]+attachment_size[1]/2
            )
        ]).exterior.xy
    ).T
    attachment_tl = Polygon(
        attachment_tl_points,
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )

    # top right
    blocks_tr = vmap(
        lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0)
    )(centroid_node_vectors[clamped_block_ids_tr], block_centroids[clamped_block_ids_tr])
    xlim_tr, ylim_tr = compute_xy_limits(blocks_tr.reshape(
        (2*n_blocks_clamped_corners-1)*geometry.n_npb, 2))
    attachment_tr_points = jnp.array(
        shape_ops.unary_union([
            *(shape_geom.Polygon(block) for block in blocks_tr),
            shape_geom.box(xlim_tr[1], ylim_tr[1], xlim_tr[0] +
                           0.25*geometry.spacing, ylim_tr[1] - leg_width),
            shape_geom.box(xlim_tr[1], ylim_tr[1], xlim_tr[1] -
                           leg_width, ylim_tr[0]+0.25*geometry.spacing),
            shape_geom.box(
                hole_position_tr[0]-attachment_size[0]/2,
                hole_position_tr[1]-attachment_size[1]/2,
                hole_position_tr[0]+attachment_size[0]/2,
                hole_position_tr[1]+attachment_size[1]/2
            )
        ]).exterior.xy
    ).T
    attachment_tr = Polygon(
        attachment_tr_points,
        facecolor=(0., 0., 0., 0.),
        edgecolor=block_color,
        linewidth=linewidth
    )

    for patch in [attachment_bl, attachment_br, attachment_tl, attachment_tr]:
        axes.add_patch(patch)

    # NOTE: Check design is contained in one 24x12" acrylic plate
    xlim, ylim = compute_xy_limits(
        jnp.concatenate([attachment_bl_points, attachment_br_points,
                        attachment_tl_points, attachment_tr_points])
    )
    print(f"Design size is {xlim[1]-xlim[0]:.2f}x{ylim[1]-ylim[0]:.2f} mm")

    axes.autoscale()
    axes.axis("equal")
    axes.axis("off")

    out_path = Path(out_file)
    # Make sure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, transparent=True)
    plt.close(fig)
    print("Saved at " + str(out_path))


def generate_blocks_continuous_bond_lines_drawing_grippable(
        geometry: RotatedSquareGeometry,
        angle,
        hinge_length,
        grip_width,
        grip_lateral_spacing,
        out_file,
        block_hole_size: Optional[float] = None,
        block_hole_type: Literal["circle", "square"] = "circle",
        block_hole_orientation=0.,
        grip_hole_size: Optional[float] = None,
        offset_type: Literal["ratio", "constant"] = "constant",
        offset_size: float = 0.,
        path_orientation: Literal["column", "row"] = "column",):
    """
    docstring
    """
    block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = geometry.get_parametrization()

    fig, axes = plt.subplots()

    block_lines = generate_block_lines(
        block_centroids(angle), centroid_node_vectors(angle))
    bond_lines = generate_continuous_bond_lines_centroid(
        block_centroids(angle),
        centroid_node_vectors(angle),
        bond_connectivity(),
        reference_bond_vectors(),
        geometry.n1_blocks,
        geometry.n2_blocks,
        offset_type,
        offset_size,
        path_orientation=path_orientation,
    )
    axes.add_collection(block_lines)
    axes.add_collection(bond_lines)

    grips = generate_grip_lines(
        geometry,
        angle,
        hinge_length,
        grip_width,
        grip_lateral_spacing,
        hole_size=grip_hole_size
    )

    # Add grip lines
    for patch in grips:
        axes.add_collection(patch)

    if block_hole_size is not None:
        central_holes = generate_central_holes(
            block_centroids(angle),
            centroid_node_vectors(angle),
            block_hole_size,
            hole_type=block_hole_type,
            orientation=block_hole_orientation
        )
        axes.add_collection(central_holes)

    axes.autoscale()
    axes.axis("equal")
    axes.axis("off")

    out_path = Path(out_file)
    # Make sure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, transparent=True)
    plt.close(fig)
    print("Saved at " + str(out_path))
