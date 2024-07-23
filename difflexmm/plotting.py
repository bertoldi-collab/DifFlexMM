import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from matplotlib import cm, colors
from matplotlib.collections import (
    LineCollection, PatchCollection, PolyCollection)
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from difflexmm.geometry import compute_xy_limits, current_coordinates, rotation_matrix
from difflexmm.utils import EigenmodeData, SolutionData, load_data


def orange_blue_cmap():
    """
    Custom colormap
    """

    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    return ListedColormap(newcolors, name='OrangeBlue')


def plot_energy(dat):
    pot_energy = []
    kin_energy = []
    for i in range(dat.fields.shape[0]):
        dx = dat.fields[i, 0, :, 0]
        dy = dat.fields[i, 0, :, 1]

        pot_energy.append(np.sum(dx**2+dy**2))
        vx = dat.fields[i, 1, :, 0]
        vy = dat.fields[i, 1, :, 1]
        kin_energy.append(np.sum(vx**2+vy**2))

    plt.figure(2)
    plt.plot(dat.timepoints, kin_energy, lw=2, label="kinetic")
    plt.plot(dat.timepoints, pot_energy, lw=2, label="potential")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.savefig("out/energy.png", dpi=300, bbox_inches='tight')


def generate_polygons(block_centroids, centroid_node_vectors, block_displacements=None, deformed=False):
    """
    docstring
    """

    if deformed and block_displacements is not None:
        polygons = [
            Polygon((rotation_matrix(DOFs[-1]) @
                    vertices.T).T + centroid + DOFs[:2])
            for vertices, centroid, DOFs in zip(centroid_node_vectors, block_centroids, block_displacements)]
    else:
        polygons = [Polygon(vertices + centroid, True)
                    for vertices, centroid in zip(centroid_node_vectors, block_centroids)]

    return polygons


def generate_patch_collection(block_centroids, centroid_node_vectors, block_displacements=None, field_values=None, deformed=False, clim=None, cmap=orange_blue_cmap()):
    """
    docstring
    """

    polygons = generate_polygons(block_centroids, centroid_node_vectors,
                                 block_displacements=block_displacements, deformed=deformed)
    patches = PatchCollection(polygons, cmap=cmap, alpha=0.95)
    if field_values is not None:
        patches.set_array(field_values)
        min_value, max_value = (
            field_values.min(), field_values.max()) if clim is None else clim
        patches.set_clim(min_value, max_value)
    patches.set(edgecolor="black", linewidth=0.5)

    return patches


def generate_bond_collection(block_centroids, centroid_node_vectors, bond_connectivity,  block_displacements=None, deformed=False):
    """
    docstring
    """

    # Generate collection of bonds as lines
    if deformed and block_displacements is not None:
        block_coords = current_coordinates(centroid_node_vectors, block_centroids,
                                           block_displacements[:, -1], block_displacements[:, :2])
    else:
        block_coords = vmap(lambda centroid, centroid_node_vector: centroid +
                            centroid_node_vector, in_axes=(0, 0))(centroid_node_vectors, block_centroids)

    n_blocks, n_npb, _ = block_coords.shape
    node_coords = block_coords.reshape((n_blocks*n_npb, 2))

    return LineCollection(node_coords[bond_connectivity], color="black", linewidth=0.5)


def plot_geometry(block_centroids, centroid_node_vectors, bond_connectivity, block_displacements=None, deformed=False, color="#2980b9", figsize=None, xlim=None, ylim=None, ax=None):
    """
    docstring
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.axis("equal")
    # Generate collection of blocks as polygons
    patches = generate_patch_collection(block_centroids, centroid_node_vectors,
                                        block_displacements=block_displacements, deformed=deformed)
    patches.set(color=color)
    patches.set(edgecolor="black", linewidth=0.5)
    ax.add_collection(patches)
    # Generate collection of bonds as lines
    collection_bonds = generate_bond_collection(
        block_centroids, centroid_node_vectors, bond_connectivity, block_displacements=block_displacements, deformed=deformed)
    ax.add_collection(collection_bonds)

    if deformed and block_displacements is not None:
        points = current_coordinates(centroid_node_vectors, block_centroids,
                                     block_displacements[:, -1], block_displacements[:, :2]).reshape((-1, 2))
    else:
        points = (block_centroids[:, None, :] +
                  centroid_node_vectors).reshape((-1, 2))

    _xlim, _ylim = compute_xy_limits(points)
    xlim = _xlim if xlim is None else xlim
    ylim = _ylim if ylim is None else ylim
    ax.set(xlim=xlim, ylim=ylim)

    fig = ax.get_figure()

    return fig, ax


def compute_field_values(data: SolutionData, field):
    if field == "ux":
        field_values = data.fields[:, 0, :, 0]
    elif field == "uy":
        field_values = data.fields[:, 0, :, 1]
    elif field == "theta":
        field_values = data.fields[:, 0, :, 2]
    elif field == "vx":
        field_values = data.fields[:, 1, :, 0]
    elif field == "vy":
        field_values = data.fields[:, 1, :, 1]
    elif field == "omega":
        field_values = data.fields[:, 1, :, 2]
    elif field == "u":
        field_values = (data.fields[:, 0, :, 0] **
                        2 + data.fields[:, 0, :, 1]**2)**0.5
    elif field == "v":
        field_values = (data.fields[:, 1, :, 0] **
                        2 + data.fields[:, 1, :, 1]**2)**0.5
    elif field == "theta_abs":
        field_values = np.abs(data.fields[:, 0, :, 2])
    else:
        raise ValueError

    return field_values


def field_name_to_label(field):
    if field == "ux":
        return r"$u_1$"
    elif field == "uy":
        return r"$u_2$"
    elif field == "theta":
        return r"$\theta$"
    elif field == "vx":
        return r"$\dot{u}_1$"
    elif field == "vy":
        return r"$\dot{u}_2$"
    elif field == "omega":
        return r"$\dot{\theta}$"
    elif field == "u":
        return r"$u$"
    elif field == "v":
        return r"$\dot{u}$"
    elif field == "theta_abs":
        return r"$\lvert\theta\rvert$"
    else:
        return field


def prepare_solution_figure(data: SolutionData, field, frame_range, figsize, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True, field_values=None):

    field_values = compute_field_values(
        data, field) if field_values is None else field_values
    _legend_label = field_name_to_label(field)

    min_value, max_value = field_values.min(), field_values.max()
    vmin, vmax = vlim if vlim is not None else (min_value, max_value)
    _legend_label = legend_label if legend_label is not None else _legend_label

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.axis("equal")
    axes.tick_params(labelsize=ticksize)
    if not axis:
        axes.axis("off")
    cb = fig.colorbar(
        cm.ScalarMappable(
            cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        pad=0.02,
        label=_legend_label,
        aspect=40
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize)
    frames = range(len(data.timepoints)
                   ) if frame_range is None else frame_range

    return field_values, min_value, max_value, fig, axes, frames


def prepare_mode_figure(data: EigenmodeData, field, mode_range, figsize, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True):

    if field == "ux":
        field_values = data.fields[:, :, 0]
        _legend_label = r"$u_1$"
    elif field == "uy":
        field_values = data.fields[:, :, 1]
        _legend_label = r"$u_2$"
    elif field == "theta":
        field_values = data.fields[:, :, 2]
        _legend_label = r"$\theta$"
    elif field == "u":
        field_values = (data.fields[:, :, 0]**2 + data.fields[:, :, 1]**2)**0.5
        _legend_label = r"$u$"
    elif field == "theta_abs":
        field_values = np.abs(data.fields[:, :, 2])
        _legend_label = r"$\lvert\theta\rvert$"
    else:
        raise ValueError

    vmin, vmax = vlim if vlim is not None else (None, None)
    _legend_label = legend_label if legend_label is not None else _legend_label

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.axis("equal")
    axes.tick_params(labelsize=ticksize)
    if not axis:
        axes.axis("off")
    cb = fig.colorbar(
        cm.ScalarMappable(
            cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        pad=0.02,
        label=_legend_label,
        aspect=40
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize)
    frames = range(len(data.fields)) if mode_range is None else mode_range

    return field_values, fig, axes, frames


def generate_mode_images(data: EigenmodeData, field, out_dir, deformed=False, mode_range=None, scale_deformation=1, figsize=None, xlim=None, ylim=None, dpi=200, geometry=None, mesh=None, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True):
    """
    mesh=None: if set to True, a mesh connecting the centroids of each block is superimposed on the images
    docstring
    """

    field_values, fig, axes, frames = prepare_mode_figure(
        data, field, mode_range, figsize, cmap=cmap, vlim=vlim, legend_label=legend_label, fontsize=fontsize, ticksize=ticksize, axis=axis
    )
    block_centroids = data.block_centroids
    centroid_node_vectors = data.centroid_node_vectors
    block_displacements = data.fields

    for i in frames:
        # Each frame refer to a mode
        patches = generate_patch_collection(
            block_centroids=block_centroids,
            centroid_node_vectors=centroid_node_vectors,
            block_displacements=block_displacements[i,
                                                    :, :] * scale_deformation,
            field_values=field_values[i],
            deformed=deformed,
            clim=None  # Normalize colors between min and max
        )
        axes.clear()
        axes.set_title(
            fr"$\Omega={data.eigenvalues[i]:.4f}$", fontsize=fontsize)
        axes.add_collection(patches)
        axes.set(xlim=xlim, ylim=ylim)

        if mesh == True:
            n1 = geometry.n1_blocks
            n2 = geometry.n2_blocks
            for j in np.arange(geometry.n2_blocks):
                row_block_coordinates = np.array([block_centroids[n1*j:n1*(j+1), 0] + block_displacements[i, n1*j:n1*(j+1), 0]*scale_deformation,
                                                  block_centroids[n1*j:n1*(j+1), 1] + block_displacements[i, n1*j:n1*(j+1), 1]*scale_deformation])
                axes.plot(row_block_coordinates[0, :],
                          row_block_coordinates[1, :], 'k')

            for k in np.arange(geometry.n1_blocks):
                col_block_coordinates = np.array([block_centroids[k:n1*(n2-1)+k+1:n1, 0] + block_displacements[i, k:n1*(n2-1)+k+1:n1, 0]*scale_deformation,
                                                  block_centroids[k:n1*(n2-1)+k+1:n1, 1] + block_displacements[i, k:n1*(n2-1)+k+1:n1, 1]*scale_deformation])
                axes.plot(col_block_coordinates[0, :],
                          col_block_coordinates[1, :], 'k')

        out_path = Path(f"{str(out_dir)}/{i:04d}.pdf")
        # Make sure parents directories exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi)

    plt.close(fig)


def generate_frames(data: SolutionData, field, out_dir, field_values=None, deformed=False, frame_range=None, figsize=None, xlim=None, ylim=None, dpi=200, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True, grid=False):
    """
    docstring
    """

    _field_values, min_value, max_value, fig, axes, frames = prepare_solution_figure(
        data, field, frame_range, figsize, cmap=cmap, vlim=vlim, legend_label=legend_label, fontsize=fontsize, ticksize=ticksize, axis=axis, field_values=field_values
    )
    block_centroids = data.block_centroids
    centroid_node_vectors = data.centroid_node_vectors
    bond_connectivity = data.bond_connectivity
    block_displacements = data.fields[:, 0, :, :]
    clim = vlim if vlim is not None else (min_value, max_value)

    for i in frames:
        # Delete old patches
        axes.clear()
        # Draw new patches
        patches = generate_patch_collection(
            block_centroids=block_centroids,
            centroid_node_vectors=centroid_node_vectors,
            block_displacements=block_displacements[i, :, :],
            field_values=_field_values[i],
            deformed=deformed,
            clim=clim,
            cmap=cmap,
        )
        axes.add_collection(patches)
        # Generate collection of bonds as lines
        collection_bonds = generate_bond_collection(
            block_centroids, centroid_node_vectors, bond_connectivity, block_displacements=block_displacements[i], deformed=deformed)
        axes.add_collection(collection_bonds)

        axes.set(xlim=xlim, ylim=ylim)
        if not grid:
            axes.grid(False)
        if not axis:
            axes.axis("off")

        out_path = Path(f"{str(out_dir)}/{i:04d}.png")
        # Make sure parents directories exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi)

    plt.close(fig)


def generate_animation(data: SolutionData, field, out_filename, field_values=None, deformed=False, frame_range=None, figsize=None, xlim=None, ylim=None, fps=20, dpi=200, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True, grid=True):
    """
    docstring
    """

    # FIXME: deformed is currently unused!
    _field_values, min_value, max_value, fig, axes, frames = prepare_solution_figure(
        data, field, frame_range, figsize, cmap=cmap, vlim=vlim, legend_label=legend_label, fontsize=fontsize, ticksize=ticksize, axis=axis, field_values=field_values
    )
    clim = vlim if vlim is not None else (min_value, max_value)
    axes.grid(grid)

    out_path = Path(f"{out_filename}.mp4")
    # Make sure parents directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate collection of blocks as polygons
    vertices = data.centroid_node_vectors
    centroids = data.block_centroids
    DOFs = data.fields[0, 0, :, :]
    block_coords = current_coordinates(
        vertices, centroids, DOFs[:, -1], DOFs[:, :2])
    collection_blocks = PolyCollection(block_coords, cmap=cmap, alpha=0.95)
    collection_blocks.set_array(_field_values[0])
    collection_blocks.set_clim(*clim)
    collection_blocks.set(edgecolor="black", linewidth=0.5)
    axes.add_collection(collection_blocks)

    if data.bond_connectivity is not None:
        # Generate collection of bonds as lines
        n_blocks, n_npb, _ = block_coords.shape
        node_coords = block_coords.reshape((n_blocks*n_npb, 2))
        collection_bonds = LineCollection(
            node_coords[data.bond_connectivity], color="black", linewidth=0.5)
        axes.add_collection(collection_bonds)

        axes.set(xlim=xlim, ylim=ylim)

        def animate_blocks_and_bonds(i):
            # Update blocks location
            DOFs = data.fields[i, 0, :, :]
            block_coords = current_coordinates(
                vertices, centroids, DOFs[:, -1], DOFs[:, :2])
            collection_blocks.set_verts(block_coords)
            collection_blocks.set_array(_field_values[i])
            # Update bonds location
            node_coords = block_coords.reshape((n_blocks*n_npb, 2))
            collection_bonds.set_segments(node_coords[data.bond_connectivity])
            axes.set(xlim=xlim, ylim=ylim)
            return collection_blocks, collection_bonds
    else:
        # Do not draw bonds
        def animate_blocks(i):
            # Update blocks location
            DOFs = data.fields[i, 0, :, :]
            block_coords = current_coordinates(
                vertices, centroids, DOFs[:, -1], DOFs[:, :2])
            collection_blocks.set_verts(block_coords)
            collection_blocks.set_array(_field_values[i])
            axes.set(xlim=xlim, ylim=ylim)
            return collection_blocks,

    animate = animate_blocks_and_bonds if data.bond_connectivity is not None else animate_blocks  # type: ignore
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, blit=True)  # type: ignore
    anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)


def plot_video_frame_field_overlaid(
        video_filename: Union[str, Path],
        solution_data: SolutionData,
        frame_number: int,
        timepoint: int,
        field: str,
        calib_xy: Tuple[float, float],
        ROI_X: Tuple[float, float],
        ROI_Y: Tuple[float, float],
        field_values: Optional[np.ndarray] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha_overlay=0.8,
        shift_px=(0, 0),
        cmap="inferno",
        figsize=(8, 5),):
    """Plot a frame of the video overlaid with the field values of the blocks.

    Args:
        video_filename (Union[str, Path]): Path to the video file.
        solution_data (SolutionData): Solution data.
        frame_number (int): Frame number to plot.
        timepoint (int): Timepoint of the solution data.
        field (str): Field to plot.
        calib_xy (Tuple[float, float]): Calibration factors for x and y.
        ROI_X (Tuple[float, float]): X range of the region of interest.
        ROI_Y (Tuple[float, float]): Y range of the region of interest.
        field_values (Optional[np.ndarray], optional): Field values. Defaults to None.
        vmin (Optional[float], optional): Minimum value of the field. Defaults to None.
        vmax (Optional[float], optional): Maximum value of the field. Defaults to None.
        alpha_overlay (float, optional): Alpha of the overlay. Defaults to 0.8.
        shift_px (Tuple[int, int], optional): Shift in pixels for alignment. Defaults to (0, 0).
        cmap (str, optional): Colormap. Defaults to "inferno".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 5).

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes.
    """

    # Load the video using opencv
    video = cv2.VideoCapture(f"{video_filename}")
    # Get frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # Read the frame
    _, frame = video.read()
    # Add alpha channel
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    # Restrict the frame to the ROI
    frame = cv2.flip(frame, 0)
    frame = frame[ROI_Y[0]:ROI_Y[1], ROI_X[0]:ROI_X[1]]
    shift_px = np.array(shift_px)

    # Compute current configuration of the blocks
    block_coordinates = current_coordinates(
        vertices=solution_data.centroid_node_vectors,
        centroids=solution_data.block_centroids,
        angles=solution_data.fields[timepoint, 0, :, 2],
        displacements=solution_data.fields[timepoint, 0, :, :2],
    )
    # Compute the field values
    field_values_all_times = compute_field_values(
        solution_data, field) if field_values is None else field_values
    field_values_min = field_values_all_times.min() if vmin is None else vmin
    field_values_max = field_values_all_times.max() if vmax is None else vmax
    _field_values = field_values_all_times[timepoint]
    # Make a colormap
    cmap = plt.get_cmap(cmap)
    # Normalize the field values
    norm = plt.Normalize(vmin=field_values_min, vmax=field_values_max)
    # Map the normalized values to colors
    field_colors = cmap(norm(_field_values))
    # Draw the blocks
    overlay = frame.copy()
    for block, color in zip(block_coordinates, field_colors):
        # Convert the block coordinates to pixels
        block_px = (np.array(block) / calib_xy[0]).astype(int) + shift_px
        # Draw the shape with the color and opacity 0.5
        cv2.fillPoly(
            overlay,
            pts=[block_px],
            # Color the block according to the field value
            color=(color[0]*255, color[1]*255, color[2]*255, 255),
        )
    # Add the overlay to the frame
    frame = cv2.addWeighted(overlay, alpha_overlay, frame, 1-alpha_overlay, 0)

    # Show the frame
    fig, ax = plt.subplots(figsize=figsize)
    # Add timepoint annotation in the bootom left corner
    ax.set_position([0, 0, 1, 1])
    ax.imshow(frame, origin="lower")
    ax.axis("off")
    # TODO:
    # Add colorbar option
    # Add timestamp label option

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        prog="DifFlexMM plotting script")
    parser.add_argument("-i", "--data-file",
                        help='Destination to pkl data file', required=True)
    parser.add_argument("-o", "--out", help="Output path.", required=True)
    parser.add_argument(
        "-f", "--field", help="Field to plot.", type=str, default="v")
    parser.add_argument("-d", "--deformed", help='Plot on deformed configuration.',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fps", help="Frame rate of the animation.", type=int, default=20)
    parser.add_argument("--dpi", help="DPI.", type=int, default=200)
    parser.add_argument("--figsize", help="Figure size.",
                        type=float, nargs=2, default=(16, 9))
    parser.add_argument("-a", "--animation", help='Produce animation or frames.',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tex", help='Use TeX fonts.',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fontsize", help='Font size.', type=int, default=20)

    parser.add_argument("-e", help='Plot Energy',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clear", help='Clear output',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "-n", help='number of processes to use', type=int, default=1)

    args = parser.parse_args()

    if args.tex:
        plt.style.use(["science"])  # enable latex fonts
    plt.rc('font', size=args.fontsize)  # font size

    data = load_data(args.data_file)

    if args.animation:
        # Generate animation
        generate_animation(data=data, field=args.field, out_filename=args.out,
                           deformed=args.deformed, fps=args.fps, dpi=args.dpi, figsize=args.figsize)
    else:
        # Generate frames
        if args.n > 1:
            # In parallel
            print(
                "Generating images in parallel.\nThere is a large overhead and may be slow.")
            global generate_frames_parallel  # Needed for multiprocessing

            def generate_frames_parallel(i):
                return generate_frames(data=data, field=args.field, out_dir=args.out,
                                       deformed=args.deformed, figsize=args.figsize, frame_range=[i])
            with Pool(args.n) as pool:
                pool.map(generate_frames_parallel, range(len(data.timepoints)))
        else:
            # Sequentially
            generate_frames(data=data, field=args.field, out_dir=args.out,
                            deformed=args.deformed, figsize=args.figsize)

    if args.e:
        plot_energy(data)


if __name__ == "__main__":
    main()
