import argparse
from typing import Literal

import cv2
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy.signal import savgol_filter
from difflexmm.geometry import compute_xy_limits
from scipy import interpolate


# Set default parameter values
max_angle_change_default = 360
conv_size_default = [[0]*3, [0]*3]  # Kernel sizes for convolution [[x, y, theta], [x_dot, y_dot, theta_dot]]
step_size_default = 1
adaptive_thresholding_block_default = 11
aspect_ratio_threshold_default = 0.3
# Cross-correlation parameters
search_window_size_default = 40
marker_template_size_default = 20
upscaling_factor_default = 5


def morphological_transformation_default(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    transformed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return transformed


def collect_as(coll_type):
    class Collect_as(argparse.Action):
        def __call__(self, parser, namespace, values, options_string=None):
            setattr(namespace, self.dest, coll_type(values))

    return Collect_as


def closest_block(node, nodes):
    node0 = node[:2]
    nodes0 = nodes[:, :2]

    dist_2 = np.sum((nodes0 - node0) ** 2, axis=1)
    i = np.argmin(dist_2)
    min_dist = np.min(dist_2)

    return i, min_dist


def interpolate_nans(solution_fields):
    """Interpolate NaNs in solution fields."""
    # Replace NaNs with interpolated values.
    n_timepoints = solution_fields.shape[0]
    mask_nans = np.isnan(solution_fields)
    not_nan_times = np.all(np.logical_not(mask_nans), axis=(1, 2, 3))
    f = interpolate.interp1d(
        np.arange(n_timepoints)[not_nan_times],
        solution_fields[not_nan_times],
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )
    _solution_fields = solution_fields.copy()
    _solution_fields[mask_nans] = f(np.arange(n_timepoints))[mask_nans]
    return _solution_fields


def calculate_displacement(rect_prev, contours_next, n_blocks, calib_xy, max_angle_change, max_displacement_change, aspect_ratio_threshold):
    """Calculate the displacement of the blocks between two frames."""

    rect_displacement = np.zeros((n_blocks, 3))
    contour_centroids_next = np.zeros((len(contours_next), 2))
    for i, contour in enumerate(contours_next):
        contour_centroids_next[i] = compute_centroid(contour)

    for r_prev in rect_prev:

        r_next = np.zeros((5,))
        next_id, _ = closest_block(r_prev, contour_centroids_next)
        method = r_prev[4]  # method is inherited from previous frame
        fitted_contour, method = fit_contour(
            contours_next[next_id],
            method=method,
            aspect_ratio_threshold=aspect_ratio_threshold,
        )
        r_next[:2] = contour_centroids_next[next_id]  # centroid
        r_next[2] = fitted_contour[-1]  # angle
        r_next[4] = method  # fitting method

        index = int(r_prev[3])

        delta_x = (r_next[0] - r_prev[0]) * calib_xy[0]
        delta_y = (r_next[1] - r_prev[1]) * calib_xy[1]
        delta_theta = r_next[2] - r_prev[2]

        if max_displacement_change is None:
            _max_displacement_change = np.inf
        else:
            _max_displacement_change = max_displacement_change

        if (delta_x**2 + delta_y**2)**0.5 > _max_displacement_change:
            # NOTE: If the displacement is too large for this contour, it is set to NaN. NaNs are going to be replaced by interpolated values.
            # This is to avoid jumps in the tracking caused by a wrong contour being matched or a contour being lost for a frame.
            rect_displacement[index][0] = np.nan
            rect_displacement[index][1] = np.nan
            rect_displacement[index][2] = np.nan
        else:
            rect_displacement[index][0] = delta_x
            rect_displacement[index][1] = delta_y

            # Correct for 90° jumps in angle due to different definitions of angle between fitEllipse and minAreaRect
            method = r_next[4]  # 0: minAreaRect, 1: fitEllipse
            angle_correction = np.sign(delta_theta) * 180 if method == 1 else np.sign(delta_theta) * 90
            if np.abs(delta_theta) > 45:
                rect_displacement[index][2] = (
                    (delta_theta - angle_correction) / 180 * np.pi
                )
            else:
                rect_displacement[index][2] = delta_theta / 180 * np.pi

            # Make sure that the angle change is not too large
            if rect_displacement[index][2] >= max_angle_change:
                rect_displacement[index][2] = 0

            r_next[3] = index
            rect_prev[index] = r_next

    return rect_displacement


def sort_contours(contours, reference_centroids, calib_xy):

    # Compute shifts between reference centroids and centroids of contours
    contour_centroids = np.array([compute_centroid(c) for c in contours]) * np.array(
        calib_xy
    )
    xylim_contours = compute_xy_limits(contour_centroids)
    xylim_reference = compute_xy_limits(reference_centroids)
    reference_centroids_shifted = reference_centroids + (
        xylim_contours[:, 0] - xylim_reference[:, 0]
    )

    contours_sorted = []
    for ref_centroid in reference_centroids_shifted:
        closest_index = np.argmin(
            np.linalg.norm(contour_centroids - ref_centroid, axis=1)
        )
        contours_sorted.append(contours[closest_index])

    return contours_sorted


def get_blob(shape, contour):
    black = np.zeros(shape[:2])
    img_blob = cv2.drawContours(black, [contour], -1, color=255, thickness=cv2.FILLED)
    blob = np.flip(np.argwhere(img_blob == 255), axis=1)
    return blob.reshape(-1, 1, 2)


def compute_centroid(contour):
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return 0, 0

    # calculate x, y coordinate of center
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
    # cX, cY = np.mean(contour, axis=0)[0]

    return cX, cY


def fit_contour(countour, method: Literal[None, 0, 1] = None, aspect_ratio_threshold=aspect_ratio_threshold_default):
    """Fit a contour to a rectangle or ellipse.
    Note: The angle of the rectangle is defined differently than the angle of the ellipse.
    method: 0: minAreaRect, 1: fitEllipse
    """

    if method is None:
        rectangle = cv2.minAreaRect(countour)
        ellipse = cv2.fitEllipse(countour)
        _, (MA, ma), _ = ellipse
        if np.abs(MA - ma)/((MA + ma)/2) < aspect_ratio_threshold:
            return rectangle, 0
        else:
            return ellipse, 1
    elif method == 0:
        return cv2.minAreaRect(countour), 0
    elif method == 1:
        return cv2.fitEllipse(countour), 1


def find_markers(template_frame, search_frame, template_markers, search_markers, search_window_size=40, marker_template_size=20, upscaling_factor=5):
    """Find the markers by cross-correlating the search frame with the template frame.

    Args:
        template_frame: The frame (grayscale) used for templates.
        search_frame: The frame (grayscale) used for search.
        template_markers: The positions of the markers in the template frame in pixel coordinates.
        search_markers: The positions of the markers in the search frame in pixel coordinates.
        search_window_size: The size of the search window. Defaults to 40px.
        marker_template_size: The size of the marker template. Defaults to 20px.
        upscaling_factor: The upscaling factor for the marker template and search window (used to reduce noise in the cross-correlation). Defaults to 5.

    Returns:
        The new positions of the markers in the search frame.
    """

    current_markers = search_markers.copy()

    # Loop over the previous marker positions and find the corresponding marker in the current frame
    for (i, template_marker), search_marker in zip(enumerate(template_markers), search_markers):
        # Marker position in the previous frame
        x, y = template_marker
        x_search, y_search = search_marker

        # Define marker template centered on the previous position
        marker_template = template_frame[
            int(max(y - marker_template_size/2, 0)):int(min(y + marker_template_size/2, template_frame.shape[0])),
            int(max(x - marker_template_size/2, 0)):int(min(x + marker_template_size/2, template_frame.shape[1]))
        ]
        # Define the search window centered on the previous marker position
        search_window = search_frame[
            int(max(y_search - search_window_size/2, 0)):int(min(y_search + search_window_size/2, search_frame.shape[0])),
            int(max(x_search - search_window_size/2, 0)):int(min(x_search + search_window_size/2, search_frame.shape[1]))
        ]

        # Upscale the marker template and search window to reduce noise in the template matching
        try:
            marker_template = cv2.resize(
                marker_template,
                (int(marker_template.shape[0]*upscaling_factor), int(marker_template.shape[1]*upscaling_factor)),
                interpolation=cv2.INTER_CUBIC
            )
        except:
            raise Exception(
                f"Marker template is {marker_template.shape[0]}x{marker_template.shape[1]}px. Marker at position {template_marker} could not be found.")
        try:
            search_window = cv2.resize(
                search_window,
                (int(search_window.shape[0]*upscaling_factor), int(search_window.shape[1]*upscaling_factor)),
                interpolation=cv2.INTER_CUBIC
            )
        except:
            raise Exception(
                f"Search window is {search_window.shape[0]}x{search_window.shape[1]}px. Marker at position {template_marker} could not be found.")
        # Compute the cross-correlation between the marker template and the search window
        # Catch exception if the template is larger than the search window
        try:
            xcorr_result = cv2.matchTemplate(search_window, marker_template, cv2.TM_CCORR_NORMED)
        except:
            raise Exception(
                f"Marker template is {marker_template.shape[0]}x{marker_template.shape[1]}px. Search window is {search_window.shape[0]}x{search_window.shape[1]}px. Marker at position {template_marker} could not be found."
            )

        # Get the position of the marker in the current frame
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(xcorr_result)
        current_markers[i] = np.array([
            x_search + (marker_template.shape[0]/2 - search_window.shape[0]/2 + max_loc[0])/upscaling_factor,
            y_search + (marker_template.shape[1]/2 - search_window.shape[1]/2 + max_loc[1])/upscaling_factor
        ])

    return current_markers


def polygon_area(vertices: np.ndarray):
    """Computes area of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (np.ndarray): array of shape (n_vertices, 2).

    Returns:
        float: Area of the polygon.
    """

    v1 = np.roll(vertices, shift=1, axis=0)
    v2 = vertices

    return np.abs(np.sum(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2)


def polygon_centroid(vertices: np.ndarray):
    """Computes centroid of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (np.ndarray): array of shape (n_vertices, 2).

    Returns:
        np.ndarray: Centroid of the polygon.
    """

    area = polygon_area(vertices)
    v1 = np.roll(vertices, shift=1, axis=0)
    v2 = vertices
    x_plus_y = v1 + v2
    v_cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    return np.array([
        np.sum(x_plus_y[:, 0] * v_cross),
        np.sum(x_plus_y[:, 1] * v_cross)
    ]) / (6 * area)


def compute_edge_lengths(vertices: np.ndarray):
    """Computes edge lengths of the polygon.

    Args:
        vertices (np.ndarray): array of shape (n_vertices, 2).

    Returns:
        np.ndarray: Edge lengths of the polygon.
    """

    return np.linalg.norm(
        np.roll(vertices, 1, axis=0) - vertices,
        axis=-1
    )


def angle_between_unit_vectors(u1, u2):
    """Computes the signed angle between two unit vectors using arctan2.

    Args:
        u1 (np.ndarray): array of shape (2, ) defining the first unit vector.
        u2 (np.ndarray): array of shape (2, ) defining the second unit vector.

    Returns:
        float: Signed angle measured from u1 to u2 (positive counter-clockwise). Result is in the range [-pi, pi].
    """
    return np.arctan2(u1[0] * u2[1] - u1[1] * u2[0], u1[0] * u2[0] + u1[1] * u2[1])


def compute_block_displacement_from_markers(previous_markers, current_markers, calib_xy=(1, 1), max_angle_change=max_angle_change_default, max_displacement_change=None):
    """Compute the displacement of the block from the marker positions.

    Args:
        previous_markers: The positions of the markers in the previous frame in pixel coordinates.
        current_markers: The positions of the markers in the current frame in pixel coordinates.
        calib_xy: The calibration factor for the x and y coordinates in mm/pixel.

    Returns:
        The displacement of the block in the current frame in physical coordinates x [mm], y [mm], theta [rad].
    """

    # Compute xy displacement of the centroid
    previous_centroid = polygon_centroid(previous_markers)
    current_centroid = polygon_centroid(current_markers)
    xy_displacement = (current_centroid - previous_centroid) * np.array(calib_xy)

    # Compute angle change using the longest edge
    previous_edges = np.roll(previous_markers, 1, axis=0) - previous_markers
    current_edges = np.roll(current_markers, 1, axis=0) - current_markers
    previous_edge_lengths = compute_edge_lengths(previous_markers)
    current_edge_lengths = compute_edge_lengths(current_markers)
    # NOTE: The angle change is weighted by the length of the edge. This makes rotation tracking more robust to noise in the marker positions.
    theta_displacement = angle_between_unit_vectors(
        previous_edges.T/previous_edge_lengths,
        current_edges.T/current_edge_lengths
    ) @ (current_edge_lengths/current_edge_lengths.sum())

    block_displacement = np.array([xy_displacement[0], xy_displacement[1], theta_displacement])

    if max_displacement_change is None:
        _max_displacement_change = np.inf
    else:
        _max_displacement_change = max_displacement_change

    if np.linalg.norm(block_displacement[:2]) > _max_displacement_change:
        # NOTE: If the displacement is too large for this contour, it is set to NaN. NaNs are going to be replaced by interpolated values.
        # This is to avoid jumps in the tracking caused by a wrong contour being matched or a contour being lost for a frame.
        block_displacement[0] = np.nan
        block_displacement[1] = np.nan
        block_displacement[2] = np.nan
    else:
        # Make sure that the angle change is not too large
        block_displacement[2] = np.sign(block_displacement[2]) * \
            min(np.abs(block_displacement[2]), max_angle_change*np.pi/180)

    return block_displacement


def smooth_fields_convolution(fields: jnp.ndarray, kernel_size=3):
    """Smooth the fields using a convolution with a kernel of size `kernel_size`."""
    if type(kernel_size) == int:
        kernel_size = [[kernel_size, kernel_size, kernel_size], [kernel_size, kernel_size, kernel_size]]
    elif type(kernel_size) == list or type(kernel_size) == tuple:
        if len(kernel_size) == 3:
            kernel_size = [kernel_size, kernel_size]
        elif len(kernel_size) == 6:
            kernel_size = [kernel_size[:3], kernel_size[3:]]

    new_fields = jnp.array(fields)
    if kernel_size != 0:
        n_blocks = fields.shape[2]
        for i, sizes in enumerate(kernel_size):
            for j, size in enumerate(sizes):
                if size != 0:
                    new_fields = new_fields.at[:, i, :, j].set(
                        vmap(
                            lambda b: jnp.convolve(
                                new_fields[:, i, b, j],
                                jnp.ones(size) / size,
                                mode="same",
                            ),
                            in_axes=0,
                        )(jnp.arange(n_blocks)).T
                    )

    return new_fields


def smooth_fields_SG(fields: jnp.ndarray, window_length=3, polyorder=1):
    """Smooth the fields using a Savitzky-Golay filter."""

    # Sanitize window length
    if type(window_length) == int:
        window_length = [[window_length]*3, [window_length]*3]
    elif type(window_length) == list or type(window_length) == tuple:
        if len(window_length) == 3:
            window_length = [window_length, window_length]
        elif len(window_length) == 6:
            window_length = [window_length[:3], window_length[3:]]

    # Sanitize polyorder
    if type(polyorder) == int:
        polyorder = [[polyorder]*3, [polyorder]*3]
    elif type(polyorder) == list or type(polyorder) == tuple:
        if len(polyorder) == 3:
            polyorder = [polyorder, polyorder]
        elif len(polyorder) == 6:
            window_length = [polyorder[:3], polyorder[3:]]

    new_fields = jnp.array(fields)
    if window_length != 0:
        for (i, sizes), orders in zip(enumerate(window_length), polyorder):
            for (j, size), order in zip(enumerate(sizes), orders):
                if size != 0:
                    new_fields = new_fields.at[:, i, :, j].set(
                        savgol_filter(
                            new_fields[:, i, :, j],
                            size,
                            order,
                            axis=0
                        )
                    )

    return new_fields
