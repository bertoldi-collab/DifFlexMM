"""
    This is the main script to track the deformations and rotations of the blocks during the experiment using a grayscale image and a xcorr-based strategy.
    To start, please specify all required parameters (see '--help' for more information).
    While most of the parameters are straightforward, some need to be identified using other scripts.
    However, this only needs to be done once for each batch of experiments.
    The script will return a SolutionData file of the various DOFs over time as well as an animation thereof.
"""

import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from difflexmm.geometry import compute_xy_limits
from difflexmm.plotting import generate_animation
from difflexmm.utils import SolutionData, load_data, save_data
from scripts.tracking.utils import (collect_as, compute_block_displacement_from_markers,
                                    compute_centroid, find_markers, fit_contour, interpolate_nans, morphological_transformation_default, smooth_fields_convolution,
                                    sort_contours, max_angle_change_default, conv_size_default, step_size_default, adaptive_thresholding_block_default, aspect_ratio_threshold_default, search_window_size_default, marker_template_size_default, upscaling_factor_default)


def preprocessing(img, blur_size, threshold, adaptive_thresholding=False, adaptive_thresholding_block=adaptive_thresholding_block_default, morphological_transformation=morphological_transformation_default, inverted_gray=False):

    median = cv2.medianBlur(img, blur_size)
    thresh_id = cv2.THRESH_BINARY if inverted_gray else cv2.THRESH_BINARY_INV
    if adaptive_thresholding:
        thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresh_id, adaptive_thresholding_block, threshold)
    else:
        _, thresh = cv2.threshold(median, threshold, 255, thresh_id)

    transformed = morphological_transformation(thresh)

    return transformed


def get_contours(img, ROI_XY, blur_size, threshold, block_area, adaptive_thresholding=False, adaptive_thresholding_block=adaptive_thresholding_block_default, morphological_transformation=morphological_transformation_default, inverted_gray=False):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_ROI = img[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]]

    thresh = preprocessing(img_ROI, blur_size, threshold, adaptive_thresholding=adaptive_thresholding,
                           adaptive_thresholding_block=adaptive_thresholding_block, morphological_transformation=morphological_transformation, inverted_gray=inverted_gray)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnts, _ = cv2.findContours(thresh.astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)

    cnts_blocks = []

    for c in cnts:
        a = cv2.contourArea(c)
        if a > block_area[0] and a < block_area[1]:
            cnts_blocks.append(c)  # get_blob(img_ROI.shape, c))

    return cnts_blocks


def mark_reference_frame(
        video_path,
        calib_xy,
        ROI_X,
        ROI_Y,
        blur_size,
        threshold,
        block_area,
        reference_centroids=None,
        reference_shapes=None,
        adaptive_thresholding=False,
        adaptive_thresholding_block=adaptive_thresholding_block_default,
        aspect_ratio_threshold=aspect_ratio_threshold_default,
        morphological_transformation=morphological_transformation_default,
        markers_scaled_position=1.,
        frame=0,
        inverted_gray=False,
        masked_areas=[],
        show=False,):
    """Place markers based on the reference geometry if provided.

    Args:
        video_path (str): Video to be processed.
        calib_xy (tuple[float, float]): Calibration constants [mm/px]
        ROI_X (tuple[int, int]): Horizontal ROI boundaries.
        ROI_Y (tuple[int, int]): Vertical ROI boundaries.
        blur_size (int): Kernel size for blurring.
        threshold (int): Threshold constant.
        block_area (tuple[int, int]): Minimum and maximum block contour area.
        reference_centroids (np.ndarray): Reference centroids.
        reference_shapes (np.ndarray): Reference shapes (e.g. centroid_node_vectors).
        adaptive_thresholding (bool): Whether to use adaptive thresholding. Default is False.
        adaptive_thresholding_block (int): Block size for adaptive thresholding. Default is 11.
        aspect_ratio_threshold (float): Threshold for aspect ratio selecting the fitting method for contours (Below threshold minAreaRect is used, above fitEllipse). Default is 0.3.
        morphological_transformation (function): Morphological transformation to be applied to the thresholded image. Default is cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2).
        markers_scaled_position (float): Scaling factor for the marker positions. Default is 1.
        frame (int): Frame to be shown. Default is 0.
        inverted_gray (bool): Whether the image is inverted. Default is False (i.e. tracking black objects).
        masked_areas (list): List of masked areas. Default is []. Each area is defines as ((x0, x1), (y0, y1)
        show (bool): Whether to show the image. Default is False.

    Returns:
        np.ndarray: Marker positions.
        np.ndarray: Centroid positions.
    """

    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, image = video_capture.read()
    # Flip y-axis in image to match physical frame.
    image = cv2.flip(image, 0)
    flipped_ROI_Y = (image.shape[0] - ROI_Y[1], image.shape[0] - ROI_Y[0])
    ROI_XY = [ROI_X, flipped_ROI_Y]
    # Set masked areas to white within the flipped ROI
    for masked_area in masked_areas:
        image[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]][masked_area[1][0]: masked_area[1][1], masked_area[0][0]: masked_area[0][1]] = 255
    # Get contours
    cnts = get_contours(image, ROI_XY, blur_size, threshold, block_area,
                        adaptive_thresholding=adaptive_thresholding, adaptive_thresholding_block=adaptive_thresholding_block, morphological_transformation=morphological_transformation, inverted_gray=inverted_gray)
    # Sort contours based on the reference geometry if provided.
    if reference_centroids is not None:
        cnts = sort_contours(cnts, reference_centroids, calib_xy)

    n_blocks = len(cnts)
    rect_prev = np.zeros((n_blocks, 5))  # axis: x, y, angle, block_id, fitting_method
    rect_prev[:, 3] = np.arange(n_blocks)  # block_id
    # Markers used for cross-correlation between frames
    marker_points = np.zeros((n_blocks, 4, 2)) if reference_shapes is None else np.zeros_like(reference_shapes)

    for i, c in enumerate(cnts):
        cX, cY = compute_centroid(c)
        rect_prev[i, :2] = cX, cY  # centroid coordinates
        # method is automatically selected based on aspect ratio
        fitted_contour, method = fit_contour(c, method=None, aspect_ratio_threshold=aspect_ratio_threshold)
        rect_prev[i, 2] = fitted_contour[-1]  # angle
        rect_prev[i, 4] = method  # fitting method
        corners = np.int0(cv2.boxPoints(fitted_contour))
        marker_points[i, :, :] = corners if reference_shapes is None else rect_prev[i, :2] + \
            reference_shapes[i, :, :] / np.array(calib_xy) * markers_scaled_position

    if show:
        # Show thresholded image with contours using matplotlib
        img = image[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]]
        cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
        plt.figure(figsize=(8, 5), constrained_layout=True)
        plt.imshow(img, origin='lower',)
        # Plot the markers used for cross-correlation
        plt.scatter(marker_points[:, :, 0], marker_points[:, :, 1], c='g', s=10, marker='x')
        plt.show()

    return marker_points, rect_prev[:, :2].copy()  # marker_points, centroids


def tracking(
        video_path,
        calib_xy,
        start_end_video,
        ROI_X,
        ROI_Y,
        blur_size,
        threshold,
        framerate,
        block_area,
        reference_centroids=None,
        reference_shapes=None,
        masked_areas=[],
        max_angle_change=max_angle_change_default,
        max_displacement_change=None,
        conv_size=conv_size_default,
        step_size=step_size_default,
        adaptive_thresholding=False,
        adaptive_thresholding_block=adaptive_thresholding_block_default,
        aspect_ratio_threshold=aspect_ratio_threshold_default,
        morphological_transformation=morphological_transformation_default,
        inverted_gray=False,
        # Parameters for cross-correlation
        search_window_size=search_window_size_default,
        marker_template_size=marker_template_size_default,
        upscaling_factor=upscaling_factor_default,
        markers_scaled_position=1.,
        template_update_rate=0,
        search_window_update_rate=1,
        # Monitoring and visualization
        monitor_progress=True,
        show_tracked_frame=False,):
    """Returns tracked block info in the form of a SolutionData object.

    Args:
        video_path (str): Video to be processed.
        calib_xy (tuple[float, float]): Calibration constants [mm/px]
        start_end_video (tuple[int, int]): Start and end frame of video.
        ROI_X (tuple[int, int]): Horizontal ROI boundaries.
        ROI_Y (tuple[int, int]): Vertical ROI boundaries.
        blur_size (int): Kernel size for blurring.
        threshold (int): Threshold constant.
        framerate (int): Framerate of the video.
        block_area (tuple[int, int]): Minimum and maximum block contour area.
        reference_centroids (np.ndarray): Reference centroids.
        reference_shapes (np.ndarray): Reference shapes (e.g. centroid_node_vectors).
        masked_areas (list): List of masked areas. Default is []. Each area is defines as ((x0, x1), (y0, y1)
        max_angle_change (float): Maximum angle change (in deg) between frames. Default is 360°.
        max_displacement_change (float): Maximum displacement change (in mm) between frames. Default is None (i.e. no limit).
        conv_size ([[int,int,int],[int,int,int]]): Size of convolution kernel to smooth fields [[x, y, theta], [x_dot, y_dot, theta_dot]]. Default is [[0,0,0],[0,0,0]] (i.e. no convolution).
        step_size (int): Step size between frames for tracking. Default is 1.
        adaptive_thresholding (bool): Whether to use adaptive thresholding. Default is False.
        adaptive_thresholding_block (int): Block size for adaptive thresholding. Default is 11.
        aspect_ratio_threshold (float): Threshold for aspect ratio selecting the fitting method for contours (Below threshold minAreaRect is used, above fitEllipse). Default is 0.3.
        morphological_transformation (function): Morphological transformation to be applied to the thresholded image. Default is cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2).
        inverted_gray (bool): Whether the image is inverted. Default is False (i.e. tracking black objects).
        search_window_size (int): Size of the search window for cross-correlation. Default is 40px.
        marker_template_size (int): Size of the marker template for cross-correlation. Default is 20px.
        upscaling_factor (int): Upscaling factor for cross-correlation. Default is 5.
        markers_scaled_position (float): Position of the markers relative to the nominal position. Default is 1.0.
        template_update_rate (int): Update rate for the reference frame (in number of frames). Default is 0 (i.e. no update).
        search_window_update_rate (int): Update rate for the search window (in number of frames). Default is 1 (i.e. update every frame).
        monitor_progress (bool): Whether to print progress. Default is True.
        show_tracked_frame (bool): Whether to show the tracked frame. Default is False.

    Returns:
        SolutionData: NamedTuple containing information of the tracked blocks.
    """

    video_capture = cv2.VideoCapture(video_path)

    startVideo, endVideo = start_end_video

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, startVideo)
    if endVideo == -1:
        endVideo = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Markers used for cross-correlation between frames
    marked_blocks, block_centroids = mark_reference_frame(
        video_path,
        calib_xy,
        ROI_X,
        ROI_Y,
        blur_size,
        threshold,
        block_area,
        reference_centroids=reference_centroids,
        reference_shapes=reference_shapes,
        adaptive_thresholding=adaptive_thresholding,
        adaptive_thresholding_block=adaptive_thresholding_block,
        aspect_ratio_threshold=aspect_ratio_threshold,
        morphological_transformation=morphological_transformation,
        markers_scaled_position=markers_scaled_position,
        frame=startVideo,
        inverted_gray=inverted_gray,
        masked_areas=masked_areas,
        show=False,
    )
    n_blocks = len(block_centroids)
    solution = np.zeros(((endVideo - startVideo) // step_size + 1, 2, n_blocks, 3))
    block_centroids = block_centroids * calib_xy
    block_displacement = np.zeros((n_blocks, 3))
    block_displacement_0_nans = np.zeros((n_blocks, 3))
    count = startVideo

    _, image = video_capture.read()
    # Flip y-axis in image to match physical frame.
    image = cv2.flip(image, 0)
    flipped_ROI_Y = (image.shape[0] - ROI_Y[1], image.shape[0] - ROI_Y[0])
    ROI_XY = [ROI_X, flipped_ROI_Y]
    image = image[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]]
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_frame = gray_frame.copy()
    current_frame = gray_frame.copy()

    template_markers_blocks = np.array(marked_blocks).astype(np.float64)
    search_markers_blocks = template_markers_blocks.copy()
    current_markers_blocks = template_markers_blocks.copy()

    while video_capture.isOpened():

        success, image = video_capture.read()
        # Flip y axis in image
        image = cv2.flip(image, 0)
        image = image[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]]
        current_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if count > endVideo:
            break

        if (count - startVideo) % step_size != 0:
            count += 1
            continue

        if success:

            if monitor_progress:
                print("#Frame_" + str(count))

            block_displacement_i = np.zeros((n_blocks, 3))
            # TODO: (Refactor) It would better to have just a list of markers and then reconstract the blocks after tracking.
            # This requires computing the displacements only after the tracking is done.
            # The best way would be reuse the track_points function from the tracking_points.py script.
            for block_id, template_markers in enumerate(template_markers_blocks):
                marker_points_current = find_markers(
                    template_frame,
                    current_frame,
                    # Used for extracting templates around the markers in the template frame
                    template_markers,
                    # Used for placing the search window around the markers in the current frame
                    search_markers_blocks[block_id],
                    search_window_size=search_window_size,
                    marker_template_size=marker_template_size,
                    upscaling_factor=upscaling_factor
                )
                block_displacement_i[block_id] = compute_block_displacement_from_markers(
                    current_markers_blocks[block_id],
                    marker_points_current,
                    calib_xy,
                    max_angle_change=max_angle_change,
                    max_displacement_change=max_displacement_change,
                )
                # Update the marker points for displacement calculation
                current_markers_blocks[block_id] = marker_points_current

            if np.any(np.isnan(block_displacement_i)) and monitor_progress:
                print("Warning: NaNs in displacement at frame " + str(count))
            rect_velocity_i = block_displacement_i * framerate / step_size
            block_displacement_0_nans += np.nan_to_num(block_displacement_i)
            block_displacement = block_displacement_0_nans + block_displacement_i

            solution[(count - startVideo) // step_size, 0, :, :] = block_displacement
            solution[(count - startVideo) // step_size, 1, :, :] = rect_velocity_i

            count += 1

            if show_tracked_frame:
                # Draw the markers on a copy of the current frame and show it
                current_frame_markers = image.copy()
                for marker_points in current_markers_blocks:
                    for marker in marker_points:
                        cv2.drawMarker(current_frame_markers, tuple(marker.astype(int)),
                                       (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
                # Show the frame and wait for key press
                cv2.imshow("Tracking", current_frame_markers)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break

        # Update the reference frame and reference markers
        if template_update_rate != 0 and video_capture.get(cv2.CAP_PROP_POS_FRAMES) % template_update_rate == 0:
            template_frame = current_frame.copy()
            template_markers_blocks = current_markers_blocks.copy()

        # Update the search window
        if search_window_update_rate != 0 and video_capture.get(cv2.CAP_PROP_POS_FRAMES) % search_window_update_rate == 0:
            search_markers_blocks = current_markers_blocks.copy()

    # Replace NaNs with interpolated values.
    solution = interpolate_nans(solution)

    # Smooth fields
    solution = smooth_fields_convolution(solution, kernel_size=conv_size)

    timepoints = np.arange(startVideo, endVideo + 1, step_size) / framerate

    # Redefine origin using reference geometry if provided.
    if reference_centroids is not None:
        block_centroids += reference_centroids[0] - block_centroids[0]

    # Release video capture and destroy all windows
    video_capture.release()
    cv2.destroyAllWindows()

    return SolutionData(
        block_centroids=block_centroids,
        centroid_node_vectors=(
            marked_blocks*calib_xy - block_centroids[:, None, :]) if reference_shapes is None else reference_shapes,
        bond_connectivity=None,
        timepoints=timepoints,
        fields=solution,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v", "--video_path", help="Indicate video to analyse", type=str, required=True
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        help="Define directory for saving resulting files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-cal",
        "--calib_xy",
        help="Calibration constants [mm/px]",
        type=float,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )
    parser.add_argument(
        "-se",
        "--start_end_video",
        help="Define desired start and end frame of video",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )
    parser.add_argument(
        "-x",
        "--ROI_X",
        help="Define horizontal ROI boundaries",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )
    parser.add_argument(
        "-y",
        "--ROI_Y",
        help="Define vertical ROI boundaries",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )
    parser.add_argument(
        "-b",
        "--blur_size",
        help="Define kernel size for blurring",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Define thresholding constant",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-a",
        "--adaptive",
        action="store_true",
        help="Use adaptive thresholding",
        default=False,
    )
    parser.add_argument(
        "-atb",
        "--adaptive_thresholding_block",
        type=int,
        help="Use adaptive thresholding with given block size",
        default=adaptive_thresholding_block_default,
    )
    parser.add_argument(
        "-f",
        "--framerate",
        help="Indicate framerate of the camera [frames/s]",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-ba",
        "--block_area",
        help="Define minimum and maximum block contour area",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )
    parser.add_argument(
        "-mac",
        "--max_angle_change",
        help="Maximum allowed change in angle between consecutive frames [optional]",
        type=int,
        default=max_angle_change_default,
    )
    parser.add_argument(
        "-mdc",
        "--max_displacement_change",
        help="Maximum allowed change in displacement between consecutive frames [optional]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-art",
        "--aspect_ratio_threshold",
        help="Threshold for aspect ratio selecting the fitting method for contours (Below threshold minAreaRect is used, above fitEllipse). [optional]",
        type=float,
        default=aspect_ratio_threshold_default,
    )
    parser.add_argument(
        "-cs",
        "--conv_size",
        help="Size of convolution kernel used to smooth fields [optional]",
        type=int,
        default=conv_size_default,
        action=collect_as(tuple),
    )
    parser.add_argument(
        "-ref",
        "--reference_data_path",
        help="File containing reference geometry information [optional]",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-ss",
        "--step_size",
        help="Step size for video analysis [optional]",
        type=int,
        default=step_size_default,
    )
    parser.add_argument(
        "-msp",
        "--markers_scaled_position",
        help="Position of the markers relative to the nominal position [optional]",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-w",
        "--search_window_size",
        help="Size of the search window for cross-correlation [optional]",
        type=int,
        default=search_window_size_default,
    )
    parser.add_argument(
        "-mt",
        "--marker_template_size",
        help="Size of the marker template for cross-correlation [optional]",
        type=int,
        default=marker_template_size_default,
    )
    parser.add_argument(
        "-uf",
        "--upscaling_factor",
        help="Upscaling factor for cross-correlation [optional]",
        type=int,
        default=upscaling_factor_default,
    )
    parser.add_argument(
        "-tr",
        "--template_update_rate",
        help="Update rate for the reference frame (in number of frames) [optional]",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-wr",
        "--search_window_update_rate",
        help="Update rate for the search window (in number of frames) [optional]",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-m",
        "--monitor_progress",
        help="Flag to monitor progress [optional]",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-ht",
        "--hide_tracked_frame",
        help="Flag to hide the tracked frame [optional]",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    # Extract reference data if provided
    if args.reference_data_path is not None:
        reference_data = load_data(args.reference_data_path)
    else:
        reference_data = None

    solution_data = tracking(
        video_path=args.video_path,
        calib_xy=args.calib_xy,
        start_end_video=args.start_end_video,
        ROI_X=args.ROI_X,
        ROI_Y=args.ROI_Y,
        blur_size=args.blur_size,
        threshold=args.threshold,
        framerate=args.framerate,
        block_area=args.block_area,
        reference_centroids=reference_data.block_centroids if reference_data is not None else None,
        reference_shapes=reference_data.centroid_node_vectors if reference_data is not None else None,
        max_angle_change=args.max_angle_change,
        max_displacement_change=args.max_displacement_change,
        conv_size=args.conv_size,
        step_size=args.step_size,
        adaptive_thresholding=args.adaptive,
        adaptive_thresholding_block=args.adaptive_thresholding_block,
        aspect_ratio_threshold=args.aspect_ratio_threshold,
        markers_scaled_position=args.markers_scaled_position,
        # Parameters for cross-correlation
        search_window_size=args.search_window_size,
        marker_template_size=args.marker_template_size,
        upscaling_factor=args.upscaling_factor,
        template_update_rate=args.template_update_rate,
        search_window_update_rate=args.search_window_update_rate,
        # Monitoring and visualization
        monitor_progress=args.monitor_progress,
        show_tracked_frame=not args.hide_tracked_frame,
    )
    # save solution data
    save_data(args.save_dir + "/tracking_data.pkl", solution_data)

    xylim = compute_xy_limits(solution_data.block_centroids)
    xylim = xylim + (np.linalg.norm(xylim[:, 1] - xylim[:, 0]) /
                     len(solution_data.block_centroids)**0.5) * np.array([-1, 1])
    generate_animation(
        data=solution_data,
        field="u",
        out_filename=args.save_dir + "/tracking_animation",
        deformed=True,
        xlim=xylim[0],
        ylim=xylim[1],
        dpi=300,
        figsize=(14, 8),
    )
