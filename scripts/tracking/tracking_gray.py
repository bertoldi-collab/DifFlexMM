"""
    This is the main script to track the deformations and rotations of the blocks during the experiment using a grayscale image.
    To start, please specify all required parameters (see '--help' for more information).
    While most of the parameters are straightforward, some need to be identified using other scripts.
    However, this only needs to be done once for each batch of experiments.
    The script will return a SolutionData file of the various DOFs over time as well as an animation thereof.
"""

import argparse

import cv2
import numpy as np
from difflexmm.geometry import compute_xy_limits
from difflexmm.plotting import generate_animation
from difflexmm.utils import SolutionData, load_data, save_data
from scripts.tracking.utils import (calculate_displacement, collect_as,
                                    compute_centroid, fit_contour, interpolate_nans, morphological_transformation_default, smooth_fields_convolution,
                                    sort_contours, max_angle_change_default, conv_size_default, step_size_default, adaptive_thresholding_block_default, aspect_ratio_threshold_default)


def preprocessing(img, blur_size, threshold, adaptive_thresholding=False, adaptive_thresholding_block=adaptive_thresholding_block_default, morphological_transformation=morphological_transformation_default):

    median = cv2.medianBlur(img, blur_size)
    if adaptive_thresholding:
        thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, adaptive_thresholding_block, threshold)
    else:
        _, thresh = cv2.threshold(median, threshold, 255, cv2.THRESH_BINARY_INV)

    transformed = morphological_transformation(thresh)

    return transformed


def get_contours(img, ROI_XY, blur_size, threshold, block_area, adaptive_thresholding=False, adaptive_thresholding_block=adaptive_thresholding_block_default, morphological_transformation=morphological_transformation_default):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_ROI = img[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]]

    thresh = preprocessing(img_ROI, blur_size, threshold, adaptive_thresholding=adaptive_thresholding,
                           adaptive_thresholding_block=adaptive_thresholding_block, morphological_transformation=morphological_transformation)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnts, _ = cv2.findContours(thresh.astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)

    cnts_blocks = []

    for c in cnts:
        a = cv2.contourArea(c)
        if a > block_area[0] and a < block_area[1]:
            cnts_blocks.append(c)  # get_blob(img_ROI.shape, c))

    return cnts_blocks


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
        max_angle_change=max_angle_change_default,
        max_displacement_change=None,
        conv_size=conv_size_default,
        step_size=step_size_default,
        adaptive_thresholding=False,
        adaptive_thresholding_block=adaptive_thresholding_block_default,
        aspect_ratio_threshold=aspect_ratio_threshold_default,
        morphological_transformation=morphological_transformation_default,
        monitor_progress=True,):
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
        max_angle_change (float): Maximum angle change (in deg) between frames. Default is 360°.
        max_displacement_change (float): Maximum displacement change (in mm) between frames. Default is None (i.e. no limit).
        conv_size ([[int,int,int],[int,int,int]]): Size of convolution kernel to smooth fields [[x, y, theta], [x_dot, y_dot, theta_dot]]. Default is [[0,0,0],[0,0,0]] (i.e. no convolution).
        step_size (int): Step size between frames for tracking. Default is 1.
        adaptive_thresholding (bool): Whether to use adaptive thresholding. Default is False.
        adaptive_thresholding_block (int): Block size for adaptive thresholding. Default is 11.
        aspect_ratio_threshold (float): Threshold for aspect ratio selecting the fitting method for contours (Below threshold minAreaRect is used, above fitEllipse). Default is 0.3.
        morphological_transformation (function): Morphological transformation to be applied to the thresholded image. Default is cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2).
        monitor_progress (bool): Whether to print progress. Default is True.

    Returns:
        SolutionData: NamedTuple containing information of the tracked blocks.
    """

    video_capture = cv2.VideoCapture(video_path)

    startVideo, endVideo = start_end_video

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, startVideo)
    _, image = video_capture.read()
    # Flip y-axis in image to match physical frame.
    image = cv2.flip(image, 0)

    if endVideo == -1:
        endVideo = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    flipped_ROI_Y = (image.shape[0] - ROI_Y[1], image.shape[0] - ROI_Y[0])
    ROI_XY = [ROI_X, flipped_ROI_Y]

    cnts = get_contours(image, ROI_XY, blur_size, threshold, block_area,
                        adaptive_thresholding=adaptive_thresholding, adaptive_thresholding_block=adaptive_thresholding_block, morphological_transformation=morphological_transformation)
    # Sort contours based on the reference geometry if provided.
    if reference_centroids is not None:
        cnts = sort_contours(cnts, reference_centroids, calib_xy)

    n_blocks = len(cnts)
    solution = np.zeros(((endVideo - startVideo) // step_size + 1, 2, n_blocks, 3))
    centroid_node_vectors_box = np.zeros((n_blocks, 4, 2))

    rect_prev = np.zeros((n_blocks, 5))  # axis: x, y, angle, block_id, fitting_method
    rect_prev[:, 3] = np.arange(n_blocks)  # block_id

    for i, c in enumerate(cnts):

        cX, cY = compute_centroid(c)

        rect_prev[i, :2] = cX, cY  # centroid coordinates
        # method is automatically selected based on aspect ratio
        fitted_contour, method = fit_contour(c, method=None, aspect_ratio_threshold=aspect_ratio_threshold)
        rect_prev[i, 2] = fitted_contour[-1]  # angle
        rect_prev[i, 4] = method  # fitting method

        corners = np.int0(cv2.boxPoints(fitted_contour))
        centroid_node_vectors_box[i, :, :] = (corners - np.array([cX, cY])) * calib_xy

    block_centroids = (np.copy(rect_prev[:, :2])) * calib_xy

    rect_displacement = np.zeros((n_blocks, 3))
    rect_displacement_0_nans = np.zeros((n_blocks, 3))
    count = startVideo
    while video_capture.isOpened():

        success, image = video_capture.read()
        # Flip y axis in image
        image = cv2.flip(image, 0)

        if count > endVideo:
            break

        if (count - startVideo) % step_size != 0:
            count += 1
            continue

        if success:

            if monitor_progress:
                print("#Frame_" + str(count))

            contours_next = get_contours(image, ROI_XY, blur_size, threshold, block_area,
                                         adaptive_thresholding=adaptive_thresholding, adaptive_thresholding_block=adaptive_thresholding_block, morphological_transformation=morphological_transformation)
            rect_displacement_i = calculate_displacement(
                rect_prev, contours_next, n_blocks, calib_xy, max_angle_change, max_displacement_change, aspect_ratio_threshold=aspect_ratio_threshold
            )  # frame to frame displacements
            if np.any(np.isnan(rect_displacement_i)) and monitor_progress:
                print("Warning: NaNs in displacement at frame " + str(count))
            rect_velocity_i = rect_displacement_i * framerate / step_size
            rect_displacement_0_nans += np.nan_to_num(rect_displacement_i)
            rect_displacement = rect_displacement_0_nans + rect_displacement_i

            solution[(count - startVideo) // step_size, 0, :, :] = rect_displacement
            solution[(count - startVideo) // step_size, 1, :, :] = rect_velocity_i

            count += 1

        else:
            break

    # Replace NaNs with interpolated values.
    solution = interpolate_nans(solution)

    # Smooth fields
    solution = smooth_fields_convolution(solution, kernel_size=conv_size)

    timepoints = np.arange(startVideo, endVideo + 1, step_size) / framerate

    # Redefine origin using reference geometry if provided.
    if reference_centroids is not None:
        block_centroids += reference_centroids[0] - block_centroids[0]

    return SolutionData(
        block_centroids=block_centroids,
        centroid_node_vectors=centroid_node_vectors_box
        if reference_shapes is None
        else reference_shapes,
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
        "-m",
        "--monitor_progress",
        help="Flag to monitor progress [optional]",
        type=bool,
        default=True,
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
        monitor_progress=args.monitor_progress,
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
