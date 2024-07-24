"""
    Script for estimating the damping ratio from free bending oscillation experiments.
"""

import argparse

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scripts.tracking.tracking_gray import tracking
from scripts.tracking.utils import collect_as


def get_damping_ratio(
        video_path,
        calib_xy,
        start_end_video,
        ROI_Y,
        ROI_X,
        blur_size,
        threshold,
        framerate,
        block_area,
        monitor_progress=True,):
    """Analyze the video of free oscillations and return the damping ratio and angular frequency.

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

    Returns:
        tuple(float, float): Damping ratio and angular frequency.
    """

    solutionData = tracking(
        video_path=video_path,
        calib_xy=calib_xy,
        start_end_video=start_end_video,
        ROI_Y=ROI_Y,
        ROI_X=ROI_X,
        blur_size=blur_size,
        threshold=threshold,
        framerate=framerate,
        block_area=block_area,
        monitor_progress=monitor_progress,
    )

    angle = solutionData.fields[:, 0, 0, 2]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))

    angle_detrend = sc.signal.detrend(angle, type="constant")

    axs[0, 0].plot(1000 * np.arange(len(angle_detrend)) / framerate, angle_detrend)
    axs[0, 0].set(xlabel="Time [ms]", ylabel="Angle [rad]")

    peaks, _ = sc.signal.find_peaks(angle_detrend, height=0, distance=12)

    axs[0, 1].plot(
        1000 * np.arange(len(angle_detrend))[peaks] / framerate,
        angle_detrend[peaks],
        "o",
        alpha=0.5,
    )
    axs[0, 1].plot(
        1000 * np.arange(len(angle_detrend)) / framerate, angle_detrend, alpha=0.5
    )
    axs[0, 1].set(xlabel="Time [ms]", ylabel="Angle [rad]")

    y = np.log(angle_detrend[peaks])
    x = 1000 * np.arange(len(angle_detrend))[peaks] / framerate

    m, b = np.polyfit(x, y, 1)

    axs[1, 0].plot(x, y, "o")
    axs[1, 0].plot(x, m * x + b)
    axs[1, 0].set(xlabel="Time [ms]", ylabel="log(Angle)")

    Y = np.fft.rfft(angle_detrend)
    freq = np.fft.rfftfreq(angle_detrend.size, d=1.0 / framerate)

    axs[1, 1].plot(freq, abs(Y))
    axs[1, 1].set(xlabel="Frequency [Hz]", ylabel="Fourier amplitude")

    freq_d = freq[np.argmax(abs(Y))]
    omega_d = 2 * np.pi * freq_d
    zeta = 1 / np.sqrt(1 + (omega_d / (1000 * m)) ** 2)

    axs[0, 0].annotate((
        r"\begin{align*}"
        rf"\omega_d &= {{{omega_d:.2f}}}\,\text{{rad/s}} \\"
        rf"f_d &= {{{freq_d:.2f}}}\,\text{{Hz}} \\"
        rf"T_d &= {{{1000*freq_d**-1:.2f}}}\,\text{{ms}} \\"
        rf"\zeta &= {{{zeta:.4f}}}"
        r"\end{align*}"),
        xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95)
    )

    fig.tight_layout()
    plt.show()

    return zeta, omega_d


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v", "--video_path", help="Indicate video to analyse", type=str, required=True
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
        "-y",
        "--ROI_Y",
        help="Define vertical ROI boundaries",
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
        "-cs",
        "--conv_size",
        help="Size of convolution kernel used to smoothen angles [optional]",
        type=int,
    )
    parser.add_argument(
        "-m",
        "--monitor_progress",
        help="Flag to monitor progress [optional]",
        type=int,
        default=True,
    )

    args = parser.parse_args()

    get_damping_ratio(
        video_path=args.video_path,
        calib_xy=args.calib_xy,
        start_end_video=args.start_end_video,
        ROI_Y=args.ROI_Y,
        ROI_X=args.ROI_X,
        blur_size=args.blur_size,
        threshold=args.threshold,
        framerate=args.framerate,
        block_area=args.block_area,
        monitor_progress=args.monitor_progress,
    )
