"""
    This script can be used to pick the size of the blurring kernel and the appropriate contour area range using a grayscale image.
    To start, please specify the threshold and ROI that you identified in the previous script.
    Slide the trackbar until you find the ideal value for each parameter. 
    Ideally, you should pick up all relevant contours (i.e. the blocks), while ignoring any noise.
"""

import argparse

import cv2
import numpy as np
from scripts.tracking.utils import collect_as


def pick_preprocessing(ROI_X, ROI_Y):

    global preprocessing_params, img, img_gray

    ROI_Y_min, ROI_Y_max = ROI_Y
    ROI_X_min, ROI_X_max = ROI_X

    img = img[ROI_Y_min:ROI_Y_max, ROI_X_min:ROI_X_max]

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cv2.namedWindow("Trackbar")
    cv2.createTrackbar("blur_size", "Trackbar", 0, 20, blur_change)
    cv2.createTrackbar("area_min", "Trackbar", 0, 100, area_min_change)
    cv2.createTrackbar("area_max", "Trackbar", 0, 100, area_max_change)
    preprocessing()

    while True:
        if cv2.waitKey(500) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit()


def blur_change(new_val):
    change_params("blur_size", 2 * new_val + 1)


def area_min_change(new_val):
    change_params("area_min", 10 * new_val)


def area_max_change(new_val):
    change_params("area_max", 100 * new_val)


def change_params(name, value):
    global preprocessing_params
    preprocessing_params[name] = value
    print(
        "Blurring =",
        preprocessing_params["blur_size"],
        "| Minimum Area  =",
        preprocessing_params["area_min"],
        "| Maximum Area = ",
        preprocessing_params["area_max"],
    )
    preprocessing()


def preprocessing():

    median = cv2.medianBlur(img_gray, preprocessing_params["blur_size"])
    if preprocessing_params["adaptive_thresholding"]:
        thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, preprocessing_params["adaptive_thresholding_block"], t)
    else:
        _, thresh = cv2.threshold(median, t, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    copy = np.copy(img)

    for c in cnts:
        a = cv2.contourArea(c)
        if (
            a > preprocessing_params["area_min"]
            and a < preprocessing_params["area_max"]
        ):
            cv2.drawContours(copy, [c], 0, (255, 255, 255), 2)

    width = img.shape[1] // 2
    height = img.shape[0] // 2
    dim = (width, height)

    # resize image
    resized = cv2.resize(copy, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("", resized)


if __name__ == "__main__":

    preprocessing_params = {"blur_size": 7, "area_min": 0, "area_max": 0,
                            "adaptive_thresholding": False, "adaptive_thresholding_block": 11}

    ap = argparse.ArgumentParser()

    ap.add_argument("-v", "--video", default="", help="Video to use for preprocessing")

    ap.add_argument(
        "--ROI_Y",
        help="Define vertical ROI boundaries",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )

    ap.add_argument(
        "--ROI_X",
        help="Define horizontal ROI boundaries",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )

    ap.add_argument(
        "-t",
        "--threshold",
        help="Define thresholding constant",
        type=int,
        required=True,
    )

    ap.add_argument(
        "-a",
        "--adaptive",
        action="store_true",
        help="Use adaptive thresholding",
        default=preprocessing_params["adaptive_thresholding"],
    )

    ap.add_argument(
        "-atb",
        "--adaptive_thresholding_block",
        type=int,
        help="Use adaptive thresholding with given block size",
        default=preprocessing_params["adaptive_thresholding_block"],
    )

    args = ap.parse_args()
    preprocessing_params["adaptive_thresholding"] = args.adaptive
    preprocessing_params["adaptive_thresholding_block"] = args.adaptive_thresholding_block

    vidcap = cv2.VideoCapture(args.video)
    _, img = vidcap.read()

    t = args.threshold

    pick_preprocessing(args.ROI_X, args.ROI_Y)
