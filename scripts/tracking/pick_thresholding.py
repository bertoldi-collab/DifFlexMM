"""
    This script can be used to pick the threshold constant and the region of interest.
    Slide the trackbar until you find the ideal value for each parameter. 
"""

import cv2
import argparse


def pick_thresholding():

    global preprocessing_params, img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cv2.namedWindow("Trackbar")
    cv2.createTrackbar("thresholding", "Trackbar", 0, 255, thresh_change)
    if preprocessing_params["adaptive_thresholding"]:
        cv2.createTrackbar("adaptive_thresholding_block", "Trackbar", 0, 500, block_change)
    preprocessing()

    while True:
        if cv2.waitKey(500) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit()


def thresh_change(new_val):
    change_params("thresholding", new_val)


def block_change(new_val):
    change_params("adaptive_thresholding_block", 2 * new_val + 1)


def change_params(name, value):
    global preprocessing_params
    preprocessing_params[name] = value
    if preprocessing_params["adaptive_thresholding"]:
        print(
            "Thresholding =",
            preprocessing_params["thresholding"],
            "| Adaptive Thresholding Block =",
            preprocessing_params["adaptive_thresholding_block"],
        )
    else:
        print(
            "Thresholding =",
            preprocessing_params["thresholding"],
        )
    preprocessing()


def preprocessing():

    median = cv2.medianBlur(img, 7)

    if preprocessing_params["adaptive_thresholding"]:
        thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, preprocessing_params["adaptive_thresholding_block"], preprocessing_params["thresholding"])
    else:
        _, thresh = cv2.threshold(median, preprocessing_params["thresholding"], 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    width = opening.shape[1] // 2
    height = opening.shape[0] // 2
    dim = (width, height)

    # resize image
    resized = cv2.resize(opening, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("", resized)


if __name__ == "__main__":

    preprocessing_params = {"thresholding": 50, "adaptive_thresholding": False, "adaptive_thresholding_block": 11}

    ap = argparse.ArgumentParser()

    ap.add_argument("-v", "--video", default="", help="Video to use for preprocessing")
    ap.add_argument("-a", "--adaptive", action="store_true",
                    default=preprocessing_params["adaptive_thresholding"], help="Use adaptive thresholding")

    args = ap.parse_args()
    preprocessing_params["adaptive_thresholding"] = args.adaptive

    vidcap = cv2.VideoCapture(args.video)
    _, img = vidcap.read()

    # select ROI function
    showCrosshair = False
    fromCenter = False

    cv2.namedWindow("First frame", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("First frame", img, showCrosshair, fromCenter)

    # print rectangle points of selected ROI
    print("ROI_Y = [", int(roi[1]), int(roi[1] + roi[3]), "] | ROI_X = [", int(roi[0]), int(roi[0] + roi[2]), "]")

    # Crop selected ROI from raw image
    img = img[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]

    pick_thresholding()
