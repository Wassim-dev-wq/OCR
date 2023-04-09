import cv2
import numpy as np
from image_processing import (get_kernel_lengths, detect_lines,
                                    combine_images, erode_dilate_images, get_contours_and_boxes,cv2_imshow,
                                    get_boxes,sort_boxes_by_row,arrange_boxes_in_order)

def recognize_structure(img):
    img_height, img_width = img.shape[:2]

    # Use the preprocessed image directly
    img_bin_inv = img    

    # Get kernel length for vertical and horizontal lines
    kernel_len_ver, kernel_len_hor = get_kernel_lengths(img_height, img_width)

    # Detect lines in the image
    img_vh = detect_lines(img_bin_inv, kernel_len_ver, kernel_len_hor)
    cv2_imshow("Detected Lines", img_vh)

    # Combine original image with detected lines
    img_median = combine_images(img_bin_inv, img_vh)
    cv2_imshow("Combined Images", img_median)

    # Erode and dilate the image for better visibility of table structure
    erode_dilate_img = erode_dilate_images(img_median, img_height, img_width)
    cv2_imshow("Eroded and Dilated Images", erode_dilate_img)

    # Get contours and bounding boxes of detected lines
    contours,bounding_boxes = get_contours_and_boxes(img_vh)
    heights = [bounding_boxes[i][3] for i in range(len(bounding_boxes))]

    # Get mean of heights
    avg_height = np.mean(heights)

    box = get_boxes(contours,img_width,img_height)
    row = sort_boxes_by_row(box, avg_height)
    final_boxes = arrange_boxes_in_order(row)

    return erode_dilate_img
    #return final_boxes, erode_dilate_img



