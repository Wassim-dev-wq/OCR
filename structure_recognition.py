import numpy as np
from image_processing import (get_kernel_lengths, get_iterations, detect_lines,
                                    combine_images, erode_dilate_images, get_contours_and_boxes,cv2_imshow, cv2_save,
                                    get_boxes,sort_boxes_by_row,arrange_boxes_in_order)

# The `recognize_structure` function takes an image as input and performs the following steps:
# 1. Gets the height and width of the image.
# 2. Detects vertical and horizontal lines in the image using a kernel length and number of
# iterations.
# 3. Combines the original image with the detected lines.
# 4. Erodes and dilates the image for better visibility of table structure.
# 5. Gets the contours and bounding boxes of the detected lines.
# 6. Calculates the average height of the bounding boxes.
# 7. Sorts the bounding boxes by row and arranges them in order.
# 8. Returns the final sorted and arranged bounding boxes and the eroded and dilated image.
def recognize_structure(img):
    img_height, img_width = img.shape[:2]

    # Use the preprocessed image directly
    img_bin_inv = img    

    # Get kernel length for vertical and horizontal lines
    kernel_len_ver, kernel_len_hor = get_kernel_lengths(img_height, img_width)
    iter_ver, iter_hor = get_iterations(img_height, img_width)


    # Detect lines in the image
    img_vh = detect_lines(img_bin_inv, kernel_len_ver, kernel_len_hor, iter_ver, iter_hor)
    cv2_save("Detected Lines", img_vh)

    # Combine original image with detected lines
    img_median = combine_images(img_bin_inv, img_vh)
    cv2_save("Combined Images", img_median)

    # Erode and dilate the image for better visibility of table structure
    erode_dilate_img = erode_dilate_images(img_median, img_height, img_width)
    cv2_save("Eroded and Dilated Images", erode_dilate_img)

    # Get contours and bounding boxes of detected lines
    contours,bounding_boxes = get_contours_and_boxes(img_vh)
    heights = [bounding_boxes[i][3] for i in range(len(bounding_boxes))]
    #cv2_imshow("Eroded and Dilated Images", erode_dilate_img)

    # Get mean of heights
    avg_height = np.mean(heights)

    box = get_boxes(contours,img_width,img_height)
    row = sort_boxes_by_row(box, avg_height)
    final_boxes = arrange_boxes_in_order(row)

    return final_boxes, erode_dilate_img



