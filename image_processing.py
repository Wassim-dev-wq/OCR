import cv2
import numpy as np

def cv2_imshow(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Check if image is loaded
    if image is None:
        raise ValueError(f"Image file not found or couldn't be opened: {image_path}")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2_imshow("Gray Image", gray_image)

    # Apply thresholding
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    thresh_image_inv = 255 - thresh_image

    return thresh_image_inv

def get_kernel_lengths(img_height, img_width):
    kernel_len_ver = max(10, img_height // 30)
    kernel_len_hor = max(10, img_width // 30)

    return kernel_len_ver, kernel_len_hor


def detect_lines(img_bin_inv, kernel_len_ver, kernel_len_hor):
    # Create kernels for vertical and horizontal lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Erode and dilate the image to get vertical lines
    image_1 = cv2.erode(img_bin_inv, vertical_kernel,iterations=2)
    vertical_lines = cv2.dilate(image_1, vertical_kernel, iterations=4)

    # Erode and dilate the image to get horizontal lines
    image_2 = cv2.erode(img_bin_inv, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(image_2, horizontal_kernel, iterations=5)

    # Combine vertical and horizontal lines
    img_vh = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 0.0)

    # Dilate the combined lines and apply thresholding
    img_vh = cv2.dilate(img_vh, kernel, iterations=2)
    _, img_vh = cv2.threshold(img_vh, 60, 255, cv2.THRESH_BINARY)

    return img_vh


def combine_images(img_bin, img_vh):
    bitor = cv2.bitwise_or(img_bin, img_vh)
    img_median = bitor

    return img_median

def erode_dilate_images(img_median, img_h, img_w):
    # Create vertical and horizontal kernels
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, img_h * 2))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_w * 2, 1))

    # Erode the image using vertical and horizontal kernels
    vertical_lines_img = cv2.erode(img_median, vertical_kernel, iterations=1)
    horizontal_lines_img = cv2.erode(img_median, horizontal_kernel, iterations=1)
    # Combine vertical and horizontal lines
    combined_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Erode the combined lines and apply thresholding
    img_combined_lines = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    img_combined_lines_inverted = cv2.bitwise_not(img_combined_lines)
    img_combined_lines = cv2.erode(img_combined_lines_inverted, combined_kernel, iterations=2)

    _, img_combined_lines = cv2.threshold(img_combined_lines, 128, 255, cv2.THRESH_BINARY)

    # Use bitwise XOR and NOT operations to get the final result
    bitwise_xor_result = cv2.bitwise_xor(img_median, img_combined_lines)
    bitwise_not_result = cv2.bitwise_not(bitwise_xor_result)

    return bitwise_not_result

# https://cvexplained.wordpress.com/2020/06/06/sorting-contours/
# Sorting boxes
'''
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
  
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
  
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
  
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
  
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
'''


def get_contours_and_boxes(img_vh):
    contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours