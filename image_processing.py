import cv2
import numpy as np


def cv2_imshow(title, img):
    cv2.imshow(title, img)
    cv2.imwrite(f"image_process/{title}.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cv2_save(title, img):
    cv2.imwrite(f"image_process/{title}.jpg", img)


# `preprocess_image` is a function that takes an image as input and performs preprocessing steps
# to prepare it for line detection. It first converts the image to grayscale and then applies
# adaptive thresholding using a Gaussian filter to obtain a binary image with white lines on a
# black background.
def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh_image

# `get_kernel_lengths` is a function that takes the height and width of an image as input and
# calculates the length of the vertical and horizontal kernels that will be used in the
# `detect_lines` function to detect vertical and horizontal lines in the image. The length of the
# kernels is calculated based on the ratio of the image width to height, and a minimum kernel size
# is set to avoid very small kernels. The function returns the length of the vertical and
# horizontal kernels as integers.
def get_kernel_lengths(img_height, img_width):
    size = img_width / img_height
    min_kernel_size = 10
    kernel_len_ver = max(min_kernel_size, int(img_height / 20 * (size / 2)))
    kernel_len_hor = max(min_kernel_size, int(img_width / 20 * (2 / size)))
    return kernel_len_ver, kernel_len_hor


def get_iterations(img_height, img_width):
    iter_ver = max(1, img_height // 300)  
    iter_hor = max(1, img_width // 300)   
    return iter_ver, iter_hor

# `remove_small_components` is a function that takes a binary image and a size parameter as input.
# It uses the `cv2.connectedComponentsWithStats` function to find connected components in the
# image and their corresponding statistics, including the size of each component. It then creates
# a new binary image and sets the pixels of the connected components whose size is greater than or
# equal to the size parameter to 255 (white), while setting the pixels of the other components to
# 0 (black). The resulting image contains only the connected components that are larger than the
# specified size.
def remove_small_components(img, size):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4) #con 4 
    sizes = stats[:, -1]
    img2 = np.zeros((output.shape), dtype=np.uint8)
    for i in range(1, nb_components):
        if sizes[i] >= size:
            img2[output == i] = 255
    return img2


# `detect_lines` is a function that takes a binary image as input and applies morphological
# operations to detect vertical and horizontal lines in the image. It first removes small
# components from the image, applies Gaussian blur, and then applies Canny edge detection to
# detect edges in the image. It then dilates the edges and applies erosion using vertical and
# horizontal kernels to detect vertical and horizontal lines respectively. Finally, it combines
# the detected vertical and horizontal lines and returns the resulting binary image.
def detect_lines(img_bin_inv, kernel_len_ver, kernel_len_hor, iter_ver, iter_hor):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bin_inv = remove_small_components(img_bin_inv, 100)

    # gaussian blur
    img_blur = cv2.GaussianBlur(img_bin_inv, (3, 3), 0)
    # threshholds
    med_val = np.median(img_blur)
    lower = int(max(0, 0.7 * int(med_val)))
    upper = int(min(255, 1.3 * int(med_val)))
    #Canny edge detection
    edges = cv2.Canny(img_blur, lower, upper)
    # Dilate the edges
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    # Detect lines on dilated edges
    img_eroded_vertical = cv2.erode(dilated_edges, vertical_kernel, iterations=iter_ver)
    cv2_save("vertical_lines_erodeOrig", img_eroded_vertical)    
    img_eroded_vertical = remove_small_components(img_eroded_vertical, img_bin_inv.shape[0]/6)

    vertical_lines = cv2.dilate(img_eroded_vertical, vertical_kernel, iterations=iter_ver)

    img_eroded_horizontal = cv2.erode(dilated_edges, horizontal_kernel, iterations=iter_hor)
<<<<<<< HEAD
    img_eroded_horizontal = remove_small_components(img_eroded_horizontal, img_bin_inv.shape[1]/10)
=======
    img_eroded_horizontal = remove_small_components(img_eroded_horizontal, img_bin_inv.shape[1]/6)
>>>>>>> 8ac85790c06a6f922ae8bf2b91f646d775ff7e28

    horizontal_lines = cv2.dilate(img_eroded_horizontal, horizontal_kernel, iterations=iter_hor)
    # Combine vertical and horizontal lines
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    cv2_save("edges", edges)        
    cv2_save("vertical_lines_dilate", vertical_lines)
    cv2_save("horizontal_lines_erode", img_eroded_horizontal)
    cv2_save("horizontal_lines_dilate", horizontal_lines)
    cv2_save("img_vh combined", img_vh)
    return ~img_vh


# `combine_images` is a function that takes two binary images as input: `img_bin` and `img_vh`. It
# performs a bitwise OR operation on the two images to combine them, and then applies median
# filtering to the resulting image to remove noise. The function returns the filtered image.
def combine_images(img_bin, img_vh):
    bitor = cv2.bitwise_or(img_bin, img_vh)
    img_median = bitor
    return img_median

# `erode_dilate_images` is a function that takes a binary image as input and applies morphological
# operations to detect vertical and horizontal lines in the image. It first erodes the image using
# vertical and horizontal kernels, then combines the resulting images using a bitwise OR
# operation. It then erodes the combined image using a combined kernel and applies thresholding to
# obtain a binary image with white lines on a black background. Finally, it uses bitwise XOR and
# NOT operations to get the final result, which is a binary image with black lines on a white
# background.
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

    _, img_combined_lines = cv2.threshold(img_combined_lines, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bitwise_xor_result = cv2.bitwise_xor(img_median, img_combined_lines)
    #cv2_imshow("bitwise_xor_result ", bitwise_xor_result)

    bitwise_not_result = cv2.bitwise_not(bitwise_xor_result)

    return bitwise_not_result

# https://cvexplained.wordpress.com/2020/06/06/sorting-contours/
# Sorting boxes
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


# `get_contours_and_boxes` is a function that takes a binary image as input and uses OpenCV's
# `findContours` function to find the contours in the image. It then sorts the contours from top
# to bottom and returns both the contours and their corresponding bounding boxes. The function
# also adds a border to the image to ensure that the edge columns are not missed during contour
# detection.
def get_contours_and_boxes(img_vh):
    contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours from top to bottom
    contours, bounding_boxes = sort_contours(contours, method="top-to-bottom")
    print("len contours: ", len(contours)," len bounding_box ",len(bounding_boxes))
    return contours, bounding_boxes


# `get_boxes` is a function that takes a list of contours and the width and height of the image as
# input. It loops through each contour and gets the bounding box for the contour using the
# `cv2.boundingRect` function. It then checks if the width and height of the bounding box are less
# than 90% of the image width and height respectively. If the bounding box is not too big, it adds
# the bounding box information to a list of boxes. Finally, it returns the list of boxes.
def get_boxes(contours,img_w,img_h):
    boxes = []
    for c in contours:
        # Get bounding box for contour
        x, y, w, h = cv2.boundingRect(c)
        print('x', x, 'y', y, 'width', w, 'height', h)
        # Check if box not too big
        if (w < 0.9*img_w and h < 0.9*img_h):
            box_info = {'x': x, 'y': y, 'width': w, 'height': h}
            boxes.append(box_info)
    print(len(boxes))
    return boxes


def sort_boxes_by_row(boxes, avg_height):
    rows_list = []
    current_column = []
    
    for index in range(len(boxes)):
        # Check first box
        if index == 0:
            current_column.append(boxes[index])
            last_box = boxes[index]
        else:
            # Check if box in same row
            if boxes[index]['y'] <= last_box['y'] + avg_height / 2:
                current_column.append(boxes[index])
                last_box = boxes[index]

                if index == len(boxes) - 1:
                    rows_list.append(current_column)
            else:
                # New row found
                rows_list.append(current_column)
                current_column = []
                last_box = boxes[index]
                current_column.append(boxes[index])
    return rows_list



def arrange_boxes_in_order(row):
    max_columns= 0
    selected_index = 0
    for i in range(len(row)):
        current_length = len(row[i])
        if current_length > max_columns:
            max_columns = current_length
            selected_index = i
    # Find centers of columns
    selected_row =row[selected_index]
    centers = []
    for j in range(len(selected_row)):
        center = int(selected_row[j]['x'] + selected_row[j]['width'] / 2)
        centers.append(center)
    centers = np.array(centers)
    centers.sort()
    organized_boxes = []
    for i in range(len(row)):
        temp_list = []
        for k in range(max_columns):
            temp_list.append([])
        for j in range(len(row[i])):
            # Find column for box
            distance = abs(centers -(row[i][j]['x'] + row[i][j]['width'] /4))
            min_distance = min(distance)
            index = list(distance).index(min_distance)
            temp_list[index].append(row[i][j])
        organized_boxes.append(temp_list)
    return organized_boxes

