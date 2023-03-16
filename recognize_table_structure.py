import cv2

def display_image(img, window_name="Image"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_contours(contours, min_size_ratio, max_size_ratio, img_size):
    filtered_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > img_size[1] * min_size_ratio and w < img_size[1] * max_size_ratio) and \
           (h > img_size[0] * min_size_ratio and h < img_size[0] * max_size_ratio):
            filtered_contours.append(c)
    return filtered_contours


def recognize_table_structure(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = gray_img.shape

    # Apply binary threshold
    thresh_value = 190
    max_value = 255
    _, thresh_img = cv2.threshold(gray_img, thresh_value, max_value, cv2.THRESH_BINARY)

    # Find contours
    mode = cv2.RETR_TREE
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(thresh_img, mode, method)

    # Filter out small contours
    min_size_ratio = 0.05
    max_size_ratio = 0.9
    filtered_contours = filter_contours(contours, min_size_ratio, max_size_ratio, (height, width))

    # Draw filtered contours on a copy of the original image
    contour_img = img.copy()
    cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)

    # Display the original image, grayscale image, binary image, and contour image
    display_image(img, "Original Image")
    display_image(gray_img, "Grayscale Image")
    display_image(thresh_img, "Binary Image")
    display_image(contour_img, "Contour Image")

    return filtered_contours



# Load image file
image_path = 'tables/table2.jpg'
image = cv2.imread(image_path)

recognize_table_structure(image)
