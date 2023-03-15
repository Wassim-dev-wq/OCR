import cv2

def display_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_table_structure(img):
    # Grayscale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = gray_img.shape
    # print('image height: ', height, 'image width: ', width)
    display_image(gray_img)

    # Thresholding
    thresh_value = 190
    max_value = 255
    thresh_type = cv2.THRESH_BINARY
    _, thresh_img = cv2.threshold(gray_img, thresh_value, max_value, thresh_type)
    display_image(thresh_img)

    # Finding contours
    mode = cv2.RETR_TREE
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(thresh_img, mode, method)
    print(contours)


# Load image file
image_path = 'tables/table2.jpg'
image = cv2.imread(image_path)

recognize_table_structure(image)
