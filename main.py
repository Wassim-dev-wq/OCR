import pytesseract
import pandas as pd
from image_processing import preprocess_image, cv2_imshow
from structure_recognition import recognize_structure
import cv2

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract" # for Linux


def main():
    image_path = "tables/table.png"
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Detect the table from the image
    table_image = recognize_structure(preprocessed_image)
    cv2_imshow("Final",table_image)
    
if __name__ == "__main__":
    main()
