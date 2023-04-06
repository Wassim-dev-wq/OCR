import pytesseract
import pandas as pd
from image_processing import preprocess_image, cv2_imshow
from structure_recognition import recognize_structure

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract" # for Linux


def main():
    image_path = "tables/table5.jpg"
    #output_csv_path = "output.csv"

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Detect the table from the image
    table_image = recognize_structure(preprocessed_image)
    cv2_imshow("Table image:",table_image)

    #print(f"CSV file generated: {output_csv_path}")

if __name__ == "__main__":
    main()
