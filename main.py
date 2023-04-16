import pytesseract
from image_processing import preprocess_image, cv2_imshow
from structure_recognition import recognize_structure
from output_processing import extract_data_to_csv

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract" # Linux



def main():

    image_path = "tables/Small.jpg"
    
    

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Detect the table from the image
    final_boxes, table_image = recognize_structure(preprocessed_image)

    
    cv2_imshow("Final",table_image)
    print("Final boxes:", final_boxes)  

    # Extracting data
    extract_data_to_csv(final_boxes, table_image)

if __name__ == "__main__":
    main()
