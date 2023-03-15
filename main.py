import cv2
import pandas as pd
import pytesseract

# Set the path to tesseract
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract" # for Linux

# Load image file
image = cv2.imread('table.png')
if image is None:
    print("Error: Image not found or loaded")
else:
    # Pre-process the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    text = pytesseract.image_to_string(gray)

    # Split the text by line
    lines = text.split('\n')

    # Define the delimiter for the data
    delimiter = ' '

    # Extract the data
    data = []
    for line in lines:
        data.append(line.split(delimiter))

    # Convert the extracted data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the extracted data to a CSV file
    df.to_csv('table.csv', index=False)
