**Image to Text Conversion using OCR and Tesseract**

This script is written in Python and uses the following libraries:

- OpenCV (cv2)
- PyOCR
- Pandas
- PyTesseract

The script performs Optical Character Recognition (OCR) on an image file and converts it into a text format. The text is then split by line and stored in a pandas dataframe. Finally, the dataframe is saved as a CSV file.

**How to run the script**
1. Install the required libraries: cv2, pyocr, pandas, and pytesseract.
```
sudo apt-get install python-opencv
pip install pyocr
pip install pandas
sudo apt-get install tesseract-ocr
pip install pytesseract
```
Note: Make sure that you have the latest version of pip installed by running pip install --upgrade pip before installing the libraries.





2. Set the path to Tesseract OCR in the script. If you are using Windows, uncomment the line pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" and comment the line pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract". If you are using Linux, uncomment the line pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract" and comment the line pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe".

3. Replace the file name in the line image = cv2.imread('table.png') with the name of the image file you want to perform OCR on.

4. Run the script using a Python environment.

**Output**

The script will generate a CSV file with the same name as the image file, with the .csv extension. The file will contain the text data extracted from the image.
