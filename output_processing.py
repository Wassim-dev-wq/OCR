import cv2
import numpy as np
import pandas as pd
from image_processing import cv2_imshow

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def is_sensible_text(text):
    return len(text) > 0



def process_cells(cells, img, configs):
    text = ''
    for cell in cells:
        y, x, w, h = cell['x'], cell['y'], cell['width'], cell['height']
        cell_img = img[x:x + h, y:y + w]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        border = cv2.copyMakeBorder(cell_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
        resized = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dilated = cv2.dilate(resized, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=2)
        cv2_imshow("Cell Image", cell_img)
        cv2_imshow("Dilated", dilated)
        cv2_imshow("Eroded", eroded)
        best_extracted_text = ""
        best_sensible = False
        for config in configs:
            extracted_text = pytesseract.image_to_string(eroded, config=config)
            print(f"Cell [{x}, {y}, {w}, {h}] - Config: {config}, Extracted Text: '{extracted_text}'")
            
            if is_sensible_text(extracted_text) and not best_sensible:
                best_extracted_text = extracted_text
                best_sensible = True

        text += " " + best_extracted_text[:-2]
    return text



def extract_data_to_csv(finalboxes, img):
    configs = [
    "--psm 3 --oem 3",
    "--psm 6 --oem 3",
    "--psm 11 --oem 3",
    "--psm 3 --oem 1",
    "--psm 6 --oem 1",
    "--psm 11 --oem 1",
    "--psm 4",
    "--psm 7",
    "--psm 10",
    "--psm 13",
]
    extracted_data = []
    for row in finalboxes:
        row_data = []
        for cell in row:
            if len(cell) == 0:
                row_data.append(' ')
            else:
                row_data.append(process_cells(cell, img, configs))
        extracted_data.append(row_data)

    dataframe = pd.DataFrame(extracted_data)
    print(dataframe)
    styled_dataframe = dataframe.style.set_properties(align="left")
    styled_dataframe.to_excel("output.xlsx")

