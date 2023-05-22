import cv2
import numpy as np
import pandas as pd
from image_processing import cv2_imshow,remove_small_components
import os
from paddleocr import PaddleOCR
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from threading import Lock

lock = Lock()

rec_model_path = 'OCR/model_final.pth'
ocr = PaddleOCR(rec_model=rec_model_path, show_logs= False, use_gpu=True,lang='en',show_log = False,use_angle_cls = True)


# `process_cells` is a function that takes in a list of cells and an image, and processes each
# cell to extract text using OCR (Optical Character Recognition). It applies various image
# processing techniques such as dilation, erosion, and resizing to the cell image to improve the
# OCR accuracy. It then uses PaddleOCR to extract text from the processed cell image and writes
# the extracted text along with the cell coordinates to a text file. Finally, it returns the
# extracted text from all the cells in the list as a single string.
def process_cells(cells, img):
    text = ''
    for cell in cells:
        y, x, w, h = cell['x'], cell['y'], cell['width'], cell['height']
        cell_img = img[x:x + h, y:y + w]
        scale_ratio = float(300) / float(min(cell_img.shape[:2]))
        cell_img = remove_small_components(cell_img, min(cell_img.shape[:2])/10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        border = cv2.copyMakeBorder(cell_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255])
        resized = cv2.resize(border, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_CUBIC)
        dilate_iter = max(3, int((w * h) / 5000)) 
        erode_iter = max(2, int((w * h) / 7500)) 

        dilated = cv2.dilate(resized, kernel, iterations=dilate_iter)
        eroded = cv2.erode(dilated, kernel, iterations=erode_iter)
        eroded_rgb = cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)
        result = ocr.ocr(eroded_rgb)
        if not result: # If we got empty text
            return None
        cv2.imwrite(f"image_process/cells/{x+y+w}.jpg",eroded_rgb)
        for line in result:
            for word_info in line:
                extracted_text = word_info[-1][0]
                text += " " + extracted_text
    return text


# `process_row` is a function that takes in a tuple of a row and an image, and processes each cell
# in the row in parallel using the `process_cells` function. It returns a list of the extracted
# text from each cell in the row.
def process_row(args):
    row , img = args
    print("Next row")
    row_data = []
    for cell in row:
        if cell:  # Check if cell is not empty
            if len(cell) == 0:
                row_data.append(' ')
            else:
                results = process_cells(cell, img)

                row_data.append(' ')  # Append space if the cell is empty
    return row_data



 # The `extract_data_to_csv` function takes in the finalboxes (a list of cells that make up a
# table), the image of the table, the Flask app object, and the table index. It then uses a
# ProcessPoolExecutor to process each row of cells in parallel using the `process_cells` function.
# The extracted text from each cell is appended to a list for each row, and the resulting list of
# rows is used to create a Pandas DataFrame. The DataFrame is then styled and saved as an Excel
# file in the app's upload folder with a filename that includes the table index. Finally, the
# filename is returned.
def extract_data_to_csv(finalboxes, img, app, table_index):
    extracted_data = []
    with ProcessPoolExecutor() as executor:
        for row in finalboxes:
            print("Next row")
            futures = [executor.submit(process_cells, cell, img) for cell in row if cell]
            row_data = [f.result() for f in futures if f.result() is not None]
            extracted_data.append(row_data)
    dataframe = pd.DataFrame(extracted_data)
    styled_dataframe = dataframe.style.set_properties(align="left")
    output_filename_excel = f"output_{table_index}.xlsx"
    output_filepath_excel = os.path.join(app.config['UPLOAD_FOLDER'], output_filename_excel)
    styled_dataframe.to_excel(output_filepath_excel)
    output_filename_csv = f"output_{table_index}.csv"
    output_filepath_csv = os.path.join(app.config['UPLOAD_FOLDER'], output_filename_csv)
    dataframe.to_csv(output_filepath_csv, index=False)
    return output_filename_excel, output_filename_csv


