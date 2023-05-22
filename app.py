import os
import cv2
from flask import Flask, redirect, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import table_structure_detection
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from image_processing import preprocess_image
from output_processing import extract_data_to_csv
from structure_recognition import recognize_structure
from concurrent.futures import ProcessPoolExecutor
from pdf2image.pdf2image import convert_from_path


cfg = get_cfg()
# Settings
cfg.merge_from_file('content/config.yaml')

# Use cuda if available else we use cpu
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Using the tableBank model
cfg.MODEL.WEIGHTS = 'content/model_final.pth'

# Create predictor
predictor = DefaultPredictor(cfg)

app = Flask(__name__, template_folder='templates')

# Folder for uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Limit upload to .png, .jpg, .jpeg files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if file.filename is None:
                return redirect(request.url)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output_filename = main(filepath)
            return render_template('index.html', result=output_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    global predictor
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

def process_table(args):
    table, i = args
    preprocessed_image = preprocess_image(table)
    final_boxes, table_image = recognize_structure(preprocessed_image)
    output_filename_excel, output_filename_csv = extract_data_to_csv(final_boxes, table_image, app, i)
    return output_filename_excel, output_filename_csv

def process_image(image, i):
    table_structure_detection.draw_detected_tables(image, predictor)
    table_list, _ = table_structure_detection.extract_table_images(image, predictor)
    with ProcessPoolExecutor() as executor:
        output_filenames_excel, output_filenames_csv = zip(*executor.map(process_table, [(table, i) for i, table in enumerate(table_list)]))
    return output_filenames_excel, output_filenames_csv


def main(file_path):
    global predictor
    ext = os.path.splitext(file_path)[1]
    if ext.lower() == '.pdf':
        images = convert_from_path(file_path)
        output_filenames_excel = []
        output_filenames_csv = []
        for i, image in enumerate(images):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            page_output_filenames_excel, page_output_filenames_csv = process_image(image, i)
            for filename in page_output_filenames_excel:
                output_filenames_excel.append(filename)
            for filename in page_output_filenames_csv:
                output_filenames_csv.append(filename)
        return output_filenames_excel, output_filenames_csv
    else:
        image = cv2.imread(file_path)
        output_filenames_excel, output_filenames_csv = process_image(image, 0)
        return output_filenames_excel, output_filenames_csv



if __name__ == "__main__":
    app.run(port=5000, debug=False)