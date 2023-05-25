#!/bin/bash

git clone https://github.com/facebookresearch/detectron2.git

cd detectron2

python -m pip install -e .

git clone https://github.com/PaddlePaddle/PaddleOCR.git

cd PaddleOCR

python -m pip install -r requirements.txt

python setup.py install

echo "Detectron2 & PaddleOCR installed successfully."




