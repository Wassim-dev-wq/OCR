import cv2
import numpy as np
from image_processing import cv2_imshow,cv2_save

def draw_detected_tables(image, detector):
    # Get predictions from the detector
    predictions = detector(image)
    # Extract bounding boxes from predictions
    bounding_boxes = predictions["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()
    
    # Iterate through each bounding box
    for box in bounding_boxes:
        # Define start and end points for the rectangle
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        # Draw rectangle on the image
        image = cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)
        
    # Display image with detected tables
    print("Table Detection:")
    #cv2_imshow("Detected tables", image)
    cv2_save(f"Detected table", image)


def extract_table_images(image, detector, border=2):
    # Get predictions from the detector
    predictions = detector(image)
    # Extract bounding boxes from predictions
    bounding_boxes = predictions["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()
    
    # Containers for table images and coordinates
    tables = []
    coords = []

    # Iterate through each bounding box
    for i, box in enumerate(bounding_boxes):
        # Define start and end points for the rectangle
        start_point = (max(0, int(box[0]) - border), max(0, int(box[1]) - border))
        end_point = (min(image.shape[1], int(box[2]) + border), min(image.shape[0], int(box[3]) + border))
        # Extract table image from the original image
        table = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        # Append to the list of tables
        tables.append(table)
        # Append to the list of coordinates
        coords.append([start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1]])
        # Display each table
        print(f"Table {i}:")
        cv2_save(f"Detected table {i}", table)
        #cv2_imshow(f"Detected table {i}", table)
        print()
    
    return tables, coords
