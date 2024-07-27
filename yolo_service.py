from ultralytics import YOLO
from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import cloudinary
from cloudinary import CloudinaryImage
import cloudinary.uploader
import cloudinary.api
import os
import json

load_dotenv()

app = Flask(__name__)
model = YOLO('./yolov8n.pt')  # Load a pre-trained YOLOv8 model

config = cloudinary.config(secure=True)
@app.route('/detect', methods=['POST'])
def detect_objects():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = Image.open(BytesIO(image_data))
    results = model(image)
    detections = []
    
    buffer = BytesIO()

    for result in results:
        annotated_image_array = result.plot()
        annotated_image = Image.fromarray(annotated_image_array)
        annotated_image.save(buffer, format='JPEG') 

    buffer.seek(0)

    upload_response =  cloudinary.uploader.upload(buffer, public_id="detected_image", unique_filename = True, overwrite=True)
    srcURL = upload_response['secure_url']

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  
        scores = result.boxes.conf.cpu().numpy() 
        class_ids = result.boxes.cls.cpu().numpy() 

        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append({
                'class': result.names[int(class_id)],
                'confidence': float(score),
                'box': [int(coord) for coord in box]
            })
    return jsonify({ 
                    "detections": detections,
                     "annotated_image": srcURL 
     })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
