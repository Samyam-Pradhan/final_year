from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Load the face detector and mask detector models
prototxtPath = r"facedetector\deploy.prototxt"
weightsPath = r"facedetector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("MSN_maskdetector.keras")

# Function to detect face and predict mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    return (locs, preds)

# Define a Pydantic model to handle the image data
class ImageData(BaseModel):
    image: str

@app.post("/")
def home():
    return {"test": "hello world"}
    

@app.post("/detect_mask")
async def detect_mask(image_data: ImageData):
    print(image_data)
    # Convert the base64 image data to an OpenCV image
    img_data = image_data.image.split(",")[1]  # Remove the base64 prefix
    nparr = np.fromstring(img_data, np.uint8)
    prototxtPath = r"facedetector\deploy.prototxt"
    weightsPath = r"facedetector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model("MSN_maskdetector.keras")

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize the frame to match your detection requirements (optional)
    frame = imutils.resize(frame, width=400)
    
    # Detect faces and predict mask
    locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # Prepare the result
    results = []
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        label = "Mask" if mask > withoutMask else "No Mask"
        confidence = max(mask, withoutMask) * 100
        
        results.append({
            "label": label,
            "confidence": confidence,
            "box": [startX, startY, endX, endY]
        })
    
    return JSONResponse(content={"results": results})

# Run the server with `uvicorn` or as per your project setup.
