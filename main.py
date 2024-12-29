import os
import cv2 as cv
from typing import Annotated
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI()
model = YOLO('model/weights/best.pt')


@app.post('/upload_image')
async def upload_image(image: UploadFile = File(...)):
    try:
        with open(f'uploads/{image.filename}', 'wb') as f:
            f.write(await image.read()) 
        return {'filename': image.filename, 'message': 'Upload successful!'}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.get('/predict/{img_name}')
async def get_prediction(img_name: str):
    source = os.path.join('uploads', img_name)
    target = os.path.join('predicts', img_name)
    
    result = model(source)
    result[0].save(filename=target)
    
    return FileResponse(target) if os.path.isfile(target) else {'error': 'Image not found'}
