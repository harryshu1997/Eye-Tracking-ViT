from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import inference
from PIL import Image
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    width, height = 1920, 1080  
    pred_x, pred_y, direction = inference.predict(image, width, height)
    return {"x" : pred_x, "y" : pred_y, "direction" : direction}
