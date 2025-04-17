from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import inference
from PIL import Image
from io import BytesIO
import os
import json
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...),screen_width: int = Form(...), screen_height: int = Form(...),):
    image = Image.open(BytesIO(await file.read()))
    result = inference.run_inference_with_calibration(image, screen_width, screen_height)
    pred_x = result["pred_x"]
    pred_y = result["pred_y"]
    direction = result["direction"]
    print(pred_x)
    return {"x" : pred_x, "y" : pred_y, "direction" : direction}

@app.post("/save_click")
async def save_click(
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    screen_width: int = Form(...),
    screen_height: int = Form(...),
    index: int = Form(...),
):
    os.makedirs("temp_data", exist_ok=True)
    json_path = "temp_data/capture_data.json"
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            json.dump({"width": screen_width, "height": screen_height, "image_data": []}, f)

    image_data = await file.read()
    filename = f"image{index}.jpg"
    filepath = os.path.join("temp_data", filename)
    with open(filepath, "wb") as f:
        f.write(image_data)

    with open(json_path, "r") as f:
        data = json.load(f)

    data["image_data"].append({"x": x, "y": y, "filename": filename})

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"status": "saved"}