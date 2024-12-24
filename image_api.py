from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import base64
import subprocess
import os
import tempfile
app = FastAPI()

# Create a model for the incoming data
class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

class ImageData(BaseModel):
    bbox: BoundingBox
    image: str  # base64 encoded image

@app.post("/process")
async def process_image(data: ImageData):
    try:
        # Decode the base64 image
        img_data = base64.b64decode(data.image.split(',')[1])
        img = Image.open(io.BytesIO(img_data))

        # Crop the image based on the bounding box
        bbox = data.bbox

        # Process the image (e.g., passing it to your model)
        result = process_model(img,bbox)

        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

def process_model(img, bbox):
    xmin, ymin, xmax, ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
    try:
        print(f"Processing image with bounding box: {xmin}, {ymin}, {xmax}, {ymax}")
        # Save the cropped image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img_file:
            img.save(temp_img_file, format='PNG')
            temp_img_path = temp_img_file.name
            print(f"Temporary image file created at: {temp_img_path}")
        
        # Run the classification script with the image and bounding box coordinates
        result = subprocess.run(
            ["python", "predict_module.py", temp_img_path, str(xmin), str(ymin), str(xmax), str(ymax)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return f"Error in script: {result.stderr.strip()}"   
        output1 = result.stdout.strip()
        return output1
                 
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
