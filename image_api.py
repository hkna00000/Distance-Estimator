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
        cropped_img = img.crop((bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax))

        # Process the cropped image (e.g., passing it to your model)
        result = process_model(cropped_img, bbox)

        # Log the result to make sure it’s correct before sending back
        print(f"Processed result: {result}")

        return {"result": result}  # This should return a JSON response with the "result"
    except Exception as e:
        print(f"Error: {e}")  # Log the error to the terminal
        return {"error": str(e)}  # Return error as a JSON response

def process_model(img, bbox):
    xmin, ymin, xmax, ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
    try:
        print(f"Processing image with bounding box: {xmin}, {ymin}, {xmax}, {ymax}")
        # Save the cropped image to a temporary file
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img_file:
        #     img.save(temp_img_file, format='PNG')
        #     temp_img_path = temp_img_file.name
        #     print(f"Temporary image file created at: {temp_img_path}")
        
        # Run the classification script with the image and bounding box coordinates
        result1 = subprocess.run(
            ["python", "predict_module.py", str(xmin), str(ymin), str(xmax), str(ymax)],
            capture_output=True,
            text=True
        )
        
        if result1.returncode != 0:
            return f"Error in predict module: {result1.stderr.strip()}"

        output1 = result1.stdout.strip()
        print(f"Distance module: {output1}")
                 
        # # Run the classification script with the cropped image path
        # result2 = subprocess.run(
        #     ["python", "classi_module.py", temp_img_path, str(xmin), str(ymin), str(xmax), str(ymax)],
        #     capture_output=True,
        #     text=True,            
        # )
        # if result2.returncode != 0:
        #     return f"Error in classi_module.py: {result2.stderr.strip()}"
        # # Capture the output from the second script
        # output2 = result2.stdout.strip()
        # print(f"Classify module: {output2}")
        # # Cleanup the temporary image file
        # if os.path.exists(temp_img_path):
        #     os.remove(temp_img_path)
        # return output2
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
