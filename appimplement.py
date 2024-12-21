import streamlit as st
import cv2
from PIL import Image
import subprocess

# Function to draw bounding box using OpenCV
def draw_bbox(image_path):
    from drawingmodule import select_bounding_box  # Import the OpenCV script
    bbox = select_bounding_box(image_path)
    return bbox

# Title
st.title("Interactive Bounding Box Selector")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save the uploaded file temporarily
    temp_file = "temp_image.jpg"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())

    # Display the uploaded image
    st.image(temp_file, caption="Uploaded Image", use_container_width=True)

    # Button to launch OpenCV drawing interface
    if st.button("Select Bounding Box"):
        bbox = draw_bbox(temp_file)
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            st.write(f"Bounding Box Coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

            # Process the coordinates using an external script
            result = subprocess.run(
                ["python", "test1image.py", str(xmin), str(ymin), str(xmax), str(ymax)],
                capture_output=True,
                text=True,
            )
            st.write("Output from script:")
            st.write(result.stdout)
