'''
Purpose: visualize data from the dataframe
'''
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\Learning\GitHub\KITTI-distance-estimation\distance-estimator\data\train.csv")

for idx, row in df.iterrows():
    # Define file path first
    fp = os.path.join("original_data/train_images/", row['filename'].replace('.txt', '.png'))

    # Check if the file exists before attempting to load it
    if os.path.exists(fp):
        print(f"Image path: {fp}")
        im = cv2.imread(fp)
        if im is None:
            print("Failed to load image.")
            continue
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])

        # Draw a line and a rectangle as intended
        cv2.line(im, (int(1224 / 2), 0), (int(1224 / 2), 370), (255, 255, 255), 2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add text to the image
        string = "({}, {})".format(row['angle'], row['zloc'])
        cv2.putText(im, string, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Convert image from BGR to RGB for displaying with matplotlib
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Display the image using matplotlib
        plt.imshow(im_rgb)
        plt.title("Detections")
        plt.axis("off")
        plt.show()
