# Distance Estimator
 This is the model to predict the distance from your camera to the object
## Purpose

To estimate distance to objects (cars, pedestrians, trucks) in the scene on the basis of detection information

## Overview
Train a deep learning model that takes in bounding box coordinates of the detected object and estimates distance to the object.

Input: bounding box coordinates (xmin, ymin, xmax, ymax) <br/>
Output: distance (z) and object label (Car, Van, Truck, Pedestrian, Misc, Cyclist, Person_sitting)

## Usage
- To train and test the models, execute the following from `distance-estimator` directory, unless mentioned otherwise
- While training, you can use the new_classification.ipynb notebook to try predicting with your model and tuning
- The predict_module.py is an important file in which you can choose your model for predicting module and classifying module integrating in-app.
- The mypredict.py file is the visualization of your result on test file from your model, modify the result directory before using.
### Training
1. Use `mytrain.py` to define your own model, choose hyperparameters, and start training!

### Inference
1. Use `mypredict.py` to generate predictions for the test set.
2. Use `visualizer.py` to visualize the predictions.
```
cd KITTI-distance-estimation/
python prediction-visualizer.py
```
### App Implementing
1. Run the image_api to start creating an api for frontend and backend link
```
python image_api.py
```
2. Run ModalAppBackend.py to call the backend
```
python ModalAppBackend.py
```
3. Run the ModelAppFrontend.html to use the app
### Results
![](GithubInstance/000.png)
## Appendix
### Prepare Data
1. **Download KITTI dataset**
```shell
# get images
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
unzip data_object_image_2.zip

# get annotations
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
unzip data_object_label_2.zip
```

Organize the data as follows:

```shell
KITTI-distance-estimation
|-- original_data
    |-- test_images
    |-- train_annots
    `-- train_images
```

2. **Convert annotations from .txt to .csv**<br/>
We only have train_annots. Put all information in the .txts in a .csv

```shell
python generate-csv.py --input=original_data/train_annots --output=annotations1.csv
```

The annotations contain the following information

```
Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
```

3. **Generate dataset for distance estimation**<br/>
Using only `annotations1.csv` (file generated using `train_annots`), split the dataset into `train.csv` and `test.csv` set.

```shell
python generate-depth-annotations.py
 ```

This dataset contains the following information:
`filename, xmin, ymin, xmax, ymax, angle, xloc, yloc, zloc`

Organize your data as follows
```
KITTI-distance-estimation
|-- original_data
|    |-- test_images
|    |-- train_annots
|    `-- train_images
`-- distance-estimator/
    |-- data
        |-- test.csv
        `-- train.csv
```

4. **Visualize the dataset**<br/>
Use `visualizer.py` to visualize and debug your dataset. Edit `visualizer.py` as you want to visualize whatever data you want.


### Acknowledgements
[KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
