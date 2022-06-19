# TensorflowSSD

This project contains two types of files:\
&emsp;**Python**: dataset generator, model, converter for OpenCV format;\
&emsp;**C++**: object detection application for camera image.

## Working steps
### Generate Dataset
1. copy PNG images of backgrounds to `back_images` folder;
2. copy PNG images for every class to folders inside `image_classes`;\
**Note:** all the images must have transparent background and contain only one cropped object.
3. execute all blocks inside `generate_image_label_dataset.ipynb` file;\
As a result there will be dataset inside `ssd_dataset` folder.

### Train model
1. execute all blocks of `ssd_learner.ipynb` file;\
It will train Single Shot Detector model of Tensorflow and save its `.h5` file.\
Also it will save byte files with ssd-boxes (will be used for C++ project).
2. execute all blocks of `convert_model_to_pb.ipynb` file;\
It will save model's `.pb` file to `output_files` folder.
3. copy all the content of `output_files` folder to `cpp_files` directory.

### Camera application (C++)
1. in `camera_detect` directory build C++ project using command:
```bash
make
```
2. execute `camera_detect` application:
```bash
./camera_detect
```
