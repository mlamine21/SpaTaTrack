# SpaTaTrack (Spatial Target gaze Tracking) 

## Installation
Dependencies : ptgaze, opencv, torch (cuda supported), omegaconf

TODO : add installation instructions

## Architecture Overview

This is a **real-time eye gaze tracking and visualization system**. The system processes video streams to detect faces, estimate 3D gaze vectors, and visualize where participants are looking in a predefined 3D scene.

**NB : This is a work in progress. The program runs in ~30fps with 2 identical cameras using cuda capabilities, with the following specs : 16gb RAM, core i5-8600K @ 3.60GHz, Nvidia GeForce GTX 1070 Ti.** 

#### Usage demo (`DemoProcessGaze.py`)
Complete gaze estimation pipeline with two streams (a camera stream and a video) and an example of scene configuration. 
- Each stream is processed in a separate GazeProcessor thread (`gazeProcessor.py`)

- Uses custom video processing thread `fileVideoStream.py` for efficient frame reading with FFmpeg backend.
- Uses ETH-XGaze model (`eth-xgaze_resnet18.pth`) with mediapipe as default face detection backend (can be changed in 'eth-xgaze.yaml). 
- Display a reconstruction of the scene using 3 2D views.

#### View System (`templateView.py`, `topView.py`, `sideView.py`, `sideViewOther.py`)
Modular 3D visualization system with inheritance hierarchy:
- **TemplateView**: Base class with common 3D→2D projection logic
- **TopView**: Bird's-eye view (X-Z plane) for spatial relationships
- **SideView**: Profile view (Y-Z plane) for depth perception
- **SideViewOther** : Front view

- **Static Views**: Pre-computed room layouts stored in `computeStaticView()`
- **Real-time Updates**: Overlay participant data without redrawing static elements
- **Coordinate Transformation**: Handles camera rotation matrices with caching

#### RoomConfiguration (`roomConfiguration.py` and `config.json`)
Construct the 3D scene (objects, cameras) by reading the config.json file :  

- **scene** : dimensions of the scene (in meters) -> list of float

- **cameras** : specify the position (3D list of floats, in meters), size (3D list of floats, in meters), angle (3D list of int, in degrees) and shape in each view (dict{tuple : string} -> (axis1;axis2) : shape {rectangle, circle}) for each camera. Each camera corresponds to one stream (can be a camera stream or video)

- **participant_colors** : define colors used to represent the participants (must match the cameras id) and their associated camera.

- **zones** : define the size of the bounding box around the participants' heads to determine if another participant is looking at them.

- **objects** : define all the objects in the scene with their position, size, color, shape and angle. In the examples, we added walls at the edges of the scene to distinguish when participants look away and when no target could be extracted (no face detected, error...)

Note : We defined a parsing system that allows to use variables in the config file. A variable starts with the character "$", and it is possible to access a field of the variable using a string between brackets (e.g. [field]), and an index using a number between brackets (e.g. [0]). 

Example : $scene[size][1] -> field "scene" of the json, field "size" of the object "scene", index 1 of the list. 

#### Participant (`participant.py`)
Data model for tracked individuals with temporal data:
- **Multi-frame Data**: Arrays of head positions and gaze vectors over time
- **Color Assignment**: Participant identification by color codes
- **Camera Integration**: Links to specific camera configurations


### Ray-Box Intersection for Gaze Targeting
The system determines what objects participants are looking at using 3D ray-box intersection:


### Known issues 

Ptgaze rasies type error caused by numpy version : Use the following commit for ptgaze : 
 https://github.com/drewp/pytorch_mpiigaze_demo/commit/e30d064cbe9f00ed6c8239f2a7cb1ff9d29e9867

