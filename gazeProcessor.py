import os
import warnings

# Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
# warnings.simplefilter(action='ignore', category=FutureWarning)

from threading import Thread
from networkx import config
import numpy as np
from omegaconf import OmegaConf
import sys
from fileVideoStream import FileVideoStream
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import cv2
import time
import pathlib
from ptgaze.gaze_estimator import GazeEstimator, GazeEstimationMode
from ptgaze.config_schema import AppConfig

from ptgaze.utils import (
    check_path_all,
    download_dlib_pretrained_model,
    download_ethxgaze_model,
    download_mediapipe_face_landmarker,
    download_mpiifacegaze_model,
    download_mpiigaze_model,
    expanduser_all,
    set_dummy_camera_params,
)

class GazeProcessor(Thread):
    def __init__(self, stream, save_csv=False, csv_filename=None):
        Thread.__init__(self)
        self.stream = stream
        self.save_csv = save_csv
        self.csv_filename = csv_filename
        self.frame_number = 0
        self.continue_processing = False
        self.stop = False
        self.initGazeEstimator()

        if(self.save_csv):
            self.gaze_data_df = pd.DataFrame(columns=['frame', 'head_x', 'head_y', 'head_z', 'gaze_x', 'gaze_y', 'gaze_z'])

        self.last_frame = None
        self.isRunning = False
    def initGazeEstimator(self):
        # Temporarily suppress stderr during GazeEstimator instantiation
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            self.config = OmegaConf.load('eth-xgaze.yaml')
            self.config = OmegaConf.merge(OmegaConf.structured(AppConfig), self.config)
            #self.config.PACKAGE_ROOT = "C:\\Users\\mla\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ptgaze".as_posix()
            expanduser_all(self.config)
            # OmegaConf.set_readonly(self.config, True)

            if self.config.face_detector.mode == "dlib":
                download_dlib_pretrained_model()
            if self.config.face_detector.mode == "mediapipe":
                download_mediapipe_face_landmarker()

            check_path_all(self.config)
            # package_root = pathlib.Path(__file__).parent.resolve()
            # self.config.PACKAGE_ROOT = package_root.as_posix()

            # self.config.face_detector.mode = "mediapipe"
            # self.config.device = "cpu"
            checkpoint_path = download_ethxgaze_model()
            self.config.gaze_estimator.checkpoint = checkpoint_path.as_posix()
            self.mode = GazeEstimationMode(self.config.mode)
            self.gaze_estimator = GazeEstimator(self.config)  # Warning occurs here
            
        finally:
            # Always restore stderr
            sys.stderr.close()
            sys.stderr = original_stderr

    def getLastFrame(self):
        return self.last_frame

    def run(self):
        self.fvs = FileVideoStream(self.stream, queueSize=1).start()
        
        if not self.fvs.stream.isOpened():
            raise ValueError("Failed to initialize video stream.")
        else:
            print(f"{self.stream} : Video stream opened successfully.")
        self.frame_number = 0
        self.isRunning = True
        while not self.continue_processing:
            print("waiting start signal...")
            time.sleep(0.01)
        while self.fvs.more() and not self.stop:
            curr_time = time.time()
            
            if self.frame_number==0 : print('in', self.stream)
            self.frame = self.fvs.read()
            
            undistorted = cv2.undistort(self.frame, self.gaze_estimator.camera.camera_matrix, self.gaze_estimator.camera.dist_coefficients)
            faces = self.gaze_estimator.detect_faces(undistorted)
            head_position = [None, None, None]
            gaze_vector = [None, None, None]
            if(len(faces) == 0):
                if(self.save_csv): 
                    self.gaze_data_df.loc[len(self.gaze_data_df)] = {'frame': str(self.frame_number), 'head_x': None, 'head_y': None, 'head_z': None, 'gaze_x': None, 'gaze_y': None, 'gaze_z': None}
            else:
                face=faces[0]
                self.gaze_estimator.estimate_gaze(undistorted, face)
                head_position = face.head_position
                gaze_vector = face.gaze_vector
                if(self.save_csv): 
                    self.gaze_data_df.loc[len(self.gaze_data_df)] = {'frame': str(self.frame_number), 'head_x': face.head_position[0], 'head_y': face.head_position[1], 'head_z': face.head_position[2], 'gaze_x': face.gaze_vector[0], 'gaze_y': face.gaze_vector[1], 'gaze_z': face.gaze_vector[2]}
            self.frame_number += 1
            self.last_frame = [head_position, gaze_vector]
            self.continue_processing = False
            
        print('out', self.stream)
        self.fvs.stop()
        self.isRunning = False
        if(self.save_csv): 
            self.gaze_data_df.to_csv(self.csv_filename, index=False)

