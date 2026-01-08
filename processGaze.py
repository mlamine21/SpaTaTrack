import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.simplefilter(action='ignore', category=FutureWarning)

from threading import Thread
import numpy as np
from omegaconf import OmegaConf
from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.common import Visualizer
from ptgaze.utils import get_3d_face_model
import gazeUtils
from topView import TopView
from sideView import SideView
from sideViewOther import SideViewOther
from roomConfiguration import RoomConfiguration
import sys
from participant import Participant
import time
from fileVideoStream import FileVideoStream
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import cv2

class FaceProcessor(Thread):
    def __init__(self, video_path, color, display=True):
        # print("CUDA-enabled devices:", cv2.cuda.getCudaEnabledDeviceCount())
        # if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        #     print("OpenCV is built with CUDA support and can use the GPU.")
        # else:
        #     print("OpenCV is NOT built with CUDA support or no CUDA device is available.")

        # print("Initializing FaceProcessor...")
        super().__init__()
        self.display = display
        self.video_path = video_path
        self.color = color

        gazeUtils.initCameraParameters()
        self.initGazeEstimator()
        print("FaceProcessor initialized successfully.")

        self.room_config = RoomConfiguration()
        self.top_view = TopView(scale_factor=500, room_config=self.room_config)
        self.side_view = SideView(scale_factor=500, room_config=self.room_config)
        self.side_view_other = SideViewOther(scale_factor=500, room_config=self.room_config)
        self.mode = video_path
        if(self.mode != 'Camera'):
            self.gaze_data_df = pd.DataFrame(columns=['frame', 'head_x', 'head_y', 'head_z', 'gaze_x', 'gaze_y', 'gaze_z'])
            # self.csv_file = csv.writer(open(f"{video_path[:video_path.index('.')]}.gaze_data.csv", "w", newline=""))
            # self.csv_file.writerow(['frame','head_x', 'head_y', 'head_z', 'gaze_x', 'gaze_y', 'gaze_z'])

    def initGazeEstimator(self):
         # Resolve np problems
        # https://github.com/drewp/pytorch_mpiigaze_demo/commit/e30d064cbe9f00ed6c8239f2a7cb1ff9d29e9867
        # Temporarily suppress stderr during GazeEstimator instantiation
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            self.config = OmegaConf.load('eth-xgaze.yaml')
            self.gaze_estimator = GazeEstimator(self.config)  # Warning occurs here
        finally:
            # Always restore stderr
            sys.stderr.close()
            sys.stderr = original_stderr
    
        if self.display:
            face_model_3d = get_3d_face_model(self.config)
            self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

    
    def run(self):
        participantInRange = False
        # print("Starting face detection...")
        if self.video_path == 'Camera':
            print("Using camera...")
            self.fvs = FileVideoStream(0).start()
        else:
            self.fvs = FileVideoStream(self.video_path).start()
        
        # self.fvs.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        # self.fvs.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        # sys.exit()
        if not self.fvs.stream.isOpened():
            print("Error: Could not open video.")
            return
        idx=-1
        self.top_view.computeStaticView()
        self.side_view.computeStaticView()
        self.side_view_other.computeStaticView()
        while self.fvs.more():
            idx+=1
            # Print progress every 10% of the video
            if(self.mode != 'Camera' and self.fvs.get_frame_count() > 0):
                if(idx % int(self.fvs.get_frame_count()/10) == 0):
                    print(10*int(idx/int(self.fvs.get_frame_count()/10)), end="%...")
            frame = self.fvs.read()
            
            # h, w = frame.shape[:2]
            
            undistorted = cv2.undistort(frame, self.gaze_estimator.camera.camera_matrix, self.gaze_estimator.camera.dist_coefficients)
            if self.display  : self.visualizer.set_image(frame.copy())
            faces = self.gaze_estimator.detect_faces(undistorted)
            resized = None
            if(len(faces) == 0):
                if(self.mode != 'Camera'): 
                    # self.csv_file.writerow([str(idx),None, None, None, None, None, None])
                    self.gaze_data_df.loc[len(self.gaze_data_df)] = {'frame': str(idx), 'head_x': None, 'head_y': None, 'head_z': None, 'gaze_x': None, 'gaze_y': None, 'gaze_z': None}
            else:
                face=faces[0]
                self.gaze_estimator.estimate_gaze(undistorted, face)

                if(self.mode != 'Camera'): 
                    # self.csv_file.writerow([str(idx),face.head_position[0], face.head_position[1], face.head_position[2], face.gaze_vector[0], face.gaze_vector[1], face.gaze_vector[2]])
                    self.gaze_data_df.loc[len(self.gaze_data_df)] = {'frame': str(idx), 'head_x': face.head_position[0], 'head_y': face.head_position[1], 'head_z': face.head_position[2], 'gaze_x': face.gaze_vector[0], 'gaze_y': face.gaze_vector[1], 'gaze_z': face.gaze_vector[2]}
                if self.display : 
                    self.visualizer.draw_bbox(face.bbox)
                    length = self.config.demo.gaze_visualization_length
                    gazeUtils.draw_gaze_vector(self.visualizer, face, length)
                    distance = gazeUtils.estimate_distance(face.bbox, self.gaze_estimator.camera.camera_matrix)

                    cv2.putText(self.visualizer.image, f"In range : {distance:.1f} cm", (face.bbox[0][0], face.bbox[0][1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # angle = self.room_config.angles_to_vector(face.gaze_vector[0], face.gaze_vector[1]) if self.color in ["rouge", "vert"] else self.room_config.angles_to_vector(face.gaze_vector[2], face.gaze_vector[0])
                    participant = Participant(self.color, head_position=[[face.head_position[0], face.head_position[1], face.head_position[2]]], gaze_vector=[[face.gaze_vector[0], face.gaze_vector[1], face.gaze_vector[2]]])
                    self.room_config.setParticipants([participant])
                    # target = self.get_gaze_target(face)                
                    # Update top and side views with face data
                    self.top_view.update_view(participant)
                    self.side_view.update_view(participant)

                    resized = cv2.resize(self.visualizer.image, (600, 500))
                    
                if self.display and (resized is not None):
                    # resized = resized[:,::-1]
                    
                    # Resize views to match each other - ensure both are exactly the same size
                    top_view_resized = cv2.resize(self.top_view.image, (600, 500))
                    side_view_resized = cv2.resize(self.side_view.image, (600, 300))
                    
                    # Create a combined visualization of the views
                    views_combined = np.hstack((resized, top_view_resized))
                    
                    # Resize the combined views to match the width of the main display
                    views_combined_resized = cv2.resize(views_combined, (resized.shape[1], int(resized.shape[1] * views_combined.shape[0] / views_combined.shape[1])))
                    
                    # Stack the main view and the combined views
                    full_display = np.vstack((views_combined_resized,side_view_resized))
                    
                    # Show the full display
                    cv2.imshow('Gaze Visualization', full_display)
                    
                    # k = cv2.waitKey(1)
                    # if k == ord("q"):
                    #     print("Quitting...")
                    #     break
                    
                    # if keyboard.is_pressed("q"):
                    #     print("Quitting...")
                    #     break
                
        self.fvs.stop()
        self.gaze_data_df.to_csv(f"{self.video_path[:self.video_path.index('.')]}.gaze_data.csv", index=False)

        cv2.destroyAllWindows()
        print("Face detection stopped.")


if __name__ == "__main__":
    mode="camera"
    
    
    if(mode == "groups"):
        for group in range(1,18):
            g = str(group).zfill(2)
            start_time = time.time()
            for session in range(1,3):
                s = str(session)
                # g='01'
                # s='2'
                threads = {"bleu": None, "rouge": None, "vert": None, "jaune": None}
                # print(f"Processing group {g}, session {s}")
                for participant in ["vert","bleu","jaune","rouge"]:
                    # print(participant)
                    # participant = "jaune" 
                    if(not os.path.exists(f"D:/Work Amine/COMPHAR/G{g}_S{s}/{participant}.gaze_data.csv") or os.path.getsize(f"D:/Work Amine/COMPHAR/G{g}_S{s}/{participant}.gaze_data.csv") == 0):
                        path = f"D:/Work Amine/COMPHAR/G{g}_S{s}/"
                        video = f"{participant}.video.mp4"

                        threads[participant] = FaceProcessor(video_path=path+video, color=participant, display=False) 
                        # main = FaceProcessor(video_path="Camera", participant=participant) 
                        threads[participant].start()
                        # threads[participant].join()
                    else:
                        print(f"Skipping {participant} in group {g}, session {s} : data already exists.")

                for participant in ["vert","bleu","jaune","rouge"]:
                    if threads[participant] is not None:
                        threads[participant].join()
                        print(f"{participant} processing completed.")
                print(f"Group {g}, session {s} processed in {time.time() - start_time:.2f} seconds.")
            
    else:    
    # path = "D:/Work Amine/COMPHAR/G01_S1/"
        participant = "bleu"
    # video = f"{participant}.video.mp4"
    # main = FaceProcessor(video_path=path+video, participant=participant, display=False) 
        main = FaceProcessor(video_path="Camera", color=participant) 
        main.start()

