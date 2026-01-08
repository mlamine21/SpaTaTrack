import sys
from gazeProcessor import GazeProcessor
from roomConfiguration import RoomConfiguration
from participant import Participant
from topView import TopView
from sideView import SideView
from sideViewOther import SideViewOther
import numpy as np
import os
import cv2
import time
from collections import Counter

# Check if the gaze processors are running
def getState(gaze_processors):
    state = {}
    for stream, gp in gaze_processors.items():
        state[stream] = gp.isRunning
    if( all(value == True for value in state.values())):
        return True
    elif all(value == False for value in state.values()):
        return False
    else: # returns the stream names that are not running
        mask = []
        for stream, running in state.items():
            if not running:
                mask.append(stream)
        return mask
    
# Define scale factor for the views, the higher the scale factor, the larger the image (heavier processing)
scale_factor=200

# Initialize the room configuration (scene size, objects, cameras)
room_config = RoomConfiguration()
# Initialize the views using the room configuration
top_view = TopView(scale_factor=scale_factor, room_config=room_config)
side_view = SideView( scale_factor=scale_factor, room_config=room_config)
side_view_other = SideViewOther(scale_factor=scale_factor, room_config=room_config)

top_view.computeStaticView()
side_view.computeStaticView()
side_view_other.computeStaticView()

combined_image = np.vstack((side_view.image, top_view.image, side_view_other.image))


# Initialize the gaze processor containing the gaze estimator model and the video stream

# The video streams specification is stored in a dict
# Each field represents a stream, and should be named as one of the cameras in the config.json file
video_streams = {
    "user1": 0,
    "user2": "video.mkv"
}
gaze_processors = {}
users = {}

for user, stream in video_streams.items():
    # Initialize GazeProcessor object (one for each stream)
    gaze_processors[stream] = GazeProcessor(stream=stream)
    # Initialize Participant object (used to store head positions and gaze vectors)
    users[stream] = Participant(id=str(user), head_position=[], gaze_vector=[])

# Add participants to the room configuration (each participant is linked to a gaze processor and a video stream)
participants = [users[list(video_streams.values())[i]] for i in range(len(video_streams))]
room_config.setParticipants(participants)

# Init the gaze processors and load the models (face detection and gaze estimation) 
for gp in gaze_processors.values():
    print("starting gaze processor for stream:", gp.stream)
    gp.start()
print("started")

ready=False
while(not ready):
    state = getState(gaze_processors)
    if(isinstance(state, bool) and state==True):
        print("all running")
        ready=True
    else:
        print("waiting for streams:", state)
    time.sleep(1)


for gp in gaze_processors.values():
    gp.continue_processing = True

frame_count = 0
start_time = time.time()
while getState(gaze_processors)!=False:
    curr_time = time.time()
    frame = {}
    # We extract the last frame from each gaze processor
    for stream, gp in gaze_processors.items():
        # But first we wait until a first frame is available (normally only once)
        while (gp.getLastFrame() is None):
            print('waiting frame for ', stream)
            time.sleep(0.01)
        frame[stream] = gp.getLastFrame() if gp.isRunning else [None,None,None,None,None,None]
        

    # Update each participant with the new head position and gaze vector
    for stream, gp in gaze_processors.items():
        users[stream].updateFrame(head_position=frame[stream][0], gaze_vector=frame[stream][1])
    
    # Compute static views before adding dynamic elements (participants)
    top_view.image = top_view.static_view.copy()
    side_view.image = side_view.static_view.copy()
    side_view_other.image = side_view_other.static_view.copy()         
    for i in range(len(participants)):
        # Extract new position in each view for each participant, and retrieve target for each view
        target_top = top_view.update_view(participants[i], frame=len(participants[i].head_position)-1, refresh=False, overlay=True)
        target_side = side_view.update_view(participants[i], frame=len(participants[i].head_position)-1, refresh=False, overlay=True)
        target_side_other = side_view_other.update_view(participants[i], frame=len(participants[i].head_position)-1, refresh=False, overlay=True)
        targets = [target_top, target_side, target_side_other]
          
        # Majority vote for target (should be the same in all views ideally)
        target_counts = Counter(targets)
        target, count = target_counts.most_common(1)[0]
        # If no consensus, default to "autre"
        if count < 2:
            target = "autre"

    combined_image = np.vstack((side_view.image, top_view.image, side_view_other.image))
    # print(gp.fvs.last_frame)
    for gp in gaze_processors.values():
        gp.continue_processing = True
        cv2.imshow("face"+str(gp.stream), gp.frame)
    cv2.imshow("gaze viz",combined_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        for gp in gaze_processors.values():
            gp.continue_processing = False
            gp.fvs.stop()
            gp.stop = True
        break

    frame_count += 1
    if curr_time - start_time >= 1.0:
        fps = frame_count / (curr_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = curr_time

cv2.destroyAllWindows()
for gp in gaze_processors.values():
    gp.join()
