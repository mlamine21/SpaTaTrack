from roomConfiguration import RoomConfiguration
from JointGazeHero import JointGazeHero

import time
import sys
from gazeProcessor import GazeProcessor
from participant import Participant
from topView import TopView
from sideView import SideView
from sideViewOther import SideViewOther
import numpy as np
import os
import cv2
from collections import Counter
import threading

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


def computeStaticViews():
    top_view.computeStaticView()
    side_view.computeStaticView()
    side_view_other.computeStaticView()


# Define scale factor for the views, the higher the scale factor, the larger the image (heavier processing)
scale_factor=200

# Initialize the room configuration (scene size, objects, cameras)
room_config = RoomConfiguration()
# Initialize the views using the room configuration
top_view = TopView(scale_factor=scale_factor, room_config=room_config)
side_view = SideView( scale_factor=scale_factor, room_config=room_config)
side_view_other = SideViewOther(scale_factor=scale_factor, room_config=room_config)

computeStaticViews()

combined_image = np.vstack((side_view.image, top_view.image, side_view_other.image))


# Initialize the gaze processor containing the gaze estimator model and the video stream

# The video streams specification is stored in a dict
# Each field represents a stream, and should be named as one of the cameras in the config.json file
# In this example we used a computer with an embedded camera (stream 0), and plugged in two additional cameras (stream 1 and stream 2).
# It is possible to process videos instead of camera streams, by directly writing the path to the video file(s) 
video_streams = {
    "user1": 0, 
    # "user2": 2
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

# ------------------- Integrate GUI and monitor -------------------

# Create GUI instance (in main thread)
joint_gaze_hero = JointGazeHero(room_config=room_config, midi_dir="midis", config_midi="config_midi.json")

# Shared monitor state
monitor_state = {
    "expected_name": None,
    "expected_index": None,
    "first_seen": None,
    "reported": False,
    "running": True
}

# Called by GUI when it highlights and requests input
def on_request_input(index):
    try:
        name = joint_gaze_hero.interaction_objects[index]["name"]
    except Exception:
        name = None
    monitor_state["expected_name"] = name
    monitor_state["expected_index"] = index
    monitor_state["first_seen"] = None
    monitor_state["reported"] = False

joint_gaze_hero.on_request_input = on_request_input

# Optional: hook to play success sound (user may set this later)
def play_success_sound():
    # placeholder: user can assign joint_gaze_hero.on_success_sound = custom_func
    pass

# Report success from GUI thread
def report_success_on_gui():
    idx = monitor_state["expected_index"]
    try:
        if idx is not None and 0 <= idx < len(joint_gaze_hero.square_frames):
            frame, lbl = joint_gaze_hero.square_frames[idx]
            frame.config(highlightbackground="green", highlightthickness=6)
    except Exception:
        pass
    try:
        if hasattr(joint_gaze_hero, "on_success_sound") and callable(joint_gaze_hero.on_success_sound):
            joint_gaze_hero.on_success_sound()
        else:
            play_success_sound()
    except Exception:
        pass
    try:
        joint_gaze_hero.on_input_result(True)
    except Exception:
        pass

# Background consensus monitor
def consensus_monitor():
    poll = 0.05
    stable_required = 0.5
    while monitor_state["running"]:
        # collect latest targets
        targets = []
        for p in participants:
            if getattr(p, "targets", None):
                targets.append(p.targets[-1])
        if not targets or monitor_state["expected_name"] is None or monitor_state["reported"]:
            monitor_state["first_seen"] = None
            time.sleep(poll)
            continue
        # majority vote
        counter = Counter(targets)
        target, count = counter.most_common(1)[0]
        if target == monitor_state["expected_name"]:
            if monitor_state["first_seen"] is None:
                monitor_state["first_seen"] = time.time()
            else:
                if time.time() - monitor_state["first_seen"] >= stable_required:
                    monitor_state["reported"] = True
                    # schedule GUI callback on main thread
                    try:
                        joint_gaze_hero.root.after(0, report_success_on_gui)
                    except Exception:
                        pass
        else:
            monitor_state["first_seen"] = None
        time.sleep(poll)

monitor_thread = threading.Thread(target=consensus_monitor, daemon=True)
monitor_thread.start()

# Run the gaze processing loop in a separate thread so Tk mainloop can run in main thread
exit_event = threading.Event()

def gaze_processing_loop():
    highlighted = False
    frame_count = 0
    start_time = time.time()
    while not exit_event.is_set() and getState(gaze_processors) != False:
        curr_time = time.time()
        frame = {}
        # We extract the last frame from each gaze processor
        for stream, gp in gaze_processors.items():
            # Wait until a first frame is available
            while (gp.getLastFrame() is None) and not exit_event.is_set():
                time.sleep(0.01)
            frame[stream] = gp.getLastFrame() if gp.isRunning else [None,None,None,None,None,None]
        # Update participants with new frames
        for stream, gp in gaze_processors.items():
            if int(stream) == 0:
                users[stream].updateFrame(head_position=frame[stream][0], gaze_vector=frame[stream][1])
            else:
                gv = [frame[stream][1][0], frame[stream][1][1], frame[stream][1][2]]
                users[stream].updateFrame(head_position=frame[stream][0], gaze_vector=gv)
        # Update views
        computeStaticViews()
        for i in range(len(participants)):
            target_top = top_view.update_view(participants[i], frame=len(participants[i].head_position)-1, refresh=False, overlay=True)
            target_side = side_view.update_view(participants[i], frame=len(participants[i].head_position)-1, refresh=False, overlay=True)
            target_side_other = side_view_other.update_view(participants[i], frame=len(participants[i].head_position)-1, refresh=False, overlay=True)
            targets = [target_top, target_side, target_side_other]
            target_counts = Counter(targets)
            target, count = target_counts.most_common(1)[0]
            if count < 2:
                target = "autre"
            participants[i].updateTargets(target)
        # Add dummy for demo behavior (keeps previous demo logic)
        participants.append(Participant(id="dummy", head_position=[1,2,3], gaze_vector=[1,2,3]))
        elapsed = curr_time - start_time
        if(elapsed<3):
            participants[-1].targets.append("box1_interaction")
        elif(elapsed>3 and elapsed<6):
            participants[-1].targets.append("box2_interaction")
        # Highlight logic (kept for visualization)
        all_targets = [p.targets[-1] for p in participants]
        if len(set(all_targets)) == 1:
            # Highlight target in room config
            if not highlighted:
                room_config.highlightObject(all_targets[0], (0, 255, 0))
                highlighted = True
        else:
            if highlighted:
                room_config.unhighlightObjects()
                highlighted = False
        participants.pop()
        combined_image = np.vstack((side_view.image, top_view.image, side_view_other.image))
        for gp in gaze_processors.values():
            gp.continue_processing = True
            cv2.imshow("face"+str(gp.stream), gp.frame)
        cv2.imshow("gaze viz",combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # signal exit
            for gp in gaze_processors.values():
                gp.continue_processing = False
                try:
                    gp.fvs.stop()
                except Exception:
                    pass
                gp.stop = True
            exit_event.set()
            break
        frame_count += 1
        if curr_time - start_time >= 1.0:
            fps = frame_count / (curr_time - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = curr_time
    # cleanup on exit
    for i in range(len(participants)):
        participants[i].toCsv(participants[i].id+'.csv')
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    for gp in gaze_processors.values():
        gp.join()

processing_thread = threading.Thread(target=gaze_processing_loop, daemon=True)
processing_thread.start()

# Finally start the GUI mainloop in the main thread
try:
    joint_gaze_hero.run()
finally:
    # stop monitor and processing on GUI exit
    monitor_state["running"] = False
    exit_event.set()
    try:
        processing_thread.join(timeout=1.0)
    except Exception:
        pass
    try:
        monitor_thread.join(timeout=1.0)
    except Exception:
        pass
