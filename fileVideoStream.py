import os
import cv2
import numpy as np
from threading import Thread
from queue import Queue
import time


class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        os.environ[ "OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"
        self.stream = cv2.VideoCapture(path)#, cv2.CAP_FFMPEG)
        self.frame_count = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
    
    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        # t.daemon = True
        t.start()
        return self
    
    def get_frame_count(self):
        # return the total number of frames in the video file
        return int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def update(self):
        # keep looping infinitely
        while True:
            time.sleep(0.001)
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)
    def read(self):
        # return next frame in the queue
        return self.Q.get()   

    def more(self):
        # return True if there are still frames in the queue
        return not self.stopped 

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

if __name__ == "__main__":
    import sys
    print(cv2.CAP_DSHOW)

    fvs = FileVideoStream(0).start()
    if not fvs.stream.isOpened():
        print("Error: Could not open video.")
        sys.exit()
    while(fvs.more()):
        print("hzere")
        frame = fvs.read()
        print(frame)
        # process the frame (e.g., display it, save it, etc.)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    fvs.stop()