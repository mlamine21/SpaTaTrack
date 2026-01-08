import csv
from datetime import datetime
from roomConfiguration import RoomConfiguration
from participant import Participant
import sys
import os
import cv2
from sideView import SideView
from sideViewOther import SideViewOther
from topView import TopView
import numpy as np
import time
import tqdm
from collections import Counter
path_prepend = "D:/WORK AMINE/COMPHAR"
# path_prepend = "COMPHAR_GAZE"


scale_factor=230
room_config = RoomConfiguration()
top_view = TopView(scale_factor=scale_factor, room_config=room_config)
side_view = SideView( scale_factor=scale_factor, room_config=room_config)
side_view_other = SideViewOther(scale_factor=scale_factor, room_config=room_config)
def nothing(x):
  return 

changed_angle = False
def changed(x):
   global changed_angle
   changed_angle = True

def mergeSuccessiveBlocks( blocks):
    new_blocks = []
    values=[]
    curr_idx = 0
    while(curr_idx< len(blocks) and curr_idx+1< len(blocks)):
        start=blocks[curr_idx]['start']
        values.append(blocks[curr_idx]['value']) 
        while(curr_idx+1< len(blocks) and blocks[curr_idx+1]['start'] - blocks[curr_idx]['end'] < 1 and blocks[curr_idx+1]['value'] == blocks[curr_idx]['value']):
            curr_idx+=1
            values.append(blocks[curr_idx]['value']) 
        new_blocks.append({'start':start, 'end':blocks[curr_idx]['end'],'value':max(values)})
        curr_idx+=1
        values=[]
    return new_blocks

def mergeSuccessiveBlocksOld( blocks):
    new_blocks = []
    values=[]
    curr_idx = 0
    while(curr_idx< len(blocks) and curr_idx+1< len(blocks)):
        start=blocks[curr_idx]['start']
        values.append(blocks[curr_idx]['value']) 
        while(curr_idx+1< len(blocks) and blocks[curr_idx+1]['start'] - blocks[curr_idx]['end'] < 1):
            curr_idx+=1
            values.append(blocks[curr_idx]['value']) 
        new_blocks.append({'start':start, 'end':blocks[curr_idx]['end'],'value':max(values)})
        curr_idx+=1
        values=[]
    return new_blocks

def deleteBriefBlocks(blocks, min_duration=1):
    new_blocks = []
    for block in blocks:
        if(block['end']-block['start']>min_duration):
            new_blocks.append(block)
    return new_blocks

def signalToAnnotation(signal, path, fps=55,merge_successive_blocks=True, delete_brief_blocks=True):
    curr_idx = 0
    blocks=[]
    dict_item_values = {"bleu":"0","rouge":"1","vert":"2","jaune":"3","ecran":"4","table":"5","autre":"6"}
    while(curr_idx < len(signal)):
      block = {'start':0,'end':0,'value':0}
      value = dict_item_values[signal[curr_idx]] 
      block['value'] = value
      block['start']=curr_idx/fps
      while(curr_idx<len(signal) and dict_item_values[signal[curr_idx]]==value):
          curr_idx+=1
      block['end']=(curr_idx-1)/fps
      blocks.append(block)
    if(merge_successive_blocks):
        blocks = mergeSuccessiveBlocks(blocks)
    if(delete_brief_blocks):
        blocks = deleteBriefBlocks(blocks, min_duration=0.02)
    with open(path, ('w')) as annotation:
        for block in blocks:
            annotation.write(str(block['start'])+';'+str(block['end'])+';'+str(block['value'])+';1\n')

if __name__ == "__main__":
  single_annotation = ('02','1')
  # single_annotation=False

  print('lets go ', datetime.now().strftime("%H:%M"))
  for group in range(1,18):
    g = str(group).zfill(2)
    start_time = time.time()
    for session in range(1,3):
      s = str(session)
      if(single_annotation):
        g = single_annotation[0]
        s = single_annotation[1]
      # display_video = False
      # display_animation = False
      # save_video = True
      # save_annotations = True
      
      display_video = True
      display_animation = True
      save_video = False
      save_annotations = False

      start_time = time.time()
      # side_view = SideView("jaune", scale_factor=scale_factor, room_config=room_config)
      
      participants = {"bleu": None, "rouge": None, "vert": None, "jaune": None}
      caps = {"bleu": None, "rouge": None, "vert": None, "jaune": None}
      annotations = {"bleu": [], "rouge": [], "vert": [], "jaune": []}
      
      fps = None
      for color in ["bleu", "rouge", "vert", "jaune"]:
        path = f"{path_prepend}/G{g}_S{s}/{color}.gaze_data.csv"
        video = f"{path_prepend}/G{g}_S{s}/{color}.video.mp4"
        
        caps[color] = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
        fps = caps[color].get(cv2.CAP_PROP_FPS)
        # fps=60
        
        nb_frames = caps[color].get(cv2.CAP_PROP_FRAME_COUNT)
        print(nb_frames)
        if not caps[color].isOpened():
          print(f"Error opening video file {video}")
          # fps=55
          # sys.exit(1)
          
        if(save_video):
          if(display_video):
            frame_width, frame_height = int(caps[color].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[color].get(cv2.CAP_PROP_FRAME_HEIGHT))
          else:
            frame_height = side_view.image.shape[0] + top_view.image.shape[0] + side_view_other.image.shape[0]
            frame_width = side_view.image.shape[1] 

        reader = csv.reader(open(path, 'r'))
        header = next(reader)
        head_position = []
        gaze_vector = []
        for row in reader:
            try:
              head_position.append([float(row[1]), float(row[2]), float(row[3])])
              gaze_vector.append([float(row[4]), float(row[5]), float(row[6])])
              # head_position.append([float(row[0]), float(row[1]), float(row[2])])
              # gaze_vector.append([float(row[3]), float(row[4]), float(row[5])])
            except ValueError:
              head_position.append([None, None, None])
              gaze_vector.append([None, None, None])
        participants[color] = Participant(color, head_position, gaze_vector)
        #Correction of y component of gaze vector because it seems that the algorithm makes mistakes with participants with glasses on  
        # if(color == 'jaune' or color =='vert' or color == 'rouge') : participants[color].offset_y_gaze_vec = -2
      if(save_video): 
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # out = cv2.VideoWriter(f"G{g}_S{s}_gaze.avi", fourcc, fps, (frame_width, frame_height), (cv2.VIDEOWRITER_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
        out = cv2.VideoWriter(f"G{g}_S{s}_gaze.mp4", fourcc, fps, (frame_width, frame_height))
        # if out.isOpened():
        #   print("Video writer opened successfully")
        # else:
        #   print("Failed to open video writer")
        #   sys.exit(1)

      room_config.setParticipants(participants.values())
      play=False
      if(display_animation):
        play = True
        cv2.namedWindow("Gaze Viewer", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('Frame', 'Gaze Viewer', 0, len(head_position), nothing)
        cv2.setTrackbarMax('Frame', 'Gaze Viewer', len(head_position)-1)

        cv2.namedWindow("Angle Viewer", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('bleu', 'Angle Viewer', room_config.ANGLE_CAMERA_BLEU, 45, changed)
        cv2.createTrackbar('rouge', 'Angle Viewer', room_config.ANGLE_CAMERA_ROUGE, 45, changed)
        cv2.createTrackbar('vert', 'Angle Viewer', room_config.ANGLE_CAMERA_VERT, 45, changed)
        cv2.createTrackbar('jaune', 'Angle Viewer', room_config.ANGLE_CAMERA_JAUNE, 45, changed)

      top_view.computeStaticView()
      side_view.computeStaticView()
      side_view_other.computeStaticView()
      # for object in room_config.objects:
      #     top_view.draw_object(object)
      # top_view.draw_participant(participants["rouge"])

      idx=0
      end_idx = len(head_position)-1
      # print('head position', len(head_position))
      pbar = tqdm.tqdm(total=end_idx)
      # end_idx = 5000
      while idx< end_idx or play:
        if(save_video or save_annotations):
          # if(idx % int(len(head_position)/10) == 0):
          #     print(10*int(idx/int(len(head_position)/10)), end="%...")
          idx+=1
          pbar.update(1)
        if(display_animation):
          frame = cv2.getTrackbarPos('Frame', 'Gaze Viewer')
          room_config.ANGLE_CAMERA_BLEU = cv2.getTrackbarPos('bleu', 'Angle Viewer')
          room_config.ANGLE_CAMERA_ROUGE = cv2.getTrackbarPos('rouge', 'Angle Viewer')
          room_config.ANGLE_CAMERA_VERT = cv2.getTrackbarPos('vert', 'Angle Viewer')
          room_config.ANGLE_CAMERA_JAUNE = cv2.getTrackbarPos('jaune', 'Angle Viewer')

          if(changed_angle):
            changed_angle = False
            room_config.initCameras()
            room_config.setParticipants(participants.values())
            top_view.computeStaticView()
            side_view.computeStaticView()
            side_view_other.computeStaticView()

        if(display_video):
          frames = {"bleu": None, "rouge": None, "vert": None, "jaune": None}
          
          for color in ["bleu", "rouge", "vert", "jaune"]:
            caps[color].set(cv2.CAP_PROP_POS_FRAMES, frame)
            res, frames[color] = caps[color].read()
          
            ratio = frames[color].shape[1]/ frames[color].shape[0]
            height = side_view.image.shape[0] + top_view.image.shape[0] + side_view_other.image.shape[0]
            if(res):
              frames[color] = cv2.resize(frames[color], (int(ratio * height//4), height//4))
            else:
              #display a black image with shape (200, 200, 3)
              frames[color] = np.zeros((int(ratio * height//4), height//4, 3), dtype=np.uint8)
        if(not display_animation and not display_video):
          frame = idx 

        top_view.image = top_view.static_view.copy()
        side_view.image = side_view.static_view.copy()
        side_view_other.image = side_view_other.static_view.copy()         
        for color in participants:
          target_top = top_view.update_view(participants[color], frame=frame, refresh=False, overlay=True)
          target_side = side_view.update_view(participants[color], frame=frame, refresh=False, overlay=True)
          target_side_other = side_view_other.update_view(participants[color], frame=frame, refresh=False, overlay=True)
          targets = [target_top, target_side, target_side_other]
          
          # Majority vote for target
          target_counts = Counter(targets)
          target, count = target_counts.most_common(1)[0]
          # If no consensus, default to "autre"
          if count < 2:
              target = "autre"
          
          if(save_annotations):
            
            if('wall' not in target):
              if(target == 'table' or 'camera' in target):
                target = 'ecran'
              annotations[color].append(target)
            else:
              annotations[color].append("autre")

        combined_image = np.vstack((side_view.image, top_view.image, side_view_other.image))
        
        
        # Display the name of each participant on the left side of the video frame
        if(display_video):
          video_frame = np.vstack((frames["bleu"], frames["rouge"], frames["vert"], frames["jaune"]))
          for i, color in enumerate(["bleu", "rouge", "vert", "jaune"]):
            cv2.putText(video_frame, color.capitalize(), (10, (i+1) * height//4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if(display_video and display_animation):
            combined_image = np.hstack((video_frame, combined_image))
        if(display_animation):
          cv2.imshow("Gaze Viewer", combined_image)
        
        if(save_video):
          out.write(combined_image)
        if(display_animation or display_video):
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if(idx == 500):
        #   break
      if(display_video):
        caps["bleu"].release()
        caps["rouge"].release()
        caps["vert"].release()
        caps["jaune"].release()
      if(save_video):
        out.release()
      if(save_annotations):
          for color in annotations:
            # with open(f"{path_prepend}/G{g}_S{s}/{color}.targets.txt", 'w') as f:
            #   for target in annotations[color]:
            #       f.write(target + '\n')
            signalToAnnotation(annotations[color], f"{path_prepend}/G{g}_S{s}/{color}.gaze_target.annotation~", fps=fps, merge_successive_blocks=True, delete_brief_blocks=True)
      print(f"\nProcessing time: G{g}_S{s}: {time.time() - start_time:.2f} seconds")
      print(datetime.now().strftime("%H:%M"))
      if(single_annotation):
        sys.exit()