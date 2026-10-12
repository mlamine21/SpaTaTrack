import numpy as np
import csv
class Participant():
  
  def __init__(self, id, head_position, gaze_vector):
    """
    Initialize the Participant with camera and room configuration.
    
    :param camera: Camera object to be used for capturing images.
    :param room_config: Configuration of the room containing scene size and axes.
    :param scale_factor: Factor to scale the scene size.
    """
    self.id = id
    self.head_position = head_position
    self.gaze_vector = gaze_vector
    self.targets = []
    self.camera = None  # Placeholder for camera object
    self.last_known_position_frame = None
    self.offset_y_gaze_vec = 0

  def setCamera(self, camera):
    """
    Set the camera for the participant.
    
    :param camera: Camera object to be used for capturing images.
    """
    self.camera = camera

  def updateFrame(self, head_position = [None,None,None], gaze_vector = [None,None,None]):
      self.head_position.append(head_position)
      self.gaze_vector.append(gaze_vector)
    
      return len(self.head_position)-1 

  def updateTargets(self, target):
    self.targets.append(target)

  def getHeadPosition(self, frame=0):
    try:
      # Invert the y and z coordinates for the yellow participant
      return [self.head_position[frame][0], -self.head_position[frame][1], -self.head_position[frame][2]]
    except (TypeError, IndexError):
      return [None, None, None]
    
  def getHeadPositionFromObject(self, object_position, frame=0):
    try:
      # Invert the y and z coordinates for the yellow participant
      return self.getHeadPosition(frame) + np.array(object_position)
    except (TypeError, IndexError):
      return [None, None, None]

  
  def getGazeVector(self, frame=0):
    try : 
      return [self.gaze_vector[frame][0]*10, -self.gaze_vector[frame][1]*10 + self.offset_y_gaze_vec, -self.gaze_vector[frame][2]*10]
    except (TypeError, IndexError):
      return [None, None, None]
    
  def toCsv(self, path):
    header = ['head_x', 'head_y', 'head_z', 'gaze_x', 'gaze_y', 'gaze_z', 'target']
    with open(path, 'w') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(header)
      for i in range(len(self.head_position)):
        line = list(map(str, self.head_position[i])) + list(map(str, self.gaze_vector[i])) + [str(self.targets[i])]
        writer.writerow(line)