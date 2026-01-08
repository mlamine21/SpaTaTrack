import numpy as np
import os
import cv2
from ptgaze.common import Face
from roomConfiguration import RoomConfiguration
from templateView import TemplateView

class SideViewOther(TemplateView):

  def __init__(self,  scale_factor=500, room_config=None, display=True):
    super().__init__( scale_factor, room_config, x_axis=1, y_axis=2,z_axis=3, display=display)
    self.name = "side_other"
  def getViewObjectPosition(self, object_position, sign=1):
    """Get the position of an object in the view : Objects in roomConfiguration are defined according to the camera position."""
    x= self.center_coords[self.x_axis] - int(object_position[self.x_axis] * self.scale_factor)
    y = self.center_coords[self.y_axis] - int(object_position[self.y_axis] * self.scale_factor)
    z = self.center_coords[self.z_axis] + int(object_position[self.z_axis] * self.scale_factor)
    # if x or y is bigger than the image size, return 0
    # if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
    #   return (0, 0, 0)
    return (x, y, z)