import numpy as np
import os
import cv2
from ptgaze.common import Face
from roomConfiguration import RoomConfiguration
from templateView import TemplateView

class TopView(TemplateView):

  def __init__(self, scale_factor=500, room_config=None, display=True):
    super().__init__(scale_factor, room_config, x_axis=1, y_axis=3, z_axis=2, display=display)
    self.name = "top"
  def getViewObjectPosition(self, object_position, sign = 1):
    """Get the position of an object in the view : Objects in roomConfiguration are defined according to the camera position."""
    # Performance optimization: cache transformations
    cache_key = f"{object_position}_{sign}"
    if hasattr(self, '_object_position_cache') and cache_key in self._object_position_cache:
        return self._object_position_cache[cache_key]
        
    x= self.center_coords[self.x_axis] - int(object_position[self.x_axis] * self.scale_factor)
    y = self.center_coords[self.y_axis] - int(object_position[self.y_axis] * self.scale_factor)
    z = self.center_coords[self.z_axis] + sign*int(object_position[self.z_axis] * self.scale_factor)
    # if x or y is bigger than the image size, return 0
    # if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
    #   return (0, 0, 0)
    
    result = (x, y, z)
    if not hasattr(self, '_object_position_cache'):
        self._object_position_cache = {}
    self._object_position_cache[cache_key] = result
    return result
  

  

if __name__ == "__main__":
    # Example usage
    top_view = TopView("rouge", scale_factor=500)
    face = Face(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    face.head_position = np.array([ 0.0041763, -0.02742862, 0.62311349])
    face.gaze_vector = np.array([ 0.27407838 , 0.6095285 , -2.77341565])

    # top_view.draw_axes()
    # top_view.draw_table()
    # top_view.draw_screen()
    # top_view.draw_camera()
    top_view.update_view(face=face)

    # top_view.update_view(face=face)
    # print("Image shape:", top_view.image.shape)
    # print("Center coordinates:", (top_view.center_x, top_view.center_y))
    # print("Camera size in pixels:", top_view.camera_size_px)

    cv2.imshow("Top View", top_view.image)

    # resized = cv2.resize(top_view.image, (800, 500))
    # cv2.imshow("Side View", resized)
    cv2.waitKey(0)
  
  

