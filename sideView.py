import numpy as np
import os
import cv2
from ptgaze.common import Face
from roomConfiguration import RoomConfiguration
from templateView import TemplateView

class SideView(TemplateView):

  def __init__(self, scale_factor=500, room_config=None, display=True):
    super().__init__(scale_factor, room_config, x_axis=3, y_axis=2,z_axis=1, display=display)
    self.name = "side"
  def getViewObjectPosition(self, object_position, sign=1):
    """Get the position of an object in the view : Objects in roomConfiguration are defined according to the camera position."""
    # Performance optimization: cache transformations
    cache_key = f"{object_position}_{sign}"
    if hasattr(self, '_object_position_cache') and cache_key in self._object_position_cache:
        return self._object_position_cache[cache_key]
        
    x= self.center_coords[self.x_axis] + int(object_position[self.x_axis] * self.scale_factor) 
    y = self.center_coords[self.y_axis] - int(object_position[self.y_axis] * self.scale_factor)
    z = self.center_coords[self.z_axis] - int(object_position[self.z_axis] * self.scale_factor)
    
    # if x or y is bigger than the image size, return 0
    # if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
    #   return (0, 0, 0)
    
    result = (x, y, z)
    if not hasattr(self, '_object_position_cache'):
        self._object_position_cache = {}
    self._object_position_cache[cache_key] = result
    return result

if __name__ == "__main__":
    def nothing(x):
        pass
    # Example usage
    cv2.namedWindow("Side View")
    room_config = RoomConfiguration()
    side_view = SideView(scale_factor=500, room_config=room_config, display=True)
    face = Face(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    face.head_position = np.array([ 0.0041763, -0.02742862, 0.62311349])
    face.gaze_vector = np.array([ 0.27407838 , 0.6095285 , -2.77341565])
    side_view.update_view(face=face)
    
    

    """Update the view by drawing the table."""

    # print("Image shape:", side_view.image.shape)
    # print("Center coordinates:", (side_view.center_y, side_view.center_z))
    # print("Camera size in pixels:", side_view.camera_size_px)
    cv2.imshow("Side View", side_view.image)
    cv2.waitKey(0)


    

    # # Create trackbars for gaze vector components (range and initial value can be adjusted)
    # cv2.createTrackbar('Gaze X', 'Side View', 0, 200, nothing)
    # cv2.createTrackbar('Gaze Y', 'Side View', 0, 200, nothing)
    # cv2.createTrackbar('Gaze Z', 'Side View', 100, 200, nothing)  # Centered at 0

    # while True:
    #     # Read trackbar positions and map them to a suitable range (e.g., -1 to 1 or -3 to 3)
    #     gx = (cv2.getTrackbarPos('Gaze X', 'Side View') - 100) / 30.0
    #     gy = (cv2.getTrackbarPos('Gaze Y', 'Side View') - 100) / 30.0
    #     gz = (cv2.getTrackbarPos('Gaze Z', 'Side View') - 100) / 30.0

    #     face.gaze_vector = np.array([gx, gy, gz])

    #     side_view.update_view(face=face)
    #     cv2.imshow("Side View", side_view.image)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cv2.destroyAllWindows()



