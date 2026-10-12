import numpy as np
import os
import cv2
from roomConfiguration import RoomConfiguration


class TemplateView:
    HIGHLIGHT_COLOR = (0, 255, 0)  # Green color for highlighting
    def __init__(self, scale_factor=500, room_config=None, x_axis = 1, y_axis=2,z_axis=3, display=True):
        if(room_config is not None):
            self.room_config = room_config
        else:
            self.room_config = RoomConfiguration()

        self.display = display  # Whether to display the image or not
        
        # Real-world scale factor (meters to pixels)
        self.scale_factor = scale_factor  # 1 pixel per centimeter

        self.x_axis = abs(x_axis)-1  # Horizontal axis in the room configuration (0 for x, 1 for y, 2 for z)
        self.y_axis = abs(y_axis)-1
        self.z_axis = abs(z_axis)-1  # The remaining axis is the vertical axis
        
        
        self.horizontal_length = int(self.room_config.scene_size[self.x_axis]*self.scale_factor)  # Length of the view (horizontal) in pixels 
        self.vertical_length = int(self.room_config.scene_size[self.y_axis]*self.scale_factor)  # Length of the view (vertical) in pixels 
        self.depth_length = int(self.room_config.scene_size[self.z_axis]*self.scale_factor)  # Length of the view (depth) in pixels

        
        self.image = np.ones((self.vertical_length, self.horizontal_length, 3), dtype=np.uint8) * 255

        self.center_coords = [0,0,0]
        # Define the center of the visualization
        self.center_coords[self.x_axis] = self.horizontal_length // 2
        self.center_coords[self.y_axis] = self.vertical_length // 2
        self.center_coords[self.z_axis] = self.depth_length // 2

        self.head_size = 0.2  # taille moyenne de la tête en mètres (20 cm)
        self.gaze_length = 1

        self.static_view = None  # Static view of the room with all objects drawn but without participants
        # We store the rotation matrix to gain time
        self.rotation_matrix = {}
        
        # Performance optimization: GPU memory management
        self._gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self._gpu_available:
            self._gpu_image_buffer = cv2.cuda_GpuMat()
            self._gpu_overlay_buffer = cv2.cuda_GpuMat()
            
        # Cache for view transformations and object properties
        self._view_transform_cache = {}
        self._object_position_cache = {}
        self._object_size_cache = {}
        self._static_elements_cached = False

    def computeStaticView(self):
        """Get the static view of the room, with all the objects drawn but without the participants."""
        self.clean_image()  # Reset the image to white
        self.draw_axes()  # Draw the axes in the image
        for object in self.room_config.objects:
            self.draw_object(object)
        self.static_view = self.image.copy()
        
      
    def update_view(self, participant, frame=0, refresh=True, overlay=True):
        
        # if self.display:
        #     if(refresh):
        #         self.clean_image()  # Reset the image to white
        #         self.draw_axes()  # Draw the axes in the image
        #         for object in self.room_config.objects:
        #             self.draw_object(object)
        # self.image = self.static_view.copy() 
        target = "autre"
        if participant is not None: 
            target = self.draw_participant(participant, frame, overlay=overlay)
        else:
            print("Participant is None, cannot update view.")
        return target
    
    def clean_image(self):
        """Reset the image to white."""
        self.image.fill(255)
    
    def draw_axes(self):
        """Draw the axes in the image."""
        axis_name=['X', 'Y', 'Z']
        # Draw horizontal axis
        cv2.line(self.image, (0, self.center_coords[self.y_axis]), (self.horizontal_length, self.center_coords[self.y_axis]), (0, 0, 0), 2)
        cv2.putText(self.image, axis_name[self.x_axis], (self.horizontal_length - 20, self.center_coords[self.y_axis] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # # Draw vertical axis
        cv2.line(self.image, (self.center_coords[self.x_axis], 0), (self.center_coords[self.x_axis], self.vertical_length), (0, 0, 0), 2)
        cv2.putText(self.image, axis_name[self.y_axis], (self.center_coords[self.x_axis] + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def draw_object(self, name):
        """Draw an object in the image."""
        position=self.room_config.objects[name]['position'].copy()
        size=self.room_config.objects[name]['size'].copy()
        angle=self.room_config.objects[name]['angle'].copy()
        shape=self.room_config.objects[name]['shape'].copy()
        
        highlight=self.room_config.objects[name]['highlight'] if 'highlight' in self.room_config.objects[name] else False

        for s in shape:
            axis = [int(s[1]), int(s[3])]
            if(self.x_axis in axis and self.y_axis in axis):
                shape = shape[s]
                break
        color=self.room_config.objects[name]['color']
        
        if color is not None:
            x, y, z = self.getViewObjectPosition(position)
            size_x_px, size_y_px, size_z_px = self.getViewObjectSize(size)
            
            # We take into account when the object is rotated more than 90 degrees around the x_axis and the y-axis in the specific view
            # Example for the top view :
            #   if we apply a rotation to a rectangular object
            #       with an angle alpha between 90 degrees and 180 degrees, 
            #       around the y_axis (y_axis in the world view) or x_axis in the view (z_axis in the world view) 
            #       instead of drawing a rectangle (meaning defining the 4 points) with a width equals to size_x (size_z in world view) then applying a rotation of angle alpha, 
            #       we need to draw the rectangle with the width equals to size_z (size_x in world view) then apply a rotation of angle (alpha-90)
            if abs(angle[self.x_axis]) >= 90 and abs(angle[self.x_axis]) < 180 :
                size_y_px = size_x_px
                angle[self.x_axis] -= np.sign(angle[self.x_axis])*90
            if abs(angle[self.y_axis]) >= 90 and abs(angle[self.y_axis]) < 180:
                size_x_px = size_z_px
                angle[self.y_axis] -= np.sign(angle[self.y_axis])*90
                            
            point1 = (x - size_x_px // 2, y - size_y_px // 2,0)
            point2 = (x - size_x_px // 2, y + size_y_px // 2,0)
            point3 = (x + size_x_px // 2, y + size_y_px // 2,0)
            point4 = (x + size_x_px // 2, y - size_y_px // 2,0)

            rpoints = []
            for point in [point1, point2, point3, point4]:
                rotated = self.rotate_point(point, center = np.array([x, y,0]), angle_deg=[angle[self.x_axis], angle[self.y_axis], angle[self.z_axis]])
                # x, y, z = self.getViewObjectPosition(rotated)
                rpoints.append([rotated[0], rotated[1]])
            
            
            if shape == 'rectangle':
                cv2.drawContours(self.image, [np.array(rpoints).astype(int)], 0, color, -1, cv2.LINE_AA)
                if(highlight):
                    cv2.drawContours(self.image, [np.array(rpoints).astype(int)], 0, self.HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
            elif shape == 'circle':
                cv2.circle(self.image, (x, y), size_x_px // 2, color, -1)
                if(highlight):
                    cv2.circle(self.image, (x, y), size_x_px // 2, self.HIGHLIGHT_COLOR, 2)

    
    
    def draw_participant(self, participant, frame=0, overlay=True):
        # head = participant.getHeadPosition(frame).copy()
        # camera = self.room_config.objects['camera_'+participant.color].copy()
        # participant.setCamera(camera)
        # The position of the head is defined through the camera position
        # We need to translate the head to have the position according to the center of the view
        head = participant.getHeadPositionFromObject(participant.camera["position"], frame)
        
        if(head[0] is None or head[1] is None or head[2] is None):
            if(frame >= 1 and participant.last_known_position_frame is not None):
                self.draw_zone(participant, frame=participant.last_known_position_frame, overlay=overlay)
            return "autre"
        else:
            participant.last_known_position_frame = frame
            # We define the eye position before moving the head position        
            eye = self.rotate_point(head, center=participant.camera["position"], angle_deg=participant.camera["angle"])
            eye_x, eye_y, eye_z = self.getViewObjectPosition(eye)

            # We move the head center position away from the camera position because the eye is not at the center of the head
            head[2]-= self.head_size/2

            head = self.rotate_point(head, center=np.array(participant.camera["position"]), angle_deg=participant.camera["angle"])

            head_x, head_y,head_z = self.getViewObjectPosition(head)

            try:
                if self.display:
                    # print(head_x, head_y, head_z)
                    head_size_px = int(self.head_size * self.scale_factor)  
                    cv2.circle(self.image, (int(head_x), int(head_y)), head_size_px//2, participant.camera["color"], -1)
                    cv2.circle(self.image, (int(eye_x), int(eye_y)), 2, (0,0,0), -1)
                
                # Draw gaze vector with corrected Z-axis direction
                gaze_length = self.gaze_length* self.scale_factor  # Length of the gaze vector in pixels
                gaze_vec = participant.getGazeVector(frame)
                
                gaze_vec = self.rotate_point(gaze_vec, center=np.array(eye), angle_deg=[0,participant.camera["angle"][1],0])
                gaze_vec = self.getViewObjectPosition(gaze_vec)
                
                ray_origin = (eye_x, eye_y, eye_z)

                target = self.get_gaze_target(ray_origin, gaze_vec, participant.camera["color"])

                self.draw_zone(participant, frame=frame, overlay=overlay)
            except Exception as e:
                print(f"Error drawing participant {participant.id} at frame {frame}: {e}")

                return "autre"
            return target

    def draw_zone(self, participant, frame=0, alpha=0.3, overlay=False):
        """Draw the zones in the image with transparency."""
        overlay_image = self.image.copy()
        zone = self.room_config.zones[participant.id]
        position = participant.getHeadPositionFromObject(participant.camera["position"], frame=frame)
        
        try:
            position = self.rotate_point(position, center=np.array(participant.camera["position"]), angle_deg=participant.camera["angle"])
        except Exception as e:
            print(f"Error rotating position for participant {participant.id}: {e}")
            return 
        zone['position'] = position
        size = zone['size']
        x, y, z = self.getViewObjectPosition(position)
        size_x_px, size_y_px, size_z_px = self.getViewObjectSize(size)
        
        if(overlay):
            # Draw filled rectangle on overlay
            cv2.rectangle(overlay_image, 
                        (x - size_x_px // 2, y - size_y_px // 2), 
                        (x + size_x_px // 2, y + size_y_px // 2), 
                        zone["color"], -1)
            # Draw border and label on main image
            cv2.rectangle(self.image, 
                        (x - size_x_px // 2, y - size_y_px // 2), 
                        (x + size_x_px // 2, y + size_y_px // 2), 
                        zone["color"], 1)

            # Blend overlay with main image
            self.image = cv2.addWeighted(overlay_image, alpha, self.image, 1 - alpha, 0)
        else:
            # Draw filled rectangle on overlay
            cv2.rectangle(self.image, 
                        (x - size_x_px // 2, y - size_y_px // 2), 
                        (x + size_x_px // 2, y + size_y_px // 2), 
                        zone["color"], -1)

    def get_gaze_target(self, ray_origin, ray_dir, color = (0,0,0)):
        cv2.circle(self.image, (ray_origin[0], ray_origin[1]), 2, (0, 0, 0), 1)
        targets_in_range = []
        #Append the objects and the zones 
        objects_zones = self.room_config.zones.copy()
        objects_zones.update(self.room_config.objects)

        for object_name in objects_zones:
            object = objects_zones[object_name]
            if None in object['position'] or None in object['size']:
                # print("Warning: Object {} has None in position or size, skipping.".format(object_name))
                continue
            plane, intersection = self.intersect_ray_with_box(ray_origin=np.array(ray_origin),
                                                    ray_dir=np.array(ray_dir),
                                                    box_center=np.array(self.getViewObjectPosition(object['position'])),
                                                    box_size=np.array(self.getViewObjectSize(object['size'])))
            if intersection is not None:
                targets_in_range.append((object_name, intersection))
        
        if len(targets_in_range) > 0:
            
            # Extract the target with the closest intersection
            target = min(targets_in_range, key=lambda x: np.linalg.norm(np.array(ray_origin) - np.array(x[1])))
            if("wall" not in target[0]):
                thickness = 2
            else:
                thickness = 1
            
            color_target = objects_zones[target[0]]['color']
            cv2.circle(self.image, (int(target[1][0]), int(target[1][1])), 5, color_target, -1)
            cv2.circle(self.image, (int(target[1][0]), int(target[1][1])), 5, (0,0,0), 1)
            cv2.line(self.image, (ray_origin[0], ray_origin[1]), 
                        (int(target[1][0]), int(target[1][1])),  
                    color, thickness)
            return target[0]
        else:            
            cv2.line(self.image, (ray_origin[0], ray_origin[1]), 
                    (ray_origin[0]+int(ray_dir[0]), 
                    ray_origin[1]+int(ray_dir[1]) ),  
                color, 1)
        return "autre"
                

    
    def getViewObjectSize(self, object_size):
        """Get the size of an object in the view coordinates with caching."""
        cache_key = str(object_size)
        if cache_key in self._object_size_cache:
            return self._object_size_cache[cache_key]
            
        x = int(object_size[self.x_axis] * self.scale_factor)
        y = int(object_size[self.y_axis] * self.scale_factor)
        z = int(object_size[self.z_axis] * self.scale_factor)
        
        result = (x, y, z)
        self._object_size_cache[cache_key] = result
        return result
    

    def get_rotation_matrix(self, angle_deg=np.array([0.0, 0.0, 0.0])):
        if(str(angle_deg) in self.rotation_matrix):  
            return self.rotation_matrix[str(angle_deg)]
        pitch = np.deg2rad(angle_deg[0])
        
        yaw = np.deg2rad(angle_deg[1])
        roll = np.deg2rad(angle_deg[2])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        #We first rotate around the y-axis, then the x-axis, and finally the z-axis
        self.rotation_matrix[str(angle_deg)] = Rz @ Rx @ Ry
        return self.rotation_matrix[str(angle_deg)]

    def rotate_point(self, point, center=np.array([0.0, 0.0, 0.0]), angle_deg=np.array([0.0, 0.0, 0.0])):
        R = self.get_rotation_matrix(angle_deg)
        # Step 1: Translate so center is at origin
        translated = point - center
        # Step 2: Rotate
        rotated = R @ translated
        # Step 3: Translate back
        return rotated + center
    
    
    def translate(self, point, translation):
        """Translate a point by a given translation vector."""
        return np.array(point) + np.array(translation)


    def intersect_ray_with_plane(self, ray_origin, ray_dir, plane_point, plane_normal):
        denom = np.dot(ray_dir, plane_normal)
        if abs(denom) < 1e-6:
            return None
        d = np.dot(plane_point - ray_origin, plane_normal) / denom
        if d < 0:
            return None
        intersection = ray_origin + d * ray_dir
        return intersection
    
    def intersect_ray_with_box(self,ray_origin, ray_dir, box_center, box_size):
        """
        ray_origin: np.array([x, y, z])
        ray_dir: np.array([dx, dy, dz]) (doit être normalisé)
        box_center: np.array([x, y, z])
        box_size: np.array([size_x, size_y, size_z])
        Retourne (nom_du_plan, point_intersection) ou (None, None) si pas d'intersection
        """
        half_size = box_size / 2.0
        box_min = box_center - half_size
        box_max = box_center + half_size

        tmin = -np.inf
        tmax = np.inf

        for i in range(3):
            if abs(ray_dir[i]) < 1e-8:
                if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                    return None, None
            else:
                t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
                t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]
                t_near = min(t1, t2)
                t_far = max(t1, t2)
                if t_near > tmin:
                    tmin = t_near
                    hit_axis = i
                    hit_sign = -1 if t1 > t2 else 1
                tmax = min(tmax, t_far)
                if tmin > tmax or tmax < 0:
                    return None, None

        if tmin < 0:
            return None, None

        intersection = ray_origin + tmin * ray_dir

        # Trouver le plan touché
        epsilon = 1e-6
        plane = None
        normal = np.zeros(3)
        for i, name in zip(range(3), ['x', 'y', 'z']):
            if abs(intersection[i] - box_min[i]) < epsilon:
                plane = f"{name}_min"
                normal = np.zeros(3)
                normal[i] = -1
                break
            elif abs(intersection[i] - box_max[i]) < epsilon:
                plane = f"{name}_max"
                normal = np.zeros(3)
                normal[i] = 1
                break

        return plane, intersection


