import numpy as np
from participant import Participant
from camera import Camera
import json
import re
import sys


class RoomConfiguration():
    

    def __init__(self):
        self.config = json.load(open("config.json"))
        self.initSceneSize()
        self.initObjects()
        self.initCameras()
    
    def initSceneSize(self):
        self.scene_size = self.getFieldValue(self.config["scene"]["size"])

    def initObjects(self):
        self.objects = {}
        for name in self.config["objects"]:
            self.objects[name] = {}
            object = self.config["objects"][name]
            # for object_field in object.keys()
            for key in object.keys():
                val = self.getFieldValue(object[key])
                self.objects[name][key] = val
        
    def initCameras(self):
        for name in self.config["cameras"]:
            self.objects['camera_'+name] = {}
            object = self.config["cameras"][name]
            # for object_field in object.keys()
            for key in object.keys():
                val = self.getFieldValue(object[key])
                self.objects['camera_'+name][key] = val
        self.initZones()
    
    def initZones(self):
        self.zones = {}
        zone_size = self.getFieldValue(self.config["zones"]["size"])
        for name in self.config["cameras"]:
            self.zones[name] = {
                "position" : [0,0,0], # On initialise la position de la zone à (0,0,0) mais elle va être modifiée pour suivre la position du visage du participant correspondant à la caméra avec le même id
                "size" : zone_size,
                "color" : self.objects['camera_'+name]["color"],
                "shape" : {'(0,1)': 'rectangle', '(0,2)': 'rectangle', '(1,2)': 'rectangle'},
                "angle" : [0,self.objects['camera_'+name]["angle"][1],0]
            }


    def getVarAttrs(self, var):
        attrs = []
        # on récupère les attributs que la variable va utiliser
        if('[' in var):
            attrs.append(var[:var.index('[')])
        while('[' in var):
            var = var[var.index('[')+1:]
            attrs.append(var[:var.index(']')])        
        return attrs

    def getVarValueFromConfig(self, attrs):
        attr_field = self.config.copy()
        # On récupère la variable dans le dictionnaire
        while len(attrs)>0:
            #On vérifie si l'attribut est un chiffre, 
            # et qui corresponderait donc à une position de vecteur/liste
            try: 
                i=int(attrs[0])
                # On vérifie qu'on est bien dans un vecteur/liste, puis on extrait l'élément à l'indice donné
                if(attr_field[0] in ['[','(']):
                    vec = attr_field[1:-1].split(',')
                    attr_field= vec[i]
                else:
                    raise ValueError("Expected a list or array for indexing")
            except ValueError:
                attr_field= attr_field[attrs[0]]
            attrs.pop(0)
        return attr_field

    def getFieldValue(self, field):
        # On vérifie si le champ est None
        if("None" in field):
            return None
        #On retire tous les espaces pour simplifier le traitement
        # Si la valeur du champ est une liste, un tuple ou un dictionnaire,
        # on récupère chaque élément, sinon on obtient une liste avec un seul élément
        if(field[0] in ['[', '(', '{']):
            # On isole le champ entre crochets, parenthèses ou accolades
            field_list = field[1:-1].strip().split(',')
        else:
            field_list = field.strip().split(',')

        value = []
        vars = {}
        if(len(field_list) == 1):
            if('$' in field_list[0]):
                attrs = self.getVarAttrs(field_list[0][1:])
                attr_field = self.getVarValueFromConfig(attrs)
                return self.getFieldValue(attr_field)
            return eval(field_list[0].strip())
        
        for dim in field_list:
            if('$' in dim):
                #On récupère toutes les variables (commence par un $)
                vars_idx =[m.start() for m in re.finditer(r'\$', dim)]
                for num,idx in enumerate(vars_idx):
                    # On cherche la fin de la variable en fonction de ce quil y a après
                    # si il y a une autre variable, on s'arrête juste avant le prochain $
                    if(num+1<len(vars_idx)):
                        var = dim[idx+1:vars_idx[num+1]-1]
                    else:# sinon on récupère tout jusqu'à la fin
                        var = dim[idx+1:]
                    #On isole la variable (de $ à dernier occurence de ']')
                    var = var[:var.rindex(']')+1]
                    var_str = '$'+var
                    
                    attrs = self.getVarAttrs(var)
                    attr_field = self.getVarValueFromConfig(attrs)
                    vars[var_str] = attr_field
                for var in vars:
                    dim = dim.replace(var, str(vars[var]))
                
                value.append(eval(dim))
            else:
                try:
                    value.append(float(dim))
                except ValueError:
                    # Si la conversion en float échoue, on est dans le cas d'un dictionnaire
                    value.append(dim)
        if(field[0] == '['):
            return np.array(value)
        elif(field[0] == '('):
            return tuple(value)
        elif(field[0] == '{'):
            dic = {}
            for v in value:
                key, val = v.split(':', 1)
                key = key.replace('\'','').replace('\"','').strip()
                val = val.replace('\'','').replace('\"','').strip()
                dic[key] = val
            return dic
        return value
       
    def setParticipants(self, participants):
        """
        Set the participants in the room configuration.
        
        :param participants: Dictionary of participants.
        """
        self.participants = participants
        for participant in participants:
            # Set the camera for each participant
            participant.setCamera(self.objects['camera_'+participant.id].copy())


    def angles_to_vector(self, angle_x, angle_y):
        """Convert angles to a 3D vector."""
        x = np.cos(angle_y) * np.sin(angle_x)
        y = np.sin(angle_y)
        z = -np.cos(angle_y) * np.cos(angle_x)
        return np.array([x, y, z])        
   
if __name__ == "__main__":
    from sideView import SideView
    from topView import TopView
    from sideViewOther import SideViewOther
    import os
    import cv2
    import sys
    from ptgaze.common import Face

    def nothing(x):
        pass
    # Example usage
    cv2.namedWindow("Gaze Tracker")
    # Example usage
    scale_factor = 300
    color = "vert"
    room_config = RoomConfiguration()
    
    
    side_view = SideViewOther(scale_factor=scale_factor, room_config=room_config)
    top_view = TopView(scale_factor=scale_factor, room_config=room_config)

    
    # side_view_other = SideViewOther(scale_factor=scale_factor, room_config=room_config)

    

    face = Face(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    face.head_position = np.array([ 0.0041763, -0.02742862, 0.62311349])
    face.head_position = np.array([ 0, -0.325,-1])
    face.gaze_vector = np.array([ 0.27407838 , 0.6095285 , -2.77341565])

    # side_view.update_view(None)
    # top_view.update_view(None)
    

    # Create trackbars for gaze vector components (range and initial value can be adjusted)
    cv2.createTrackbar('Gaze X', 'Gaze Tracker', 0, 180, nothing)
    cv2.createTrackbar('Gaze Y', 'Gaze Tracker', 0, 180, nothing)
    cv2.createTrackbar('Gaze Z', 'Gaze Tracker', 0, 180, nothing)

    cv2.createTrackbar('Head X', 'Gaze Tracker', 0, 200, nothing)
    cv2.createTrackbar('Head Y', 'Gaze Tracker', 0, 200, nothing)
    cv2.createTrackbar('Head Z', 'Gaze Tracker', 0, 200, nothing)

    

    cv2.setTrackbarMin('Gaze X', 'Gaze Tracker', -80)
    cv2.setTrackbarMin('Gaze Y', 'Gaze Tracker', -80)
    cv2.setTrackbarMin('Gaze Z', 'Gaze Tracker', -80)
    cv2.setTrackbarMin('Head X', 'Gaze Tracker', -60)
    cv2.setTrackbarMin('Head Y', 'Gaze Tracker', -10)
    cv2.setTrackbarMin('Head Z', 'Gaze Tracker', -100)

    cv2.setTrackbarMax('Gaze X', 'Gaze Tracker', 89)
    cv2.setTrackbarMax('Gaze Y', 'Gaze Tracker', 89)
    cv2.setTrackbarMax('Gaze Z', 'Gaze Tracker', 89)
    cv2.setTrackbarMax('Head X', 'Gaze Tracker', 60)
    cv2.setTrackbarMax('Head Y', 'Gaze Tracker', 70)
    cv2.setTrackbarMax('Head Z', 'Gaze Tracker', 0)

    cv2.setTrackbarPos('Gaze X', 'Gaze Tracker', 0)
    cv2.setTrackbarPos('Gaze Y', 'Gaze Tracker', 0)
    cv2.setTrackbarPos('Gaze Z', 'Gaze Tracker', 0)
    cv2.setTrackbarPos('Head X', 'Gaze Tracker', 0) 
    cv2.setTrackbarPos('Head Y', 'Gaze Tracker', 6)
    # cv2.setTrackbarPos('Head Y', 'Gaze Tracker', 0)
    cv2.setTrackbarPos('Head Z', 'Gaze Tracker', -60)

    h_top, w_top = top_view.image.shape[:2]
    h_side, w_side = side_view.image.shape[:2]
    

    # pad = np.full((20,w_top,3), 0, dtype=np.uint8)  # Create a white padding image
    pad_text = np.full((30,w_side,3),255, dtype=np.uint8)  # Create a white padding image for text
    pad_text[0,:]=0
    pad_text[-1,:]=0
    
    while True:
        prob = {"table": 0, "screen":0,"zone_left": 0, "zone_right": 0, "zone_front": 0, "camera":0, "other": 0}
    
        # Read trackbar positions and map them to a suitable range (e.g., -1 to 1 or -3 to 3)
        gx = cv2.getTrackbarPos('Gaze X', 'Gaze Tracker') * np.pi / 180
        gy = cv2.getTrackbarPos('Gaze Y', 'Gaze Tracker') * np.pi / 180
        gz = cv2.getTrackbarPos('Gaze Z', 'Gaze Tracker') * np.pi / 180

        hx = cv2.getTrackbarPos('Head X', 'Gaze Tracker') / 100.0
        hy = -cv2.getTrackbarPos('Head Y', 'Gaze Tracker') / 100.0
        hz = -cv2.getTrackbarPos('Head Z', 'Gaze Tracker') / 100.0
        
        angle = room_config.angles_to_vector(gx, gy) if color in ["rouge", "vert"] else room_config.angles_to_vector(gz, gy)
        participant = Participant(color, head_position=[[hx, hy, hz]], gaze_vector=[angle])
        
        room_config.setParticipants([participant])
        
        # print(side_view.getViewObjectPosition(angle))
        # print(top_view.getViewObjectPosition(angle))
        # face.gaze_vector = room_config.angles_to_vector(gx, gy)  # Convert angles to gaze vector
        # face.head_position = np.array([hx, hy, hz])
        
        # face.head_position = np.array([ 0,0,0])
        
        angles = [5, 10, 15, 20]
        # angles = [0]
        
        targets = side_view.update_view(participant)  # Update side view with multiple angles
        
        top_view.update_view(participant)
        
        # side_view_other.update_view(participant)
        
        # top_view_padded = np.hstack((top_view.image, pad))
        combined_image = np.vstack((pad_text, side_view.image, pad_text, top_view.image, pad_text))
        # print(targets)
        # for target in targets:
        #     prob[target] += 1

        # cv2.putText(combined_image, "Side View", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        # cv2.putText(combined_image, "Top View", (0, h_side+30+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        # prob = {k: 100*v / len(angles) for k, v in prob.items()}
        # text_prob = f" ".join([f"{k}: {v:.1f}%" for k, v in prob.items()])
        # cv2.putText(combined_image, text_prob, (0, combined_image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        cv2.imshow("Side View", combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    