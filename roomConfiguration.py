import numpy as np
from participant import Participant
# from camera import Camera
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
   