import numpy as np
import random

def normalize(vector):
    return vector / np.linalg.norm(vector)

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def triangle_intersect(vertexes,camera,direction):
  """
  vertexes = A,B,C
  camera = position of camera (eye)
  direction = vector from origin to pixel
  """

  N = np.cross(vertexes[1] - vertexes[0], vertexes[2] - vertexes[0])
  N = normalize(N)

  D = -np.dot(N,vertexes[0])

  t = - (np.dot(N, camera) + D) / np.dot(N, direction)

  P = camera + t*direction

  edge0 = vertexes[1] - vertexes[0] 
  edge1 = vertexes[2] - vertexes[1] 
  edge2 = vertexes[0] - vertexes[2]

  C0 = P - vertexes[0] 
  C1 = P - vertexes[1]
  C2 = P - vertexes[2]

  if (np.dot(N, np.cross(edge0, C0)) > 0 and np.dot(N, np.cross(edge1, C1)) > 0 and np.dot(N, np.cross(edge2, C2)) > 0) and t > 0:
     return (t,N)
  else:
    return None

def nearest_intersected_triangle(objects, ray_origin, ray_direction):
    distances = [triangle_intersect(obj['vertexes'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def nearest_intersected_sphere(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def nearest_intersected(objects, ray_origin, ray_direction, names, geom = 'triangle'):

    distances = []
    new_objects = []

    if geom == 'sphere':
      distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    if geom == 'triangle':
      for obj in objects:
        if obj[0]['name'] in names:
          distances.append(triangle_intersect(obj[0]['vertexes'], ray_origin, ray_direction))
          new_objects.append(obj)

    nearest_object = None
    min_distance = np.inf
    normal = None
    index_aux = None

    for index, distance in enumerate(distances):

        try:

          if distance[0] and distance[0] < min_distance:
              min_distance = distance[0]
              nearest_object = new_objects[index][0]
              normal = distance[1]
              index_aux = index
        except:
          pass
          
    if index_aux != None:
      return nearest_object, min_distance, normal, new_objects[index_aux][0]['name']
    else:
      return nearest_object, min_distance, normal, None

class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    #vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))

            f.close()
        except IOError:
            print(".obj file not found.")

def read_obj(name,path,r,g,b,ka,kd,ks,kt,n,reflection):
  objs = []
  object_ = ObjLoader(path)

  x_min = []
  y_min = []
  z_min = []
  x_max = []
  y_max = []
  z_max = []

  for i in object_.faces:

    x_min.append(min(list(object_.vertices[int(i[0])-1])[0],list(object_.vertices[int(i[1])-1])[0],list(object_.vertices[int(i[2])-1])[0]))
    y_min.append(min(list(object_.vertices[int(i[0])-1])[1],list(object_.vertices[int(i[1])-1])[1],list(object_.vertices[int(i[2])-1])[1]))
    z_min.append(min(list(object_.vertices[int(i[0])-1])[2],list(object_.vertices[int(i[1])-1])[2],list(object_.vertices[int(i[2])-1])[2]))

    x_max.append(max(list(object_.vertices[int(i[0])-1])[0],list(object_.vertices[int(i[1])-1])[0],list(object_.vertices[int(i[2])-1])[0]))
    y_max.append(max(list(object_.vertices[int(i[0])-1])[1],list(object_.vertices[int(i[1])-1])[1],list(object_.vertices[int(i[2])-1])[1]))
    z_max.append(max(list(object_.vertices[int(i[0])-1])[2],list(object_.vertices[int(i[1])-1])[2],list(object_.vertices[int(i[2])-1])[2]))

    objs.append([{  'name' : name
                    ,'vertexes': np.array([list(object_.vertices[int(i[0])-1]),
                                         list(object_.vertices[int(i[1])-1]),
                                         list(object_.vertices[int(i[2])-1])]),
                    'ambient': np.array([r*ka,g*ka,b*ka]),
                    'diffuse': np.array([r*kd,g*kd,b*kd]),
                    'specular': np.array([ks,ks,ks]),
                    'transparency': np.array([r*kt,g*kt,b*kt]),
                    'shininess': 4*n,
                    'reflection': reflection,
                    'ka': ka,
                    'kd': kd,
                    'ks': ks,
                    'kt': kt}])

  boundings = {'name': name, 'min':np.array([min(x_min),min(y_min),min(z_min)]), 'max':np.array([max(x_max),max(y_max),max(z_max)])}

  return boundings, objs

def read_light_obj(name,path,Ia,r,g,b,Ip):

  objs = []
  object_ = ObjLoader(path)

  x_min = []
  y_min = []
  z_min = []
  x_max = []
  y_max = []
  z_max = []

  for i in object_.faces:

    x_min.append(min(list(object_.vertices[int(i[0])-1])[0],list(object_.vertices[int(i[1])-1])[0],list(object_.vertices[int(i[2])-1])[0]))
    y_min.append(min(list(object_.vertices[int(i[0])-1])[1],list(object_.vertices[int(i[1])-1])[1],list(object_.vertices[int(i[2])-1])[1]))
    z_min.append(min(list(object_.vertices[int(i[0])-1])[2],list(object_.vertices[int(i[1])-1])[2],list(object_.vertices[int(i[2])-1])[2]))

    x_max.append(max(list(object_.vertices[int(i[0])-1])[0],list(object_.vertices[int(i[1])-1])[0],list(object_.vertices[int(i[2])-1])[0]))
    y_max.append(max(list(object_.vertices[int(i[0])-1])[1],list(object_.vertices[int(i[1])-1])[1],list(object_.vertices[int(i[2])-1])[1]))
    z_max.append(max(list(object_.vertices[int(i[0])-1])[2],list(object_.vertices[int(i[1])-1])[2],list(object_.vertices[int(i[2])-1])[2]))

    objs.append([{  'name': name,
                    'vertexes': np.array([list(object_.vertices[int(i[0])-1]),
                                        list(object_.vertices[int(i[1])-1]),
                                        list(object_.vertices[int(i[2])-1])]),
                    'ambient': Ia,
                    'diffuse': np.array([r*Ip,g*Ip,b*Ip]),
                    'specular': np.array([r*Ip,g*Ip,b*Ip])}])
    
  boundings = {'name': name, 'min':np.array([min(x_min),min(y_min),min(z_min)]), 'max':np.array([max(x_max),max(y_max),max(z_max)])}

  return boundings, objs
    
def light_position(objs):

  rdn = random.randint(0, 1)

  r1 = np.random.uniform()
  r2 = np.random.uniform()

  A = objs[rdn][0]['vertexes'][0]*(1-np.sqrt(r1)) 
  B = objs[rdn][0]['vertexes'][1]*(np.sqrt(r1)*(1-r2))
  C = objs[rdn][0]['vertexes'][2]*r2*np.sqrt(r1)

  return np.array(A+B+C)

def reflected(vector, axis):
  return vector - 2 * np.dot(vector, axis) * axis

def object_intersection(boundings,origin,direction):

  tmin = (boundings['min'][0] - origin[0]) / direction[0]
  tmax = (boundings['max'][0] - origin[0]) / direction[0] 
 
  if (tmin > tmax):
    tmin, tmax = tmax, tmin 
 
  tymin = (boundings['min'][1] - origin[1]) / direction[1] 
  tymax = (boundings['max'][1] - origin[1]) / direction[1] 
 
  if (tymin > tymax):
    tymin, tymax = tymax, tymin 
 
  if ((tmin > tymax) or (tymin > tmax)):
    return False 
 
  if (tymin > tmin): 
    tmin = tymin 
 
  if (tymax < tmax): 
    tmax = tymax 
 
  tzmin = (boundings['min'][2] - origin[2]) / direction[2] 
  tzmax = (boundings['max'][2] - origin[2]) / direction[2]
 
  if (tzmin > tzmax):
    tzmin, tzmax = tzmax, tzmin 
 
  if ((tmin > tzmax) or (tzmin > tmax)): 
    return False 
 
  if (tzmin > tmin): 
    tmin = tzmin 
 
  if (tzmax < tmax): 
    tmax = tzmax 
 
  return True

def names_intersected(boundings,origin,direction):
  
  names = []

  for i in boundings:
    if object_intersection(i,origin,direction):
      names.append(i['name'])

  return names