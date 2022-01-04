import numpy as np
import time
import matplotlib.pyplot as plt

from geometry import read_light_obj, read_obj, normalize
from quadrics import trace_quadric

aux_time = time.time()

boundings_light, area_light = read_light_obj(name = 'light',path = 'objs/luzcornell.obj',Ia=0.5,r=1,g=1,b=1,Ip=1)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings1, objs1 = read_obj(name = 'cube1', path = 'objs/cube1.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5,reflection = 0.5)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings2, objs2 = read_obj(name = 'cube2', path = 'objs/cube2.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5,reflection = 0.5)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings3, objs3 = read_obj(name = 'floor',path = 'objs/floor.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, reflection = 0)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings4, objs4 = read_obj(name = 'ceiling',path = 'objs/ceiling.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, reflection = 0)
#1.0 0.0 0.0 0.3 0.7 0 0 5
boundings5, objs5 = read_obj(name = 'leftwall',path = 'objs/leftwall.obj',r=1,g=0,b=0,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, reflection = 0)
#0.0 1.0 0.0 0.3 0.7 0 0 5
boundings6, objs6 = read_obj(name='rightwall',path = 'objs/rightwall.obj',r=0,g=1,b=0,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, reflection = 0)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings7, objs7 = read_obj(name='back',path = 'objs/back.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, reflection = 0)

objs = objs3 + objs4 + objs5 + objs6 + objs7 + area_light
objs_wo_light = objs3 + objs4 + objs5 + objs6 + objs7

quadric1 = [{"a":1,"b":1,"c":1,"d":0,"e":0,"f":0,"g":2,"h":2,"j":28,"k":791,
  'ambient': np.array([0, 0.3, 0]),
  'diffuse': np.array([0, 0.7, 0]),
  'name': 'quadric1',
  'reflection': 0.5,
  'shininess': 20,
  'specular': np.array([0, 0, 0]),
  'transparency': np.array([0, 0, 0]),
  'ka':0.3,
  'kd':0.7,
  'ks':0,
  'kt':0}]

quadric2 = [{"a":1,"b":1,"c":1,"d":0,"e":0,"f":0,"g":-2,"h":2,"j":25,"k":632,
  'ambient': np.array([0, 0.3, 0.3]),
  'diffuse': np.array([0, 0.7, 0.7]),
  'name': 'quadric2',
  'reflection': 0.5,
  'shininess': 20,
  'specular': np.array([0, 0, 0]),
  'transparency': np.array([0, 0, 0]),
  'ka':0.3,
  'kd':0.7,
  'ks':0,
  'kt':0}]

quadric3 = [{"a":1,"b":1,"c":1,"d":0,"e":0,"f":0,"g":1,"h":0,"j":22,"k":481,
  'ambient': np.array([0.3, 0.3, 0.3]),
  'diffuse': np.array([0.7, 0.7, 0.7]),
  'name': 'quadric3',
  'reflection': 0.5,
  'shininess': 20,
  'specular': np.array([0, 0, 0]),
  'transparency': np.array([0, 0, 0]),
  'ka':0.3,
  'kd':0.7,
  'ks':0,
  'kt':0}]

quadric4 = [{"a":1/6,"b":1/3,"c":1/3,"d":0,"e":0,"f":0,"g":0,"h":0,"j":27/3,"k":242,
  'ambient': np.array([0.3, 0.3, 0]),
  'diffuse': np.array([0.7, 0.7, 0]),
  'name': 'quadric4',
  'reflection': 0.5,
  'shininess': 20,
  'specular': np.array([0, 0, 0]),
  'transparency': np.array([0, 0, 0]),
  'ka':0.3,
  'kd':0.7,
  'ks':0,
  'kt':0}]

quadric5 = [{"a":1/2,"b":1/2,"c":0,"d":0,"e":0,"f":0,"g":-1,"h":1/2,"j":1/12,"k":5.17,
  'ambient': np.array([0.3, 0.3, 0.3]),
  'diffuse': np.array([0.7, 0.7, 0.7]),
  'name': 'quadric5',
  'reflection': 0.5,
  'shininess': 20,
  'specular': np.array([0, 0, 0]),
  'transparency': np.array([0, 0, 0]),
  'ka':0.3,
  'kd':0.7,
  'ks':0,
  'kt':0}]

x_min, y_min, z_min = -3, -3, -29
x_max, y_max, z_max = -1, -1, -27

boundings_quadric1 = {'name': quadric1[0]['name'], 'min':np.array([x_min,y_min,z_min]), 'max':np.array([x_max,y_max,z_max])}

x_min, y_min, z_min = -2, -3, -29
x_max, y_max, z_max = -1, -1, -27

boundings_quadric1_half = {'name': quadric1[0]['name'], 'min':np.array([x_min,y_min,z_min]), 'max':np.array([x_max,y_max,z_max])}

x_min, y_min, z_min = 1, -3, -26
x_max, y_max, z_max = 3, -1, -24

boundings_quadric2 = {'name': quadric2[0]['name'], 'min':np.array([x_min,y_min,z_min]), 'max':np.array([x_max,y_max,z_max])}

x_min, y_min, z_min = -3, -2, -24
x_max, y_max, z_max = 1, 2, -20

boundings_quadric3 = {'name': quadric3[0]['name'], 'min':np.array([x_min,y_min,z_min]), 'max':np.array([x_max,y_max,z_max])}

x_min, y_min, z_min = -np.sqrt(6), -np.sqrt(3), -27-np.sqrt(3)
x_max, y_max, z_max = np.sqrt(6), np.sqrt(3), -27+np.sqrt(3)

boundings_quadric4 = {'name': quadric4[0]['name'], 'min':np.array([x_min,y_min,z_min]), 'max':np.array([x_max,y_max,z_max])}

x_min, y_min, z_min = 2-np.sqrt(7), -1-np.sqrt(4), -33
x_max, y_max, z_max = 2+np.sqrt(3), -1+np.sqrt(4), -18

boundings_quadric5 = {'name': quadric5[0]['name'], 'min':np.array([x_min,y_min,z_min]), 'max':np.array([x_max,y_max,z_max])}

##
quadrics = quadric1 + quadric2

all_boundings = list([boundings3, boundings4, boundings5, boundings6, boundings7, boundings_light, 
                      boundings_quadric1, boundings_quadric2])
all_boundings_wo_light = list([boundings3, boundings4, boundings5, boundings6, boundings7,
                               boundings_quadric1, boundings_quadric2])

screen = (-1, 1, 1, -1)
camera = np.array([0, 0, 5.7])

height, width = 100, 100

image = np.zeros((height, width, 3))

cnt = 0

for i, y in enumerate(np.linspace(screen[1], screen[3], height)):

  for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
    
    color = np.zeros((3))

    samples = 10

    for k in range(samples):
      pixel = np.array([x, y, 0])
      origin = camera
      direction = normalize(pixel - origin)

      color += trace_quadric(camera, origin, direction, objs, objs_wo_light, area_light, quadrics, all_boundings, max_depth = 4)

    color = np.clip(color/samples, 0, 1)
    image[i, j] = color/(color+1)

    cnt+=1

    print(round(cnt/(height*width),2),"---",round(time.time()-aux_time), end='\r')
        
plt.imsave('image.png', image)