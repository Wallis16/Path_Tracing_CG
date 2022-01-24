import numpy as np
import matplotlib.pyplot as plt
import time

from geometry import read_light_obj, read_obj, normalize
from path_tracing import trace

aux_time = time.time()

#eye 0.0 0.0 5.7
#size 200 200
#ortho -1 -1 1 1
#background 0.0 0.0 0.0
#ambient 0.5

#light luzcornell.obj 1.0 1.0 1.0 1.0
#object <name.obj> red green blue ka kd ks kt n

width = 200
height = 200

camera = np.array([0, 0, 5.7])
screen = (-1, 1, 1, -1) # left, top, right, bottom

light = { 'position': np.array([-0.9100, 3.8360, -23.3240]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

boundings_light, area_light = read_light_obj(name = 'light',path = 'objs/luzcornell.obj',Ia=0.5,r=1,g=1,b=1,Ip=1)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings1, objs1 = read_obj(name = 'cube1', path = 'objs/cube1.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5,refraction = 1.33)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings2, objs2 = read_obj(name = 'cube2', path = 'objs/cube2.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5,refraction = 0)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings3, objs3 = read_obj(name = 'floor',path = 'objs/floor.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, refraction = 0)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings4, objs4 = read_obj(name = 'ceiling',path = 'objs/ceiling.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, refraction = 0)
#1.0 0.0 0.0 0.3 0.7 0 0 5
boundings5, objs5 = read_obj(name = 'leftwall',path = 'objs/leftwall.obj',r=1,g=0,b=0,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, refraction = 0)
#0.0 1.0 0.0 0.3 0.7 0 0 5
boundings6, objs6 = read_obj(name='rightwall',path = 'objs/rightwall.obj',r=0,g=1,b=0,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, refraction = 0)
#1.0 1.0 1.0 0.3 0.7 0 0 5
boundings7, objs7 = read_obj(name='back',path = 'objs/back.obj',r=1,g=1,b=1,ka=0.3,kd=0.7,ks=0,kt=0,n = 5, refraction = 0)

objs = objs1 + objs2 + objs3 + objs4 + objs5 + objs6 + objs7 + area_light
objs_wo_light = objs1 + objs2 + objs3 + objs4 + objs5 + objs6 + objs7
#objs = objs1 + objs3

all_boundings = list([boundings1,boundings2,boundings3,boundings4,boundings5,boundings6,boundings7, boundings_light])
all_boundings_wo_light = list([boundings1,boundings2,boundings3,boundings4,boundings5,boundings6,boundings7])

#################

height, width = 100, 100

image = np.zeros((height, width, 3))

cnt = 0

for i, y in enumerate(np.linspace(screen[1], screen[3], height)):

  for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
    
    color = np.zeros((3))

    samples = 3

    for k in range(samples):
      pixel = np.array([x, y, 0])
      origin = camera
      direction = normalize(pixel - origin)

      color += trace(camera, origin, direction, objs, objs_wo_light, all_boundings, area_light, max_depth = 5)
    
    image[i, j] = np.clip(color/samples, 0, 1)

    cnt+=1

    print(round(cnt/(height*width),2),"---",round(time.time()-aux_time), end='\r')

plt.imsave('image.png', image)