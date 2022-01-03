import numpy as np

from geometry import names_intersected, nearest_intersected, light_position, normalize, reflected

def trace(camera, origin, direction, objs, objs_wo_light, all_boundings, area_light, max_depth):

  color = np.zeros((3))
  reflection = 1

  for k in range(max_depth):

    names = names_intersected(all_boundings, origin, direction)

    # check for intersections
    nearest_object, min_distance, normal, name = nearest_intersected(objs, origin, direction, names)
            
    if name == 'light':
      if k == 0:
        color = np.array([255,255,255])
      break

    if nearest_object is None:
      break

    # compute intersection point between ray and nearest object
    intersection = origin + min_distance * direction
    
    if np.dot(normal,direction) > 0:
      normal*=-1

    shifted_point = intersection + 1e-5*normal

    light_point = light_position(area_light)

    intersection_to_light = normalize(light_point - shifted_point)

    names = names_intersected(all_boundings, shifted_point, intersection_to_light)

    _, min_distance, _,aux = nearest_intersected(objs_wo_light, shifted_point, intersection_to_light, names)

    intersection_to_light_distance = np.linalg.norm(light_point - intersection)
    is_shadowed = min_distance < intersection_to_light_distance

    ###
    kd, ks, kt = nearest_object["kd"], nearest_object["ks"], nearest_object["kt"]
    ktot = kd + ks + kt

    R = np.random.uniform(0,ktot)

    if R < kd:
      ray = 'diffuse'

    if R >= kd and R < kd + ks:
      ray = 'specular'

    if R >= kd + ks:
      ray = 'transmission'

    if ray == 'diffuse':

      phi = np.arccos(np.sqrt(np.random.uniform()))
      theta = 2*np.pi*np.random.uniform()

      direction_diffuse = np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])

      if np.dot(normal,direction_diffuse) < 0:
        direction = -direction_diffuse
      else:
        direction = direction_diffuse

    if ray == 'specular':
      direction = reflected(direction, normal)

    if ray == 'transmission':
      #n12 = n1/n2
      n12 = 0.67 
      U1 = (n12*(np.dot(normal,direction)) - np.sqrt(1-np.dot((n12**2),1-(np.dot(normal,direction))**2)))
      direction = np.dot(U1,normal) - n12*direction
      #shifted_point = intersection - 1e-5*normal

    origin = shifted_point
    ### 

    if is_shadowed:
      continue

    if k == 0:
      # RGB
      illumination = np.zeros((3))

      # ambiant
      illumination += nearest_object['ambient'] * area_light[0][0]['ambient']

      # diffuse
      illumination += nearest_object['diffuse'] * area_light[0][0]['diffuse'] * np.dot(intersection_to_light, normal)

      # specular
      intersection_to_camera = normalize(camera - intersection)
      H = normalize(intersection_to_light + intersection_to_camera)
      illumination += nearest_object['specular'] * area_light[0][0]['specular'] * np.dot(normal, H) ** (nearest_object['shininess'] / 4)

    else:
      # RGB
      illumination = np.zeros((3))

      # ambiant
      illumination += nearest_object['ambient'] * area_light[0][0]['ambient']

      # diffuse
      illumination += nearest_object['diffuse'] * area_light[0][0]['diffuse'] * np.dot(intersection_to_light, normal)

      # specular
      intersection_to_camera = normalize(camera - intersection)
      H = normalize(intersection_to_light + intersection_to_camera)
      illumination += nearest_object['specular'] * area_light[0][0]['specular'] * np.dot(normal, H) ** (nearest_object['shininess'] / 4)

    # reflection
    color += reflection * illumination
    reflection *= nearest_object['kd']
    #reflection *= color*nearest_object['kd']

  return color