import numpy as np
from geometry import names_intersected, nearest_intersected, light_position, normalize, reflected

def quadric_intersect(origin, direction, quadric):

  #a, b, c, d, e, f, g, h, j, k -- quadric surface
  #acoef, bcoef, ccoef -- Intersection coefficents
  #dx, dy, dz -- Direction - origin coordinates
  #disc -- Distance to intersection
  #root -- Root of distance to intersection
  #double  t -- Distance along ray to intersection
  #double  x0, y0, z0 -- Origin coordinates

  a = quadric["a"]
  b = quadric["b"]
  c = quadric["c"]
  d = quadric["d"]
  e = quadric["e"]
  f = quadric["f"]
  g = quadric["g"]
  h = quadric["h"]
  j = quadric["j"]
  k = quadric["k"]

  dx = direction[0]
  dy = direction[1]
  dz = direction[2]

  x0 = origin[0]
  y0 = origin[1]
  z0 = origin[2]

  acoef = 2*f*dx*dz + 2*e*dy*dz + c*dz*dz + b*dy*dy + a*dx*dx + 2*d*dx*dy

  bcoef = 2*b*y0*dy + 2*a*x0*dx + 2*c*z0*dz + 2*h*dy + 2*g*dx + 2*j*dz + 2*d*x0*dy + 2*e*y0*dz + 2*e*z0*dy + 2*d*y0*dx + 2*f*x0*dz + 2*f*z0*dx

  ccoef = a*x0*x0 + 2*g*x0 + 2*f*x0*z0 + b*y0*y0 + 2*e*y0*z0 + 2*d*x0*y0 + c*z0*z0 + 2*h*y0 + 2*j*z0 + k

  if ( 1.0 + acoef == 1.0 ):
    if ( 1.0 + bcoef == 1.0 ):
      return (None,np.inf,None,None)
    t = (-ccoef)/(bcoef)
   
  else:

    disc = bcoef*bcoef - 4*acoef*ccoef

    if ( 1.0 + disc < 1.0 ):
      return (None,np.inf,None,None)

    root = np.sqrt(disc)
    t = (-bcoef - root)/(2*acoef)

    if ( t < 0.0 ):
      t = (-bcoef + root)/(2*acoef)

  if ( t < 0.001 ):
    return (None,np.inf,None,None)

  x = origin[0] + direction[0]*t
  y = origin[1] + direction[1]*t
  z = origin[2] + direction[2]*t

  normal = np.array([2*a*x + 2*d*y + 2*f*z + 2*g, 2*b*y + 2*d*x + 2*e*z + 2*h, 2*c*z + 2*e*y + 2*f*x + 2*j])

  return (quadric,t, normalize(normal),quadric['name'])

def nearest_quadric(origin, direction, quadrics, names_quadrics):

  t = [np.inf]
  quadric = [None]
  normal = [None]
  name = [None]

  for i in quadrics:
    if i['name'] in names_quadrics:
      _quadric, _t, _normal, _name = quadric_intersect(origin, direction, i)
      
      quadric.append(_quadric)
      t.append(_t)
      normal.append(_normal)
      name.append(_name)

  index = t.index(min(t))

  return quadric[index], t[index], normal[index], name[index]

def trace_quadric(camera, origin, direction, objs, objs_wo_light, area_light, quadrics, all_boundings, max_depth):

  color = np.zeros((3))
  attenuation = 1

  for k in range(max_depth):

    names = names_intersected(all_boundings, origin, direction)

    # check for intersections
    nearest_object1, min_distance1, normal1, name1 = nearest_quadric(origin, direction, quadrics, names)
    nearest_object2, min_distance2, normal2, name2 = nearest_intersected(objs, origin, direction, names)

    if min_distance1 < min_distance2:
      nearest_object, min_distance, normal, name = nearest_object1, min_distance1, normal1, name1
    else:
      nearest_object, min_distance, normal, name = nearest_object2, min_distance2, normal2, name2
            
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

    _, min_distance_aux_quadric, _,_ = nearest_quadric(shifted_point, intersection_to_light, quadrics, names)
    _, min_distance_aux_objs, _,_ = nearest_intersected(objs_wo_light, shifted_point, intersection_to_light, names)
    min_distance = min(min_distance_aux_quadric,min_distance_aux_objs)

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
      U1 = (n12*(np.dot(normal,direction)) - np.sqrt(1-np.dot((n12**2),1-(np.dot(normal,direction))**2)))
      direction = np.dot(U1,normal) - n12*direction

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

    ##
    color += attenuation * illumination
    attenuation *= 0.4

  return color
