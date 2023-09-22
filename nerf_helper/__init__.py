import math
import mathutils
import bpy
import random
import json
import os

# For more information on camera intrinsics and extrinsics, see NerfStudio documentation:
# https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
# For more information on how this data is extracted from Blender, see Maxime Raafat's
# BlenderNeRF repository, specifically, blender_nerf_operator.py:
# https://github.com/maximeraafat/BlenderNeRF
def get_intrinsic_camera_data(scene, camera):
    camera_angle_x = camera.data.angle_x # Camera FOV.
    camera_angle_y = camera.data.angle_y
    sensor_size_mm = camera.data.sensor_width
    focal_length_mm = camera.data.lens
    render_resolution_x = scene.render.resolution_x
    render_resolution_y = scene.render.resolution_y
    optical_center_x = render_resolution_x / 2
    optical_center_y = render_resolution_y / 2
    s_u = focal_length_mm / sensor_size_mm * render_resolution_x
    s_v = focal_length_mm / sensor_size_mm * render_resolution_y

    intrinsic_camera_data = {
        'camera_angle_x': camera_angle_x,
        'camera_angle_y': camera_angle_y,
        'fl_x': s_u,
        'fl_y': s_v,
        'k1': 0.0,
        'k2': 0.0,
        'p1': 0.0,
        'p2': 0.0,
        'cx': optical_center_x,
        'cy': optical_center_y,
        'w': render_resolution_x,
        'h': render_resolution_y,
        'aabb_scale': 1
        #scene.aabb, #this has something to do with the bounding box of the scene.
    } 

    # Debug.
    print('camera_angle_x: ', camera_angle_x)
    print('camera_angle_y: ', camera_angle_y)
    print('sensor_size_mm: ', sensor_size_mm)
    print('focal_length_mm: ', focal_length_mm)
    print('render_resolution_x: ', render_resolution_x)
    print('render_resolution_y: ', render_resolution_y)
    print('s_u: ', s_u)
    print('s_v: ', s_v)

    return intrinsic_camera_data

# Builds an extrinsic camera data element for a single frame.
def build_extrinsics_element(file_path, transformation_matrix):
    return {
        'file_path': file_path, 
        'transform_matrix': transformation_matrix
    }

# Transform data is the combination of intrinsic and extrinsic camera data.
def build_transform_data(intrinsic, extrinsic):
    return {**intrinsic, **extrinsic}

# Transform data contains intrinsic and extrinsic camera data for entire rendered dataset.
def save_transform_data(transform_data, save_directory):
    file_path = os.path.join(save_directory, 'transforms.json')
    with open(file_path, 'w') as f:
        json.dump(
            transform_data,
            f,
            indent=4
        )
    
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def get_bounding_sphere(bounding_box):
    x_center = (bounding_box[0][0] + bounding_box[6][0]) / 2
    y_center = (bounding_box[0][1] + bounding_box[6][1]) / 2
    z_center = (bounding_box[0][2] + bounding_box[6][2]) / 2

    max_vector_size = -1
    for i in range(len(bounding_box)):
        current_vector = bounding_box[i]
        x_size = (x_center - current_vector[0])**2
        y_size = (y_center - current_vector[1])**2
        z_size = (z_center - current_vector[2])**2
        current_vector_size = x_size + y_size + z_size
        if current_vector_size > max_vector_size:
            max_vector_size = current_vector_size

    sphere_radius = math.sqrt(max_vector_size)
    sphere_origin = (x_center, y_center, z_center)
    return sphere_radius, sphere_origin

def render_multiple_on_plane(
    plane_width, 
    plane_height, 
    horizontal_steps, 
    vertical_steps, 
    render_name,
    camera,
    scene
):
    frame_meta_data = []
    horizontal_step_size = plane_width / (horizontal_steps - 1)
    vertical_step_size = plane_height / (vertical_steps - 1)
    start_translation = (
        -plane_width / 2,
        0,
        -plane_height / 2
    )
    bpy.ops.transform.translate(value=start_translation, orient_type='LOCAL')

    files_rendered = 0
    for x_step in range(horizontal_steps):
        for z_step in range(vertical_steps):
            camera_translation = (
                x_step * horizontal_step_size,
                0,
                z_step * vertical_step_size
            )
            bpy.ops.transform.translate(value=camera_translation, orient_type='LOCAL')
            
            render_path = f'data/renders/{render_name}_{files_rendered}.png'
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still = True)
            files_rendered += 1

            frame_meta_data.append({
                'file_path': render_path, 
                'location': [*camera.location], 
                'rotation': [*camera.rotation_euler]
            })

            reset_translation = (
                -x_step * horizontal_step_size,
                0,
                -z_step * vertical_step_size
            )
            bpy.ops.transform.translate(value=reset_translation, orient_type='LOCAL') 
    return frame_meta_data

def select_only_camera(camera):
    bpy.ops.object.select_all(action='DESELECT')
    camera.select_set(True)

def render_on_planes(camera, scene):
    select_only_camera(camera)

    render_views = [
        ['front', (0, -30, 0), (math.radians(90), 0, 0)],
        ['right', (30, 0, 0), (math.radians(90), 0, math.radians(90))],
        ['back', (0, 30, 0), (math.radians(90), 0, math.radians(180))],
        ['left', (-30, 0, 0), (math.radians(90), 0, math.radians(270))]
    ]

    frame_meta_data = []
    for i in range(len(render_views)):
        current_view_name = render_views[i][0]
        camera.location = render_views[i][1]
        camera.rotation_euler = render_views[i][2]

        plane_width = 5
        plane_height = 5
        horizontal_steps = 2
        vertical_steps = 2
        current_frame_meta_data = render_multiple_on_plane(
            plane_width, 
            plane_height, 
            horizontal_steps, 
            vertical_steps, 
            current_view_name,
            camera,
            scene
        )
        frame_meta_data.extend(current_frame_meta_data)
    return frame_meta_data

def sample_sphere(origin, radius):
    theta = random.uniform(0, 2*math.pi)
    phi = random.uniform(-math.pi/2, math.pi/2)
    x = origin[0] + radius * math.cos(phi) * math.cos(theta)
    y = origin[1] + radius * math.cos(phi) * math.sin(theta)
    z = origin[2] + radius * math.sin(phi)
    return (x, y, z)

def random_render_on_sphere(camera, scene, sphere_radius, sphere_origin, num_renders):
    extrinsic_camera_data = []
    select_only_camera(camera)
    for i in range(num_renders):
        camera.location = sample_sphere(sphere_origin, sphere_radius)
        origin = mathutils.Vector(sphere_origin)
        direction = origin - camera.location
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
        render_name = f'render_{i}.png'
        scene.render.filepath = os.path.join('data/renders', render_name)
        bpy.ops.render.render(write_still = True)
        extrinsic_camera_data.append(
            build_extrinsics_element(render_name, listify_matrix(camera.matrix_world))
        )
    return {'frames': extrinsic_camera_data}
