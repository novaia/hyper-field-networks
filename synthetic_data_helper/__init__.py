import math
import mathutils
import bpy
import random
import json
import os
import warnings
import sys
import argparse

def set_renderer_cycles_gpu():
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.scene.cycles.device = 'GPU'
    bpy.data.scenes['Scene'].render.engine = 'CYCLES'

def set_general_render_settings():
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True

def set_cycles_render_settings():
    bpy.context.scene.cycles.samples = 100

def delete_all_and_add_camera():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.object.camera_add()
    bpy.context.scene.camera = bpy.context.object

def set_background(color, strength):
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = color
    bg.inputs[1].default_value = strength

def set_background_white():
    set_background((1.0, 1.0, 1.0), 1.0)

def move_bounding_sphere_to_origin(d_obj):
    bounding_box = d_obj.bound_box
    sphere_radius, sphere_origin = get_bounding_sphere(bounding_box)
    d_obj.location = -mathutils.Vector(sphere_origin)
    return sphere_radius

def get_arguments():
    arguments = sys.argv[sys.argv.index("--")+1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_directory', type=str, default='data/renders')
    parser.add_argument('--num_renders', type=int, default=200)
    return parser.parse_args(arguments)

def fetch_and_save_tri_count(save_directory):
    stats = bpy.context.scene.statistics(bpy.context.view_layer)
    tri_count = int(stats.split("Tris:")[1].split(' ')[0].replace(',', ''))
    with open(os.path.join(save_directory, 'tri_count.txt'), 'w') as f:
        f.write(str(tri_count))

# For more information on camera intrinsics and extrinsics, see NerfStudio documentation:
# https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
# For more information on how this data is extracted from Blender, see Maxime Raafat's
# BlenderNeRF repository, specifically, blender_nerf_operator.py:
# https://github.com/maximeraafat/BlenderNeRF
def get_intrinsic_camera_data():
    scene = bpy.context.scene
    camera = scene.camera
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

# AABB scale defines the side length of the bounding box in which the NeRF will trace rays.
# It was introduced by NVIDIA with Instant-NGP: https://github.com/NVlabs/instant-ngp
def get_aabb_scale(bounding_box):
    # Find the vertex of the bounding box that is furthest from the origin.
    furthest_bounding_vertex = None
    furthest_bounding_vertex_distance = -1
    for vertex in bounding_box:
        current_bounding_vertex_distance = vertex[0]**2 + vertex[1]**2 + vertex[2]**2
        if current_bounding_vertex_distance > furthest_bounding_vertex_distance:
            furthest_bounding_vertex = vertex
            furthest_bounding_vertex_distance = current_bounding_vertex_distance
    
    # Find the largest element of the vertex that is furthest from the origin.
    # This is equivalent to finding the bounding box's furthest distance from the origin along
    # one of the x, y, or z axes, or finding the bounding box's furthest axial distance.
    furthest_axial_distance = -1
    for i in range(len(furthest_bounding_vertex)):
        if furthest_bounding_vertex[i] > furthest_axial_distance:
            furthest_axial_distance = furthest_bounding_vertex[i]
    
    # Use the bounding box's furthest axial distance to determine the smallest 
    # AABB scale that will cover it. For example, if the furthest axial distance
    # is 10, then the smallest AABB scale that will cover it is 16.
    # As per Instant-NGP, the AABB scale is restricted to powers of 2.
    # Here it is also restricted to be <= 32 for efficiency because it is unlikely 
    # that larger values will be required.
    possible_aabb_scales = [1, 2, 4, 8, 16, 32]
    if furthest_axial_distance > possible_aabb_scales[-1]:
        warnings.warn(
            'Furthest axial distance is greater than max AABB scale, ' +
            'this means that parts of the object may be cut off when the NeRF is rendered. ' +
            'Consider setting the AABB scale manually in transforms.json. ' +
            f'Furthest axial distance: {furthest_axial_distance}, ' + 
            f'max AABB scale: {possible_aabb_scales[-1]}'
        )
        return possible_aabb_scales[-1]
    # Binary search is used to find the correct AABB scale given the constraints.
    low_index = 0
    high_index = len(possible_aabb_scales) - 1
    while high_index - low_index > 1:
        aabb_scale_index = (high_index + low_index) // 2
        if possible_aabb_scales[aabb_scale_index] < furthest_axial_distance:
            low_index = aabb_scale_index
        elif possible_aabb_scales[aabb_scale_index] > furthest_axial_distance:
            high_index = aabb_scale_index
    return possible_aabb_scales[high_index]

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

def random_render_on_sphere(
    camera, scene, sphere_radius, sphere_origin, num_renders, save_directory
):
    extrinsic_camera_data = []
    select_only_camera(camera)
    for i in range(num_renders):
        camera.location = sample_sphere(sphere_origin, sphere_radius)
        origin = mathutils.Vector(sphere_origin)
        direction = origin - camera.location
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
        render_name = f'render_{i}.png'
        scene.render.filepath = os.path.join(save_directory, render_name)
        bpy.ops.render.render(write_still = True)
        extrinsic_camera_data.append(
            build_extrinsics_element(render_name, listify_matrix(camera.matrix_world))
        )
    return {'frames': extrinsic_camera_data}

def global_mirror_offset_extrude(offset, ops_mesh):
    ops_mesh.extrude_vertices_move(
        MESH_OT_extrude_verts_indiv={'mirror':False}, 
        TRANSFORM_OT_translate={
            'value':offset, 
            'orient_type':'GLOBAL', 
            'orient_matrix':((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
            'orient_matrix_type':'GLOBAL', 
            'constraint_axis':(True, True, True), 
            'mirror':True
        }
    )
