import bpy
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
import synthdata.common as sd

def main():
    args = sd.get_arguments()
    bpy.ops.wm.open_mainfile(filepath='data/blend_files/3d_anime_girl.blend')
    bpy.ops.object.camera_add()
    bpy.context.scene.camera = bpy.context.object
    d_obj = bpy.context.scene.objects['Body']
    sphere_radius = sd.move_bounding_sphere_to_origin(d_obj)
    sd.set_renderer_cycles_gpu()
    sd.set_background_white()
    sd.set_general_render_settings()
    sd.set_cycles_render_settings()
    camera = bpy.context.scene.objects['Camera']
    extrinsic_camera_data = sd.random_render_on_sphere(
        camera, bpy.context.scene, sphere_radius+3, 
        (0, 0, 0), args.num_renders, args.save_directory
    )
    intrinsic_camera_data = sd.get_intrinsic_camera_data()
    transform_data = sd.build_transform_data(intrinsic_camera_data, extrinsic_camera_data)
    sd.save_transform_data(transform_data, args.save_directory)
    sd.fetch_and_save_tri_count(args.save_directory)
    bpy.ops.wm.save_as_mainfile(filepath='data/blend_files/test.blend')

if __name__ == '__main__':
    main()