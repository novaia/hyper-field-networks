import bpy
import bmesh
import random
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
import synthdata.common as sd

def generate():
    extrude = sd.global_mirror_offset_extrude
    rng = random.uniform
    
    hip_offset = (rng(0.4, 1), rng(-0.2, 0.2), rng(-0.4, 0))
    leg_offset = (0, 0, rng(-4, -0.5))
    spine_offset = (0, rng(-0.3, 0.3), rng(1.5, 4))
    collar_bone_offset = (rng(0.5, 1.5), rng(-0.2, 0.2), rng(-0.2, 0.2))
    arm_offset = (rng(0.7, 3), 0, 0)
    neck_base_offset = (0, 0, rng(0.2, 0.5))
    neck_offset = (0, 0, rng(0, 0.5))
    neck_offset = (0, rng(-0.3, 0.3), rng(0.3, 1))
    head_base_offset = (0, rng(-1, 1), rng(0, 0.7))
    head_top_offset = (0, rng(-2, 2), rng(0, 2))
    
    spine_radius = rng(0.4, 1)
    spine_radius = (spine_radius, spine_radius)
    root_radius = rng(0.4, 1)
    root_radius = (root_radius, root_radius)
    hip_radius = rng(0.4, 1)
    hip_radius = (hip_radius, hip_radius)
    collar_bone_radius = rng(0.4, 0.6)
    collar_bone_radius = (collar_bone_radius, collar_bone_radius)
    neck_radius = rng(0.3, 1)
    neck_radius = (neck_radius, neck_radius)
    head_base_radius = rng(0.5, 2)
    head_base_radius = (head_base_radius, head_base_radius)
    head_top_radius = rng(0.5, 2)
    head_top_radius = (head_top_radius, head_top_radius)
    
    color = (rng(0, 1), rng(0, 1), rng(0, 1), 1)

    bpy.ops.mesh.primitive_plane_add(
        size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1)
    )
    ops_obj = bpy.ops.object
    ops_obj.editmode_toggle()
    ops_mesh = bpy.ops.mesh
    ops_mesh.select_all(action='SELECT')
    ops_mesh.merge(type='CENTER')

    ops_obj.modifier_add(type='MIRROR')
    ops_obj.modifier_add(type='SKIN')
    ops_obj.modifier_add(type='SUBSURF')

    ctx_obj = bpy.context.object
    ctx_obj.modifiers['Skin'].use_smooth_shade = True
    ctx_obj.modifiers["Subdivision"].levels = 3
    ctx_obj.modifiers["Subdivision"].render_levels = 3

    d_obj = bpy.data.objects['Plane']
    b_mesh = bmesh.from_edit_mesh(d_obj.data)
    b_mesh.verts.ensure_lookup_table()
    b_mesh.verts[0].select = True

    # Extrude hip.
    extrude(hip_offset, ops_mesh)

    b_mesh.verts.ensure_lookup_table()
    b_mesh.verts[0].select = False
    b_mesh.verts[1].select = True

    # Extrude leg.
    extrude(leg_offset, ops_mesh)

    b_mesh.verts.ensure_lookup_table()
    b_mesh.verts[1].select = False
    b_mesh.verts[0].select = True

    # Extrude spine.
    extrude(spine_offset, ops_mesh)

    # The indexes change as new vertices are created.
    # I think 1 might always be the last created vertex, while 0 is the root.
    # Need to test.
    b_mesh.verts.ensure_lookup_table()
    b_mesh.verts[0].select = False
    b_mesh.verts[1].select = True

    # Extrude collar bone.
    extrude(collar_bone_offset, ops_mesh)

    b_mesh.verts.ensure_lookup_table()
    b_mesh.verts[4].select = True
    b_mesh.verts[1].select = False

    # Extrude arm.
    extrude(arm_offset, ops_mesh)

    b_mesh.verts.ensure_lookup_table()
    b_mesh.verts[4].select = False
    b_mesh.verts[5].select = False
    b_mesh.verts[1].select = True

    # Extrude neck base.
    extrude(neck_base_offset, ops_mesh)
    # Extrude neck.
    extrude(neck_offset, ops_mesh)
    # Extrude head base.
    extrude(head_base_offset, ops_mesh)
    # Extrude had top.
    extrude(head_top_offset, ops_mesh)

    b_mesh.verts.ensure_lookup_table()

    # TODO: mark vertex 0 as root for skin modifier so symmetry is maintained,
    # when using different scales for each axis.

    skin_layer = b_mesh.verts.layers.skin.verify()
    # Spine.
    b_mesh.verts[1][skin_layer].radius = spine_radius
    # Root.
    b_mesh.verts[0][skin_layer].radius = root_radius
    # Hip.
    b_mesh.verts[3][skin_layer].radius = hip_radius
    # Collar bone.
    b_mesh.verts[4][skin_layer].radius = collar_bone_radius
    # Neck.
    b_mesh.verts[7][skin_layer].radius = neck_radius
    # Head base.
    b_mesh.verts[8][skin_layer].radius = head_base_radius
    # Head top.
    b_mesh.verts[9][skin_layer].radius = head_top_radius

    mat = bpy.data.materials.new(name="Material")
    mat.diffuse_color = color
    ctx_obj.data.materials.append(mat)

    ops_obj.editmode_toggle()
    ops_obj.modifier_apply(modifier='Mirror')
    ops_obj.modifier_apply(modifier='Skin')
    ops_obj.modifier_apply(modifier='Subdivision')

    return d_obj

def main():
    args = sd.get_arguments()
    sd.delete_all_and_add_camera()
    d_obj = generate()
    sphere_radius = sd.move_bounding_sphere_to_origin(d_obj)
    sd.set_renderer_cycles_gpu()
    sd.set_background_white()
    sd.set_general_render_settings()
    sd.set_cycles_render_settings()
    camera = bpy.context.scene.objects['Camera']
    extrinsic_camera_data = sd.random_render_on_sphere(
        camera, bpy.context.scene, sphere_radius+10, 
        (0, 0, 0), args.num_renders, args.save_directory
    )
    intrinsic_camera_data = sd.get_intrinsic_camera_data()
    transform_data = sd.build_transform_data(intrinsic_camera_data, extrinsic_camera_data)
    sd.save_transform_data(transform_data, args.save_directory)
    sd.fetch_and_save_tri_count(args.save_directory)
    bpy.ops.wm.save_as_mainfile(filepath='data/blend_files/test.blend')

if __name__ == '__main__':
    main()