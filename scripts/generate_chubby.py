import bpy
import bmesh
import random
import math

def extrude(offset, ops_mesh):
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

def generate(
    hip_offset,
    leg_offset,
    spine_offset,
    collar_bone_offset,
    arm_offset,
    neck_base_offset,
    neck_offset,
    head_base_offset,
    head_top_offset,
    spine_radius, 
    root_radius, 
    hip_radius, 
    collar_bone_radius, 
    neck_radius, 
    head_base_radius, 
    head_top_radius,
    color,
    has_hair
):
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

    if has_hair:
        ops_obj.particle_system_add()
        psys = ctx_obj.particle_systems[-1]
        psys.settings.type = 'HAIR'
        psys.settings.hair_length = 0.17
        psys.settings.child_type = 'INTERPOLATED'

if __name__ == '__main__':
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.object.camera_add(location=(0, -30, 1), rotation=(math.radians(90), 0, 0))
    bpy.context.scene.camera = bpy.context.object

    # Vertex offsets.
    hip_offset = (-0.5, 0, 0)
    leg_offset = (0, 0, -3)
    spine_offset = (0, 0, 2.5)
    collar_bone_offset = (-1, 0, 0)
    arm_offset = (3, 0, 0)
    neck_base_offset = (0, 0, 0.5)
    neck_offset = (0, 0, 1)
    head_base_offset = (0, 0.6, 0)
    head_top_offset = (0, 0, 1)

    # Randomize vertex offsets.
    hip_offset = (
        random.uniform(0.4, 1), 
        random.uniform(-0.2, 0.2), 
        random.uniform(-0.4, 0)
    )
    leg_offset = (
        0, 
        0, 
        random.uniform(-4, -0.5)
    )
    spine_offset = (
        0, 
        random.uniform(-0.3, 0.3), 
        random.uniform(1.5, 4)
    )
    collar_bone_offset = (
        random.uniform(0.5, 1.5), 
        random.uniform(-0.2, 0.2), 
        random.uniform(-0.2, 0.2)
    )
    arm_offset = (
        random.uniform(0.7, 3),
        0,
        0,
    )
    neck_offset = (
        0,
        0,
        random.uniform(0, 0.5)
    )
    neck_offset = (
        0, 
        random.uniform(-0.3, 0.3), 
        random.uniform(0.3, 1)
    )
    head_base_offset = (
        0,
        random.uniform(-1, 1),
        random.uniform(0, 0.7)
    )
    head_top_offset = (
        0,
        random.uniform(-2, 2),
        random.uniform(0, 2)
    )

    # Randomize radii.
    spine_radius = random.uniform(0.4, 1)
    spine_radius = (spine_radius, spine_radius)
    root_radius = random.uniform(0.4, 1)
    root_radius = (root_radius, root_radius)
    hip_radius = random.uniform(0.4, 1)
    hip_radius = (hip_radius, hip_radius)
    collar_bone_radius = random.uniform(0.4, 0.6)
    collar_bone_radius = (collar_bone_radius, collar_bone_radius)
    neck_radius = random.uniform(0.3, 1)
    neck_radius = (neck_radius, neck_radius)
    head_base_radius = random.uniform(0.5, 2)
    head_base_radius = (head_base_radius, head_base_radius)
    head_top_radius = random.uniform(0.5, 2)
    head_top_radius = (head_top_radius, head_top_radius)

    # Randomize color.
    color = (
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.uniform(0, 1),
        1
    )

    generate(
        hip_offset,
        leg_offset,
        spine_offset,
        collar_bone_offset,
        arm_offset,
        neck_base_offset,
        neck_offset,
        head_base_offset,
        head_top_offset,
        spine_radius, 
        root_radius, 
        hip_radius, 
        collar_bone_radius, 
        neck_radius, 
        head_base_radius, 
        head_top_radius,
        color,
        True
    )

    # TODO: install CUDA on this docker image.

    # Enable GPU rendering.
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.scene.cycles.device = 'GPU'

    #bpy.data.scenes['Scene'].render.engine = 'CYCLES'

    # Set world background color.
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (1, 1, 1)
    bg.inputs[1].default_value = 1.0

    # Render.
    bpy.ops.object.select_all(action='DESELECT')
    camera = bpy.context.scene.objects['Camera']
    camera.select_set(True)
    
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    plane_position = [0, -30, 0]
    plane_width = 5
    plane_height = 5
    horizontal_steps = 5
    vertical_steps = 5
    horizontal_step_size = plane_width / horizontal_steps
    vertical_step_size = plane_height / vertical_steps
    start_translation = (
        -plane_width / 2,
        0,
        -plane_height / 2
    )
    bpy.ops.transform.translate(value=start_translation, orient_type='LOCAL')

    #'''
    files_rendered = 0
    for x_step in range(horizontal_steps):
        for y_step in range(vertical_steps):
            camera_translation = (
                x_step * horizontal_step_size,
                0,
                y_step * vertical_step_size
            )

            bpy.ops.transform.translate(value=camera_translation, orient_type='LOCAL')
            bpy.context.scene.render.filepath = f'data/renders/render{files_rendered}.png'
            bpy.ops.render.render(write_still = True)
            files_rendered += 1
        
            reset_translation = (
                -x_step * horizontal_step_size,
                0,
                -y_step * vertical_step_size
            )
            bpy.ops.transform.translate(value=reset_translation, orient_type='LOCAL')
    #'''
    '''
    for i in range(horizontal_steps):
        camera_translation = (
            0,
            0,
            i * horizontal_step_size
        )
        bpy.ops.transform.translate(value=camera_translation, orient_type='LOCAL')
        bpy.context.scene.render.filepath = f'data/renders/render{files_rendered}.png'
        bpy.ops.render.render(write_still = True)
        files_rendered += 1
    '''     
    bpy.ops.wm.save_as_mainfile(filepath='data/blend_files/test.blend')
