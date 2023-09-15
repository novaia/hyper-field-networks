import bpy
import bmesh
import random

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
    head_top_radius
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
    ctx_obj.modifiers["Subdivision"].levels = 2

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

if __name__ == '__main__':
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Vertex offsets.
    hip_offset = (-0.5, 0, 0)
    leg_offset = (0, 0, -3)
    spine_offset = (0, 0, 2.5)
    collar_bone_offset = (-1, 0, 0)
    arm_offset = (-3, 0, 0)
    neck_base_offset = (0, 0, 0.5)
    neck_offset = (0, 0, 1)
    head_base_offset = (0, 0.6, 0)
    head_top_offset = (0, 0, 1)

    # Randomize radii.
    spine_radius = random.uniform(0.4, 1)
    spine_radius = (spine_radius, spine_radius)
    root_radius = random.uniform(0.4, 1)
    root_radius = (root_radius, root_radius)
    hip_radius = random.uniform(0.4, 1)
    hip_radius = (hip_radius, hip_radius)
    collar_bone_radius = random.uniform(0.4, 0.6)
    collar_bone_radius = (collar_bone_radius, collar_bone_radius)
    neck_radius = random.uniform(0.2, 0.6)
    neck_radius = (neck_radius, neck_radius)
    head_base_radius = random.uniform(0.5, 2)
    head_base_radius = (head_base_radius, head_base_radius)
    head_top_radius = random.uniform(0.5, 2)
    head_top_radius = (head_top_radius, head_top_radius)

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
        head_top_radius
    )

    bpy.ops.wm.save_as_mainfile(filepath='data/blend_files/test.blend')
