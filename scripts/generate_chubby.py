import bpy
import bmesh

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

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

# Create hip by extruding root to the left.
ops_mesh.extrude_vertices_move(
    MESH_OT_extrude_verts_indiv={'mirror':False}, 
    TRANSFORM_OT_translate={
        'value':(-1, 0, 0), 
        'orient_type':'GLOBAL', 
        'orient_matrix':((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
        'orient_matrix_type':'GLOBAL', 
        'constraint_axis':(True, False, False), 
        'mirror':True
    }
)

b_mesh.verts.ensure_lookup_table()
b_mesh.verts[0].select = False
b_mesh.verts[1].select = True

# Create leg by extruding hip downwards.
ops_mesh.extrude_vertices_move(
    MESH_OT_extrude_verts_indiv={'mirror':False}, 
    TRANSFORM_OT_translate={
        'value':(0, 0, -3), 
        'orient_type':'GLOBAL', 
        'orient_matrix':((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
        'orient_matrix_type':'GLOBAL', 
        'constraint_axis':(False, False, True), 
        'mirror':True
    }
)

b_mesh.verts.ensure_lookup_table()
b_mesh.verts[1].select = False
b_mesh.verts[0].select = True

# Create spine by extruding root upwards.
bpy.ops.mesh.extrude_vertices_move(
    MESH_OT_extrude_verts_indiv={'mirror':False}, 
    TRANSFORM_OT_translate={
        'value':(0, 0, 2.5), 
        'orient_type':'GLOBAL', 
        'orient_matrix':((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
        'orient_matrix_type':'GLOBAL', 
        'constraint_axis':(False, False, True), 
        'mirror':True
    }
)

# The indexes change as new vertices are created.
# I think 1 might always be the last created vertex, while 0 is the root.
# Need to test.
b_mesh.verts.ensure_lookup_table()
b_mesh.verts[0].select = False
b_mesh.verts[1].select = True

# Create collar bone by extruding spine to the left.
ops_mesh.extrude_vertices_move(
    MESH_OT_extrude_verts_indiv={'mirror':False}, 
    TRANSFORM_OT_translate={
        'value':(-1, 0, 0), 
        'orient_type':'GLOBAL', 
        'orient_matrix':((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
        'orient_matrix_type':'GLOBAL', 
        'constraint_axis':(True, False, False), 
        'mirror':True
    }
)

b_mesh.verts.ensure_lookup_table()
b_mesh.verts[4].select = True
b_mesh.verts[1].select = False

# Create arm by extruding collar bone to the left.
ops_mesh.extrude_vertices_move(
    MESH_OT_extrude_verts_indiv={'mirror':False}, 
    TRANSFORM_OT_translate={
        'value':(-3, 0, 0), 
        'orient_type':'GLOBAL', 
        'orient_matrix':((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
        'orient_matrix_type':'GLOBAL', 
        'constraint_axis':(True, False, False), 
        'mirror':True
    }
)

bpy.ops.wm.save_as_mainfile(filepath='data/blend_files/test.blend')