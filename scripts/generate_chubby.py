import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_plane_add(
    size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1)
)
#obj = bpy.data.objects['Plane']

bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.merge(type='CENTER')
bpy.ops.object.editmode_toggle()

bpy.ops.object.modifier_add(type='MIRROR')
bpy.ops.object.modifier_add(type='SKIN')
bpy.ops.object.modifier_add(type='SUBSURF')

bpy.context.object.modifiers['Skin'].use_smooth_shade = True
bpy.context.object.modifiers["Subdivision"].levels = 2

#bpy.ops.object.data.vertices[0].select = True
#bpy.ops.transform.translate(
#    value=(-1, 0, 0), 
#    orient_type='GLOBAL', 
#    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
#    orient_matrix_type='GLOBAL', 
#    constraint_axis=(True, False, False), 
#    mirror=True, 
#    use_proportional_edit=False, 
#    proportional_edit_falloff='SMOOTH', 
#    proportional_size=1
#)

bpy.ops.object.editmode_toggle()
bpy.context.object.data.vertices[0].select = True
bpy.ops.mesh.extrude_vertices_move(
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

bpy.ops.mesh.extrude_vertices_move(
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

# Toggle in and out of object mode to synchronize mesh.
bpy.ops.object.editmode_toggle()

#bpy.context.object.data.vertices[0].select = True
bpy.ops.object.data.mesh.verts[0].select = True
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

bpy.ops.wm.save_as_mainfile(filepath='data/blend_files/test.blend')