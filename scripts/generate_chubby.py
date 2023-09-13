import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_plane_add(
    size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1)
)

bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.merge(type='CENTER')
bpy.ops.object.editmode_toggle()

bpy.ops.object.modifier_add(type='MIRROR')
bpy.ops.object.modifier_add(type='SKIN')
bpy.ops.object.modifier_add(type='SUBSURF')

bpy.ops.wm.save_as_mainfile(filepath='data/blend_files/test.blend')