"""Build a passive hinged appliance from the exact initial-preview geometry.

Call build_asset after launching Isaac Sim. The appliance remains fixed, while
the door's revolute joint is moved only by physical robot contacts.
"""

from pathlib import Path
import math


def build_asset(output: Path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    import render_close_microwave_initial_state as ref

    boxes = []
    original = ref._add_box
    ref._add_box = lambda scene, name, **kw: boxes.append(dict(name=name, **kw))
    try:
        ref._add_microwave(None)
    finally:
        ref._add_box = original
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return output
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/Microwave')
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, '/Microwave/Body')
    fixed = UsdPhysics.FixedJoint.Define(stage, '/Microwave/FixedBaseJoint')
    fixed.CreateBody1Rel().SetTargets([body.GetPath()])
    door = UsdGeom.Xform.Define(stage, '/Microwave/Door')
    hinge = (0.170, 0.1585, 0.0)
    door.AddTranslateOp().Set(Gf.Vec3d(*hinge))
    for prim, mass in ((body.GetPrim(), 8.0), (door.GetPrim(), .45)):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    material = UsdShade.Material.Define(stage, '/Microwave/PhysicsMaterial')
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.65)
    physics.CreateDynamicFrictionAttr(.5)
    physics.CreateRestitutionAttr(0.0)
    original_yaw = math.radians(180-ref.MICROWAVE_DOOR_OPEN_DEG)
    for box in boxes:
        is_door = box['name'].startswith('microwave_door_')
        parent = '/Microwave/Door' if is_door else '/Microwave/Body'
        cube = UsdGeom.Cube.Define(stage, parent+'/'+box['name'])
        cube.CreateSizeAttr(1.0)
        pos = box['local_position']
        if is_door:
            dx, dy = pos[0]-hinge[0], pos[1]-hinge[1]
            along = math.cos(original_yaw)*dx + math.sin(original_yaw)*dy
            outward = -math.sin(original_yaw)*dx + math.cos(original_yaw)*dy
            pos = (-along, -outward, pos[2])
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(*box['size']))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        collision.CreateContactOffsetAttr(.002)
        collision.CreateRestOffsetAttr(0.0)
        look = UsdShade.Material.Define(stage, '/Microwave/Looks/'+box['name'])
        shader = UsdShade.Shader.Define(stage, str(look.GetPath())+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        from pxr import Sdf
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*box['color']))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(box.get('roughness',.92))
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(look)
        binding.Bind(material, materialPurpose='physics')
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/Microwave/DoorHingeJoint')
    joint.CreateBody0Rel().SetTargets([body.GetPath()])
    joint.CreateBody1Rel().SetTargets([door.GetPath()])
    joint.CreateAxisAttr('Z')
    joint.CreateLocalPos0Attr(Gf.Vec3f(*hinge))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateLowerLimitAttr(-100.0)
    joint.CreateUpperLimitAttr(0.0)
    joint.CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
    return output
