"""Contact-only fruit placement; no object attachment or state writes in rollout."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--order', choices=['orange_first', 'banana_first'], required=True)
parser.add_argument('--plate', choices=['purple', 'green'], default='purple')
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--max-steps', type=int, default=1600)
parser.add_argument('--replay-episode',type=Path,help='Independently audit saved action and joint targets without rendering.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_dir.exists():
    raise FileExistsError(args.output_dir)
args.enable_cameras = args.replay_episode is None
app = AppLauncher(args).app


def physics_asset(source, target, dynamic):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    if target.exists():
        return target
    stage = Usd.Stage.CreateNew(str(target))
    root = UsdGeom.Xform.Define(stage, '/Asset').GetPrim()
    root.GetReferences().AddReference('./'+source.name)
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(not dynamic)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.12 if dynamic else 1.)
    material = UsdShade.Material.Define(stage, '/Asset/ContactMaterial')
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(.9)
    mat.CreateDynamicFrictionAttr(.7)
    mat.CreateRestitutionAttr(0.)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull' if dynamic else 'none')
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(.002)
            collision.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
    stage.GetRootLayer().Save()
    return target


def main():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    import isaaclab.sim as sim_utils
    import isaaclab.envs.mdp as mdp
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_inv
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import FrankaCubeLiftEnvCfg
    from track_b_envs.envs.microwave_closing.microwave_env_cfg import Events, Terminations
    from track_b_envs.envs.common.observations import franka_proprio
    from track_b_envs.envs.common.compositing import RealBackgroundCompositor
    from track_b_envs.envs.common.constants import FOREGROUND_SEMANTIC_TAG, FRANKA_HOME_JOINTS_RAD
    from track_b_envs.envs.common import task_scene as original
    from track_b_envs.envs.two_fruit_placement import two_fruit_scene as layout
    from track_b_envs.collection.experts.common import absolute_pose_to_relative_action, advance_pose_toward_waypoint

    orange_asset = physics_asset(layout.YCB_ORANGE_USD, layout.YCB_ASSET_ROOT/'017_orange_physics.usda', True)
    plate_asset = physics_asset(layout.YCB_PLATE_USD, layout.YCB_ASSET_ROOT/'029_plate_physics.usda', False)

    @configclass
    class Obs:
        @configclass
        class Policy(ObservationGroupCfg):
            proprio = ObservationTermCfg(func=franka_proprio, params={'robot_cfg':SceneEntityCfg('robot'), 'ee_frame_cfg':SceneEntityCfg('ee_frame')})
        policy: Policy = Policy()

    @configclass
    class Cfg(FrankaCubeLiftEnvCfg):
        observations: Obs = Obs()
        events: Events = Events()
        terminations: Terminations = Terminations()
        def __post_init__(self):
            super().__post_init__()
            self.commands = self.curriculum = self.rewards = None
            self.scene = layout.build_two_fruit_scene_cfg()
            if args.replay_episode is not None:
                self.scene.policy_camera=None
            for name, asset, scale, position, rotation in [
                ('banana', original.YCB_BANANA_USD, layout.YCB_BANANA_SCALE, layout.BANANA_INITIAL_POSITION_M, layout.BANANA_STABLE_ROTATION_WXYZ),
                ('orange', orange_asset, layout.YCB_ORANGE_SCALE, layout.ORANGE_INITIAL_POSITION_M, (1.,0.,0.,0.)),
            ]:
                setattr(self.scene, name, RigidObjectCfg(prim_path='{ENV_REGEX_NS}/'+name.capitalize(),
                    spawn=sim_utils.UsdFileCfg(usd_path=str(asset), scale=(scale,)*3, activate_contact_sensors=True,
                        semantic_tags=[FOREGROUND_SEMANTIC_TAG], rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                            max_depenetration_velocity=2.)),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=rotation)))
                setattr(self.scene, name+'_contact', ContactSensorCfg(prim_path='{ENV_REGEX_NS}/'+name.capitalize(),
                    filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/panda_leftfinger','{ENV_REGEX_NS}/Robot/panda_rightfinger',
                        '{ENV_REGEX_NS}/'+('TargetPlate' if args.plate=='purple' else 'DistractorPlate')],
                    history_length=5, max_contact_data_count_per_prim=256))
            for name in ['target_plate','distractor_plate']:
                old = getattr(self.scene, name)
                old.spawn.usd_path = str(plate_asset)
            self.scene.robot.actuators['panda_hand'].stiffness = 6000.
            self.scene.robot.actuators['panda_hand'].effort_limit_sim = 35.
            self.actions.arm_action.scale = (.05,.05,.05,.08,.08,.08)
            self.sim.dt = .01
            self.decimation = self.sim.render_interval = 5
            self.episode_length_s = 150.

    cfg = Cfg(); cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg)
    process = None
    args.output_dir.mkdir(parents=True)
    try:
        env.reset(seed=20260827)
        for _ in range(10): env.step(torch.zeros(env.action_space.shape,device=env.device))
        env.reset(seed=20260827)
        robot = env.scene['robot']
        assert torch.allclose(robot.data.joint_pos[0,:7],torch.tensor(FRANKA_HOME_JOINTS_RAD,device=env.device),atol=1e-5)
        if args.replay_episode is not None:
            import shutil
            with np.load(args.replay_episode) as source:
                replay={key:source[key].copy() for key in source.files}
            arm_ids,_=robot.find_joints('panda_joint[1-7]')
            term=env.action_manager.get_term('arm_action');cartesian_apply=term.apply_actions
            selected={'target':None}
            def replay_apply():
                if selected['target'] is None:cartesian_apply()
                else:robot.set_joint_position_target(selected['target'],joint_ids=arm_ids)
            term.apply_actions=replay_apply
            errors=[];plate_history=[]
            for step,action in enumerate(replay['actions']):
                current=np.concatenate([robot.data.joint_pos[0].cpu().numpy(),*[env.scene[n].data.root_state_w[0].cpu().numpy() for n in ['banana','orange']]])
                errors.append(float(np.max(abs(current-replay['state'][step]))))
                joint=replay['joint_position_command'][step]
                selected['target']=None if np.isnan(joint).any() else torch.tensor(joint[None],device=env.device,dtype=torch.float32)
                env.step(torch.tensor(action[None],device=env.device,dtype=torch.float32))
                plate_history.append([float(env.scene[n+'_contact'].data.force_matrix_w_history[0].reshape(5,3,3)[:,2].norm(dim=-1).max()) for n in ['banana','orange']])
                if step%200==0:print(json.dumps(dict(replay_step=step,max_state_error=max(errors))),flush=True)
            names=['banana','orange'];initial={n:replay['state'][0,9+13*i:16+13*i].tolist() for i,n in enumerate(names)}
            final={n:env.scene[n].data.root_state_w[0].tolist() for n in names}
            home_error=float((robot.data.joint_pos[0,:7]-torch.tensor(FRANKA_HOME_JOINTS_RAD,device=env.device)).abs().max())
            verified={};lifted={}
            for i,n in enumerate(names):
                carrying=replay['phase']==n+':transfer'
                verified[n]=bool(carrying.any() and (replay['finger_contact_forces'][carrying]>.1).all(1).mean()>.95)
                lifted[n]=bool(carrying.any() and np.min(replay['state'][carrying,11+13*i])-initial[n][2]>.07)
            supported=bool(np.min(np.asarray(plate_history)[-20:],axis=0).min()>.05)
            settled=all(float(env.scene[n].data.root_lin_vel_w.norm())<.02 and float(env.scene[n].data.root_ang_vel_w.norm())<.3 for n in names)
            success=bool(max(errors)<.005 and home_error<.015 and supported and settled and all(verified.values()) and all(lifted.values()))
            report=dict(success=success,order=['orange','banana'] if args.order=='orange_first' else ['banana','orange'],plate=args.plate,
                frames=len(replay['actions']),fps=20,resolution=[1280,720],initial=initial,final=final,
                bilateral_contact_verified=verified,physical_lift_verified=lifted,
                final_plate_contact_N=dict(zip(names,plate_history[-1])),final_home_joint_max_error_rad=home_error,
                max_replay_state_error=max(errors),validation='Independent non-rendering physics replay of recorded Cartesian actions and joint-position homing commands',
                physics_contract='Dynamic fruits, physical finger contacts and gravity only; no object-pose writes or attachments after reset',
                control_contract='7D Cartesian actions during manipulation plus recorded joint_position_command during homing',
                source_episode=str(args.replay_episode.resolve()))
            (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
            if not success:raise RuntimeError('Independent physics replay failed')
            shutil.copy2(args.replay_episode,args.output_dir/'episode.npz')
            shutil.copy2(args.replay_episode.parent/'demo.mp4',args.output_dir/'demo.mp4')
            return
        camera = env.scene['policy_camera']
        def ee():
            d=env.scene['ee_frame'].data
            return torch.cat((d.target_pos_w[:,0],d.target_quat_w[:,0]),-1)
        home=ee().clone(); command=home.clone()
        home_joints=robot.data.default_joint_pos[:,:7].clone()
        arm_ids,_=robot.find_joints('panda_joint[1-7]')
        arm_action=env.action_manager.get_term('arm_action')
        cartesian_apply=arm_action.apply_actions
        homing={'active':False,'command':home_joints.clone()}
        def apply_arm_control():
            if homing['active']:
                robot.set_joint_position_target(homing['command'],joint_ids=arm_ids)
            else:
                cartesian_apply()
        arm_action.apply_actions=apply_arm_control
        scale=torch.tensor(cfg.actions.arm_action.scale,device=env.device)
        compositor=RealBackgroundCompositor(env.device)
        env.sim.render(); camera.update(0.,force_recompute=True)
        process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output_dir/'demo.mp4')],stdin=subprocess.PIPE)
        order=['orange','banana'] if args.order=='orange_first' else ['banana','orange']
        index=0; phase='rest'; wait=0; target=home.clone(); grasp=None; carry_offset=None
        initial={n:env.scene[n].data.root_pose_w[0].cpu().numpy().copy() for n in order}
        rows=[]; actions=[]; phases=[]; contacts=[]; joint_commands=[]; lifted={n:False for n in order}; verified={n:False for n in order}
        settled=0; success=False
        nominal_q=torch.tensor([original.BANANA_STABLE_ROTATION_WXYZ],device=env.device)
        nominal_tool=torch.tensor([[0.,.95371695,.30070580,0.]],device=env.device)
        offset_local=quat_apply_inverse(nominal_q,torch.tensor([[.01433941,-.02047880,0.]],device=env.device))*(layout.YCB_BANANA_SCALE/original.BANANA_SCALE)
        plate=np.array(layout.TARGET_PLATE_POSITION_M if args.plate=='purple' else layout.DISTRACTOR_PLATE_POSITION_M)
        destinations={n:np.array(p)+plate-np.array(layout.TARGET_PLATE_POSITION_M) for n,p in [('banana',layout.BANANA_FINAL_POSITION_M),('orange',layout.ORANGE_FINAL_POSITION_M)]}
        for step in range(args.max_steps):
            name=order[index]; obj=env.scene[name]; pose=obj.data.root_pose_w
            contact_force=env.scene[name+'_contact'].data.force_matrix_w_history[0].reshape(5,3,3).norm(dim=-1).max(dim=0).values
            finger_force=contact_force[:2]
            if phase=='rest': target=home.clone()
            elif phase=='approach':
                grasp=ee().clone()
                if name=='banana':
                    grasp[:,:3]=pose[:,:3]+quat_apply(pose[:,3:7],offset_local)
                    grasp[:,2]+=.002
                    grasp[:,3:]=quat_mul(pose[:,3:7],quat_mul(quat_inv(nominal_q),nominal_tool))
                else:
                    grasp[:,:3]=pose[:,:3]; grasp[:,2]-=.008
                    grasp[:,3:]=torch.tensor([0.,1.,0.,0.],device=env.device)
                target=grasp.clone(); target[:,2]=.18
            elif phase in ('descend','grasp'): target=grasp.clone()
            elif phase=='lift': target=grasp.clone(); target[:,2]=.20
            elif phase in ('transfer','place','release'):
                target=grasp.clone()
                target[:,:3]=torch.tensor(destinations[name],device=env.device)+carry_offset
                target[:,2]=.20 if phase=='transfer' else target[:,2]+.012
            elif phase=='retreat': target=grasp.clone(); target[:,:2]=torch.tensor(destinations[name][:2],device=env.device);target[:,2]=.23
            elif phase in ('between_home','home','final'): target=home.clone()
            close=phase in ('grasp','lift','transfer','place')
            use_joint_home=phase in ('between_home','home','final')
            if use_joint_home and not homing['active']:
                homing['command']=robot.data.joint_pos[:,:7].clone()
            homing['active']=use_joint_home
            if use_joint_home:
                homing['command']+=(home_joints-homing['command']).clamp(-.025,.025)
            joint_commands.append(homing['command'][0].cpu().numpy().copy() if use_joint_home else np.full(7,np.nan))
            speed=.10 if phase in ('approach','transfer','retreat','between_home','home') else .045
            command=advance_pose_toward_waypoint(command,target,speed*.05,.05)
            action=absolute_pose_to_relative_action(ee(),command,torch.tensor([-1. if close else 1.],device=env.device),scale,max_translation_step_m=.05,max_rotation_step_rad=.08)
            frame=compositor.compose(camera.data.output['rgb'],camera.data.output['semantic_segmentation'],camera.data.intrinsic_matrices)[0].cpu().numpy()
            process.stdin.write(frame.tobytes())
            rows.append(np.concatenate([robot.data.joint_pos[0].cpu().numpy(),*[env.scene[n].data.root_state_w[0].cpu().numpy() for n in ['banana','orange']]]))
            actions.append(action[0].cpu().numpy());phases.append(name+':'+phase);contacts.append(finger_force.cpu().numpy())
            if step%80==0:
                Image.fromarray(frame).save(args.output_dir/f'frame_{step:04d}.png')
                print(json.dumps(dict(step=step,fruit=name,phase=phase,ee=ee()[0,:3].tolist(),target=target[0,:3].tolist(),fruit_position=pose[0,:3].tolist(),forces=finger_force.tolist(),robot_joints=robot.data.joint_pos[0,:7].tolist())),flush=True)
            env.step(action);wait+=1
            error=float((ee()[0,:3]-target[0,:3]).norm())
            orientation_error=2*math.acos(min(1.,float((ee()[0,3:]*target[0,3:]).sum().abs())))
            next_phase=None
            if phase=='rest' and wait>=20: next_phase='approach'
            elif phase=='approach' and error<.008 and orientation_error<.12: next_phase='descend'
            elif phase=='descend' and error<.007: next_phase='grasp'
            elif phase=='grasp':
                verified[name]=bool((finger_force>.15).all())
                if wait>=20:
                    if not verified[name]: raise RuntimeError(f'No bilateral contact for {name}')
                    next_phase='lift'
            elif phase=='lift' and error<.012:
                lifted[name]=bool(float(obj.data.root_pos_w[0,2])-initial[name][2]>.07)
                if not lifted[name] or not bool((finger_force>.1).all()): raise RuntimeError('Object not physically held after lift')
                carry_offset=ee()[:,:3]-obj.data.root_pos_w
                next_phase='transfer'
            elif phase=='transfer' and error<.012: next_phase='place'
            elif phase=='place' and error<.012: next_phase='release'
            elif phase=='release' and wait>=24: next_phase='retreat'
            elif phase=='retreat' and error<.015:
                if index==0:
                    index=1
                    # Retract into the central workspace after the far-away
                    # banana grasp, avoiding a local IK limit on the next pick.
                    next_phase='between_home' if name=='banana' else 'approach'
                else: next_phase='home'
            elif phase=='between_home' and float((robot.data.joint_pos[:,:7]-home_joints).abs().max())<.015:
                command=ee().clone()
                next_phase='approach'
            elif phase=='home' and float((robot.data.joint_pos[:,:7]-home_joints).abs().max())<.015: next_phase='final'
            elif phase=='final':
                good=all(float((env.scene[n].data.root_pos_w[0,:2]-torch.tensor(destinations[n][:2],device=env.device)).norm())<.045
                    and float(env.scene[n].data.root_lin_vel_w.norm())<.02
                    and float(env.scene[n].data.root_ang_vel_w.norm())<.3
                    and float(env.scene[n+'_contact'].data.force_matrix_w_history[0].reshape(5,3,3)[:,2].norm(dim=-1).max())>.05
                    for n in order)
                settled=settled+1 if good else 0
                if settled>=20: success=True;break
            if next_phase:
                print(json.dumps(dict(transition=phase+'->'+next_phase,step=step)),flush=True)
                phase=next_phase;wait=0
        process.stdin.close();code=process.wait();process=None
        if code: raise RuntimeError('Encoder failed')
        np.savez_compressed(args.output_dir/'episode.npz',state=rows,actions=actions,phase=phases,finger_contact_forces=contacts,joint_position_command=joint_commands)
        report=dict(success=success,order=order,plate=args.plate,frames=len(rows),fps=20,resolution=[1280,720],bilateral_contact_verified=verified,physical_lift_verified=lifted,
            final_plate_contact_N={n:float(env.scene[n+'_contact'].data.force_matrix_w_history[0].reshape(5,3,3)[:,2].norm(dim=-1).max()) for n in order},
            final_home_joint_max_error_rad=float((robot.data.joint_pos[:,:7]-home_joints).abs().max()),
            control_contract='Cartesian differential IK during manipulation; bounded joint-position PD targets during between_home/home/final (saved separately in joint_position_command)',
            initial={k:v.tolist() for k,v in initial.items()},final={n:env.scene[n].data.root_state_w[0].tolist() for n in order},physics_contract='dynamic rigid fruits; mesh collisions; no attachment, pose writes or auxiliary transport forces after reset')
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        if not success: raise RuntimeError('Fruit task did not pass release-and-settle checks')
    finally:
        if process:
            process.stdin.close();process.wait()
        env.close()


if __name__=='__main__':
    try: main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally: app.close()
