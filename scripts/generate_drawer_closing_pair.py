"""Close the same open upper drawer by fingertip push, palm push, or handle grasp."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',choices=['panel_push','handle_grasp','palm_push'],required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--max-steps',type=int,default=800)
parser.add_argument('--keep-gripper-open',action='store_true')
parser.add_argument('--panel-lateral-offset',type=float,default=0.)
parser.add_argument('--panel-height-offset',type=float,default=.020)
parser.add_argument('--balance-panel-fingers',action='store_true')
parser.add_argument('--panel-pitch-deg',type=float,default=0.)
parser.add_argument('--balance-gain',type=float,default=.000002)
parser.add_argument('--fingertip-stiffness',type=float,default=0.)
parser.add_argument('--fingertip-damping',type=float,default=100.)
parser.add_argument('--palm-offset',type=float,default=.015)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.keep_gripper_open and args.method=='handle_grasp':
    parser.error('--keep-gripper-open is not supported for handle_grasp')
if args.output_dir.exists(): raise FileExistsError(args.output_dir)
args.enable_cameras=True
app=AppLauncher(args).app


def main():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils.math import quat_apply,quat_apply_inverse,quat_mul
    from track_b_envs.envs.drawer_subtasks.drawer_subtasks_env_cfg import PutObjectInUpperDrawerAndCloseEnvCfg
    from track_b_envs.envs.microwave_closing.microwave_env_cfg import Terminations,ContactPointHistorySensor
    from track_b_envs.envs.common.compositing import RealBackgroundCompositor
    from track_b_envs.envs.common.constants import FRANKA_HOME_JOINTS_RAD
    from track_b_envs.collection.experts.common import drawer_subtask_state,absolute_pose_to_relative_action,advance_pose_toward_waypoint

    cfg=PutObjectInUpperDrawerAndCloseEnvCfg()
    cfg.sim.device=args.device
    cfg.terminations=Terminations();cfg.rewards=None
    cfg.episode_length_s=100.
    cfg.scene.cabinet.spawn.activate_contact_sensors=True
    if args.fingertip_stiffness>0:
        original_spawn=cfg.scene.robot.spawn.func
        def spawn_with_compliant_fingertips(prim_path,spawn_cfg,*positional,**kwargs):
            import isaaclab.sim as sim_utils
            from pxr import UsdShade
            prim=original_spawn(prim_path,spawn_cfg,*positional,**kwargs)
            root=str(prim.GetPath())
            material_path=root+'/OpenPushFingertipMaterial'
            material=sim_utils.RigidBodyMaterialCfg(static_friction=.5,dynamic_friction=.5,
                compliant_contact_stiffness=args.fingertip_stiffness,
                compliant_contact_damping=args.fingertip_damping)
            material.func(material_path,material)
            for finger in ('panda_leftfinger','panda_rightfinger'):
                finger_prim=prim.GetStage().GetPrimAtPath(root+'/'+finger)
                UsdShade.MaterialBindingAPI.Apply(finger_prim).Bind(
                    UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path)),
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose='physics')
            return prim
        cfg.scene.robot.spawn.func=spawn_with_compliant_fingertips
    contact_bodies=['panda_leftfinger','panda_rightfinger','panda_hand']
    if args.method=='palm_push':
        contact_bodies += ['panda_link'+str(i) for i in range(7,-1,-1)]
    cfg.scene.closing_contact=ContactSensorCfg(class_type=ContactPointHistorySensor,
        prim_path='{ENV_REGEX_NS}/Cabinet/UpperDrawer',track_contact_points=True,
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+body for body in contact_bodies],
        history_length=5,max_contact_data_count_per_prim=256)
    env=ManagerBasedRLEnv(cfg=cfg);process=None
    args.output_dir.mkdir(parents=True)
    try:
        env.reset(seed=20260827)
        for _ in range(10):env.step(torch.zeros(env.action_space.shape,device=env.device))
        obs,_=env.reset(seed=20260827)
        robot=env.scene['robot'];cabinet=env.scene['cabinet'];sensor=env.scene['closing_contact']
        assert torch.allclose(robot.data.joint_pos[0,:7],torch.tensor(FRANKA_HOME_JOINTS_RAD,device=env.device),atol=1e-5)
        camera=env.scene['policy_camera'];compositor=RealBackgroundCompositor(env.device)
        env.sim.render();camera.update(0.,force_recompute=True)
        scale=torch.tensor(cfg.actions.arm_action.scale,device=env.device)
        task=drawer_subtask_state(env,'UpperDrawerJoint');home=task['ee_pose'].clone();command=home.clone()
        direction=quat_apply(task['handle_pose'][:,3:],torch.tensor([[0.,1.,0.]],device=env.device))
        lateral=quat_apply(task['handle_pose'][:,3:],torch.tensor([[1.,0.,0.]],device=env.device))
        drawer_id=cabinet.find_bodies('UpperDrawer')[0][0]
        horizontal=quat_mul(task['handle_pose'][:,3:],torch.tensor([[.70710678,.70710678,0.,0.]],device=env.device))
        top=torch.tensor([[0.,1.,0.,0.]],device=env.device)
        palm_orientation=quat_mul(task['handle_pose'][:,3:],torch.tensor([[0.,.70710678,.70710678,0.]],device=env.device))
        phase='rest';wait=0;retreat=None;success=False;grasp_verified=False;panel_pitch_adjust=math.radians(args.panel_pitch_deg)
        states=[];actions=[];forces=[];points=[];local_points=[];phases=[];openings=[];pitch_history=[]
        initial=obs['critic'][0].cpu().numpy().copy()
        binding=dict(schema_version=1,provenance='Exported from the instantiated simulation, not inferred from human video',
            asset=cfg.scene.cabinet.spawn.usd_path,cabinet_root_pose_wxyz=cabinet.data.root_pose_w[0].tolist(),
            drawer=dict(scene_entity='cabinet',body_prim='/World/envs/env_0/Cabinet/UpperDrawer',joint='UpperDrawerJoint',initial_opening_m=float(task['joint_position'][0])),
            handle=dict(scene_entity='cabinet_frame',frame='task_handle',initial_pose_wxyz=task['handle_pose'][0].tolist()),
            panel=dict(body_prim='/World/envs/env_0/Cabinet/UpperDrawer',
                contact_height_offset_from_handle_m=args.panel_height_offset,
                contact_lateral_offset_from_handle_m=args.panel_lateral_offset),
            controller='velocity-limited Cartesian waypoints + differential IK + physical contacts',
            success=dict(max_opening_m=.01,settle_duration_s=.5,requires_robot_withdrawal=True))
        (args.output_dir/'runtime_bindings.json').write_text(json.dumps(binding,indent=2)+'\n')
        process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output_dir/'demo.mp4')],stdin=subprocess.PIPE)
        for step in range(args.max_steps):
            task=drawer_subtask_state(env,'UpperDrawerJoint')
            ee=task['ee_pose'];handle=task['handle_pose'][:,:3];q=float(task['joint_position'][0])
            force=sensor.data.force_matrix_w_history[0].reshape(5,len(contact_bodies),3)
            f=force.norm(dim=-1).max(dim=0).values
            if args.balance_panel_fingers and args.method=='panel_push' and phase=='close' and float(f[:2].sum())>.2:
                # Small contact-feedback pitch correction balances the two
                # open fingertips against the planar face. Positive local-X
                # pitch advances the upper finger, so correct in reverse.
                panel_pitch_adjust-=float(np.clip(float(f[0]-f[1])*args.balance_gain,-.00025,.00025))
                panel_pitch_adjust=float(np.clip(panel_pitch_adjust,-.035,.035))
            # Fingertip/handle modes point the tool into the drawer; the palm
            # mode points fingers down and presents the palm's side instead.
            target=ee.clone();target[:,3:]=horizontal
            if args.method=='palm_push':target[:,3:]=palm_orientation
            if args.balance_panel_fingers and args.method=='panel_push':
                target[:,3:]=quat_mul(horizontal,torch.tensor([[math.cos(panel_pitch_adjust/2),math.sin(panel_pitch_adjust/2),0.,0.]],device=env.device))
            target[:,:3]=handle
            if args.method in ('panel_push','palm_push'):
                target[:,2]+=args.panel_height_offset
                target[:,:3]+=args.panel_lateral_offset*lateral
            if phase=='rest':target=home.clone()
            elif phase=='above':target[:,:3]+=.10*direction;target[:,2]=.46
            elif phase=='front':target[:,:3]+=.10*direction
            elif phase in ('touch','grasp'):
                target[:,:3]+=(args.palm_offset if args.method=='palm_push' else (-.014 if args.method=='handle_grasp' else -.008))*direction
            elif phase=='close':
                offset=(args.palm_offset-.020*min(wait/30.,1.)) if args.method=='palm_push' else -(.006+.025*min(wait/30.,1.))
                target[:,:3]+=offset*direction
            elif phase in ('release','retreat','hold'):target=retreat.clone()
            closed=args.method=='panel_push' or phase in ('grasp','close')
            if args.keep_gripper_open:closed=False
            command=advance_pose_toward_waypoint(command,target,(.06 if phase in ('touch','close') else .12)*.05,.05)
            action=absolute_pose_to_relative_action(ee,command,torch.tensor([-1. if closed else 1.],device=env.device),scale,max_translation_step_m=.05,max_rotation_step_rad=.08)
            frame=compositor.compose(camera.data.output['rgb'],camera.data.output['semantic_segmentation'],camera.data.intrinsic_matrices)[0].cpu().numpy()
            process.stdin.write(frame.tobytes())
            states.append(obs['critic'][0].cpu().numpy());actions.append(action[0].cpu().numpy());forces.append(force.cpu().numpy());points.append(sensor.contact_pos_w_history[0].cpu().numpy());phases.append(phase);openings.append(q)
            pitch_history.append(panel_pitch_adjust)
            world_points=sensor.contact_pos_w_history[0].reshape(-1,3)
            local_points.append(quat_apply_inverse(cabinet.data.body_quat_w[0,drawer_id].expand(len(world_points),-1),
                world_points-cabinet.data.body_pos_w[0,drawer_id]).reshape(5,len(contact_bodies),3).cpu().numpy())
            if step%60==0:
                Image.fromarray(frame).save(args.output_dir/f'frame_{step:04d}.png')
                print(json.dumps(dict(step=step,phase=phase,opening=q,force=f.tolist(),ee=ee[0,:3].tolist(),target=target[0,:3].tolist(),fingers=task['finger_positions'][0].tolist())),flush=True)
            obs,_,term,trunc,_=env.step(action);wait+=1
            if bool(term[0]|trunc[0]):raise RuntimeError('Unexpected termination')
            after=drawer_subtask_state(env,'UpperDrawerJoint');error=float((after['ee_pose'][0,:3]-target[0,:3]).norm())
            next_phase=None
            if phase=='rest' and wait>=15:next_phase='above'
            elif phase=='above' and error<.01:next_phase='front'
            elif phase=='front' and error<.008:next_phase='touch'
            elif phase=='touch' and (error<.003 or (args.method in ('panel_push','palm_push') and float(f.max())>.5)):next_phase='grasp' if args.method=='handle_grasp' else 'close'
            elif phase=='grasp':
                fingers=after['finger_positions'][0]
                valid_gap=float(fingers.min())>=.0015 and float(fingers.max())<=.012 and float(abs(fingers[0]-fingers[1]))<=.004
                grasp_verified=bool((f[:2]>.5).all()) and valid_gap
                if wait>=20:
                    if not grasp_verified:raise RuntimeError('No bilateral handle grasp')
                    next_phase='close'
            elif phase=='close' and float(after['joint_position'][0])<.008:
                retreat=after['ee_pose'].clone();command=retreat.clone();next_phase='release'
            elif phase=='release' and wait>=15:
                retreat[:,:3]+=.12*direction;retreat[:,2]+=.12;next_phase='retreat'
            elif phase=='retreat' and error<.01:next_phase='hold'
            elif phase=='hold' and wait>=20:
                success=max(openings[-10:])<.01 and float(f.max())<.5
                break
            if next_phase:
                print(json.dumps(dict(transition=phase+'->'+next_phase,step=step)),flush=True);phase=next_phase;wait=0
        process.stdin.close();code=process.wait();process=None
        if code:raise RuntimeError('Encoding failed')
        np.savez_compressed(args.output_dir/'episode.npz',oracle=states,action=actions,contact_forces=forces,contact_points_w=points,
            contact_points_drawer=local_points,phase=phases,drawer_opening=openings,panel_pitch_adjust_rad=pitch_history,contact_body_names=contact_bodies)
        bilateral_panel_fraction=None;bilateral_all_close=None;contacting_close_count=None
        if args.keep_gripper_open and args.method=='panel_push':
            normals=np.linalg.norm(np.asarray(forces),axis=-1)
            close=np.asarray(phases)=='close'
            contacting=close&(normals[:,:,:2].max(axis=(1,2))>.1)
            contacting_close_count=int(contacting.sum())
            bilateral_all_close=float((normals[close,:,:2].max(axis=1)>.1).all(axis=1).mean()) if close.any() else 0.
            bilateral_panel_fraction=float((normals[contacting,:,:2].max(axis=1)>.1).all(axis=1).mean()) if contacting.any() else 0.
            success=bool(success and np.asarray(states)[:,7:9].min()>.0375 and bilateral_panel_fraction>.9
                and contacting_close_count>=20 and normals[:,:,2].max()<.1 and np.abs(np.asarray(states)[-10:,-1]).max()<.02)
        palm_metrics={}
        if args.method=='palm_push':
            normals=np.linalg.norm(np.asarray(forces),axis=-1)
            close=np.asarray(phases)=='close'
            palm_active=close&(normals[:,:,2].max(1)>.1)
            others=[i for i in range(len(contact_bodies)) if i!=2]
            coverage=float(palm_active.sum()/max(close.sum(),1))
            palm_metrics=dict(palm_contact_close_frame_fraction=coverage,palm_contact_close_frames=int(palm_active.sum()),
                non_palm_peak_force_N=float(normals[:,:,others].max()),palm_peak_force_N=float(normals[:,:,2].max()),
                palm_contact_opening_range_m=[float(np.asarray(openings)[palm_active].min()),float(np.asarray(openings)[palm_active].max())] if palm_active.any() else None)
            success=bool(success and coverage>.7 and palm_active.sum()>=20 and normals[:,:,others].max()<.1
                and (not args.keep_gripper_open or np.asarray(states)[:,7:9].min()>.0375)
                and np.abs(np.asarray(states)[-10:,-1]).max()<.02)
        report=dict(success=success,method=args.method,frames=len(states),fps=20,resolution=[1280,720],initial_state=initial.tolist(),initial_opening_m=openings[0],final_opening_m=openings[-1],bilateral_handle_grasp_verified=grasp_verified,
            contact_body_names=contact_bodies,palm_offset_m=args.palm_offset,**palm_metrics,
            gripper_command='always_open' if args.keep_gripper_open else 'original_policy',
            panel_lateral_offset_m=args.panel_lateral_offset,panel_height_offset_m=args.panel_height_offset,
            contact_feedback_pitch_balance=args.balance_panel_fingers,
            initial_panel_pitch_deg=args.panel_pitch_deg,contact_balance_gain=args.balance_gain,
            fingertip_compliant_stiffness_N_per_m=args.fingertip_stiffness,
            fingertip_compliant_damping_N_s_per_m=args.fingertip_damping if args.fingertip_stiffness>0 else 0.,
            bilateral_panel_contact_frame_fraction=bilateral_panel_fraction,
            bilateral_fraction_denominator='close frames with at least one fingertip contact above 0.1 N',
            bilateral_all_close_frame_fraction=bilateral_all_close,contacting_close_frames=contacting_close_count,
            physics_contract='passive drawer; robot contact only; no state writes or retention forces after reset')
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        if not success:raise RuntimeError('Closure or contact-quality verification failed; see report.json')
    finally:
        if process:process.stdin.close();process.wait()
        env.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc();raise
    finally:app.close()
