"""Physically test a manually interpreted human cup-rim grasp transfer.

This is a qualitative grasp/waypoint retargeting, not measured 3-D human motion.
Do not label a grasp-approach collision as a grasp-then-transfer collision.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--human-video',type=Path,default=Path('cup_cabinet_human_pause.mp4'))
parser.add_argument('--replay-source',type=Path,default=None,help='Replay recorded actions for camera-only rendering; do not generate a new path.')
parser.add_argument('--tilt-deg',type=float,default=0.)
parser.add_argument('--approach-mode',choices=['front_diagonal','side_lateral'],default='front_diagonal')
parser.add_argument('--side-clearance',type=float,default=.26)
parser.add_argument('--side-entry-height',type=float,default=.12)
parser.add_argument('--side-stage-height',type=float,default=.32)
parser.add_argument('--max-steps',type=int,default=650)
parser.add_argument('--camera-eye',type=float,nargs=3,default=[.95,.25,.32])
parser.add_argument('--camera-target',type=float,nargs=3,default=[.52,-.15,.22])
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.output_dir.exists():raise FileExistsError(args.output_dir)
if not args.human_video.is_file():raise FileNotFoundError(args.human_video)
args.enable_cameras=True
app=AppLauncher(args).app


def replay_recorded_camera(env,obs,camera):
    import shutil
    import numpy as np
    import torch
    from PIL import Image
    source=args.replay_source.resolve()
    data=np.load(source/'episode.npz');original=json.loads((source/'report.json').read_text())
    assert np.array_equal(obs['critic'][0].cpu().numpy(),data['oracle'][0])
    process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output_dir/'demo.mp4')],stdin=subprocess.PIPE)
    differences=[];force_differences=[];hit_frames=[]
    try:
        for index,action in enumerate(data['action']):
            actual=obs['critic'][0].cpu().numpy()
            differences.append(np.abs(actual-data['oracle'][index]))
            f=env.scene['blocking_contact'].data.force_matrix_w_history[0].cpu().numpy().reshape(data['drawer_contact_forces'][index].shape)
            force_differences.append(float(np.abs(f-data['drawer_contact_forces'][index]).max()))
            if np.linalg.norm(f,axis=-1).max()>1.:hit_frames.append(index)
            frame=camera.data.output['rgb'][0,:,:,:3].cpu().numpy();process.stdin.write(frame.tobytes())
            if index in (0,original['first_collision_frame'],original['safety_stop_frame']):
                Image.fromarray(frame).save(args.output_dir/f'frame_{index:04d}.png')
            if index%60==0:print(json.dumps(dict(replay_frame=index,max_state_error=float(differences[-1].max()))),flush=True)
            if index+1<len(data['action']):
                obs,_,terminated,truncated,_=env.step(torch.tensor(action[None],device=env.device))
                assert not bool(terminated[0]|truncated[0])
    finally:
        process.stdin.close();code=process.wait()
    assert code==0
    difference=np.asarray(differences)
    validation=dict(recorded_actions_replayed=True,source=str(source),frames=len(data['action']),
        max_oracle_absolute_error=float(difference.max()),max_joint_position_error_rad=float(difference[:,:9].max()),
        max_tcp_position_error_m=float(difference[:,18:21].max()),
        max_cup_position_error_m=float(difference[:,25:28].max()),
        max_contact_force_component_error_N=max(force_differences),
        first_collision_frame=hit_frames[0] if hit_frames else None,
        state_overwrite_used=False,camera_eye_m=args.camera_eye,camera_target_m=args.camera_target)
    assert validation['max_oracle_absolute_error']<.005
    assert max(validation['max_joint_position_error_rad'],validation['max_tcp_position_error_m'],validation['max_cup_position_error_m'])<.0001
    assert validation['max_contact_force_component_error_N']<.5
    assert validation['first_collision_frame']==original['first_collision_frame']
    for name in ('episode.npz','human_motion_mapping.json'):shutil.copy2(source/name,args.output_dir/name)
    report=dict(original,camera_eye_m=args.camera_eye,camera_target_m=args.camera_target,camera_only_replay=validation)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output_dir/'replay_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)


def main():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCameraCfg,ContactSensorCfg
    from isaaclab.utils.math import quat_mul,quat_apply,quat_apply_inverse
    from track_b_envs.envs.cup_cabinet import CupToCabinetTopEnvCfg
    from track_b_envs.envs.microwave_closing.microwave_env_cfg import Terminations,ContactPointHistorySensor
    from track_b_envs.envs.common.constants import FRANKA_HOME_JOINTS_RAD
    from track_b_envs.collection.experts.common import absolute_pose_to_relative_action,advance_pose_toward_waypoint,cup_cabinet_task_state

    cfg=CupToCabinetTopEnvCfg();cfg.sim.device=args.device;cfg.seed=20260827
    cfg.terminations=Terminations();cfg.rewards=None;cfg.episode_length_s=90.
    cfg.observations.policy.rgb=None;cfg.scene.policy_camera=None
    cfg.scene.paper_camera=TiledCameraCfg(prim_path='{ENV_REGEX_NS}/PaperCamera',
        spawn=sim_utils.PinholeCameraCfg(focal_length=21.,horizontal_aperture=36.,clipping_range=(.01,20.)),
        width=1280,height=720,data_types=['rgb'],update_latest_camera_pose=True)
    cfg.scene.platform.spawn.visible=True
    cfg.scene.platform.spawn.visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12,.14,.16),roughness=.92)
    cfg.scene.real_background_key_light.spawn.intensity=1150.;cfg.scene.light.spawn.intensity=350.
    cfg.scene.object.spawn.activate_contact_sensors=True;cfg.scene.cabinet.spawn.activate_contact_sensors=True
    cfg.scene.grasp_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/BluePouringCup',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/panda_leftfinger','{ENV_REGEX_NS}/Robot/panda_rightfinger','{ENV_REGEX_NS}/Platform'],
        history_length=5,max_contact_data_count_per_prim=512)
    bodies=['panda_leftfinger','panda_rightfinger','panda_hand']+['panda_link'+str(i) for i in range(7,-1,-1)]
    cfg.scene.blocking_contact=ContactSensorCfg(class_type=ContactPointHistorySensor,
        prim_path='{ENV_REGEX_NS}/Cabinet/UpperDrawer',track_contact_points=True,
        filter_prim_paths_expr=['{ENV_REGEX_NS}/BluePouringCup']+['{ENV_REGEX_NS}/Robot/'+b for b in bodies],
        history_length=5,max_contact_data_count_per_prim=1024)
    env=ManagerBasedRLEnv(cfg=cfg);process=None;args.output_dir.mkdir(parents=True)
    try:
        env.reset(seed=cfg.seed)
        for _ in range(10):env.step(torch.zeros(env.action_space.shape,device=env.device))
        obs,_=env.reset(seed=cfg.seed)
        robot=env.scene['robot'];cup=env.scene['object'];cabinet=env.scene['cabinet']
        sensor=env.scene['blocking_contact'];drawer_id=cabinet.find_bodies('UpperDrawer')[0][0]
        assert torch.allclose(robot.data.joint_pos[0,:7],torch.tensor(FRANKA_HOME_JOINTS_RAD,device=env.device),atol=1e-5)
        camera=env.scene['paper_camera']
        camera.set_world_poses_from_view(torch.tensor([args.camera_eye],device=env.device),torch.tensor([args.camera_target],device=env.device))
        env.sim.render();camera.update(0.,force_recompute=True)
        if args.replay_source is not None:
            replay_recorded_camera(env,obs,camera)
            return
        initial=cup.data.root_pose_w[0].clone();cabq=cabinet.data.root_quat_w
        normal=quat_apply(cabq,torch.tensor([[0.,1.,0.]],device=env.device))[0]
        lateral=quat_apply(cabq,torch.tensor([[1.,0.,0.]],device=env.device))[0]
        top=quat_mul(cabq,torch.tensor([[0.,math.sqrt(.5),math.sqrt(.5),0.]],device=env.device))
        if args.approach_mode=='side_lateral':
            # For lateral entry, place the open fingers on the cup's front/back
            # sides. This keeps the opening axis perpendicular to the approach,
            # just as the earlier front-entry mode did for its own direction.
            top=quat_mul(cabq,torch.tensor([[0.,1.,0.,0.]],device=env.device))
        if args.tilt_deg:
            angle=math.radians(args.tilt_deg)/2
            top=quat_mul(top,torch.tensor([[math.cos(angle),0.,math.sin(angle),0.]],device=env.device))
        def ee():return cup_cabinet_task_state(env)['ee_pose']
        def pose(front,side,z):
            p=initial[:3]+normal*front+lateral*side;p[2]=z
            return torch.cat((p[None],top),dim=-1)
        # Retain the earlier diagonal candidate for reproducibility. The user
        # corrected the human path: lateral entry from the drawer's SIDE, not
        # its front. Stage outside that side before moving parallel to local X.
        plan=[('outside',pose(.20,.10,.27)),
            ('rim_approach',pose(0.,0.,.062)),
            ('grasp',pose(0.,0.,.062)),
            ('extract',pose(.17,0.,.065)),
            ('lift',pose(.17,0.,.50))]
        if args.approach_mode=='side_lateral':
            plan=[('side_high',pose(.08,args.side_clearance,args.side_stage_height)),
                ('outside',pose(0.,args.side_clearance,args.side_entry_height)),
                ('side_entry',pose(0.,0.,args.side_entry_height)),
                ('rim_approach',pose(0.,0.,.062)),
                ('grasp',pose(0.,0.,.062)),
                ('extract',pose(0.,args.side_clearance,.065)),
                ('lift',pose(0.,args.side_clearance,.50))]
        destination=cup_cabinet_task_state(env)['target_pose'].clone();destination[:,3:]=top
        above=destination.clone();above[:,2]=.50
        place=destination.clone();place[:,2]+=.062
        plan += [('above_cabinet',above),('place',place),('release',place)]
        planned={name:value[0].tolist() for name,value in plan}
        (args.output_dir/'human_motion_mapping.json').write_text(json.dumps(dict(
            source_video=str(args.human_video.resolve()),source_sha256=hashlib.sha256(args.human_video.read_bytes()).hexdigest(),
            source_duration_s=203/30,
            observed_sequence=['reach laterally from the SIDE of the drawer to above the cup' if args.approach_mode=='side_lateral' else 'legacy front-diagonal approximation (direction rejected by user)',
                'grasp cup from above; annotated pause in video',
                'move cup clear of drawer then lift','place upright on cabinet top'],
            mapping='Manually authored qualitative grasp-direction and waypoint transfer; not calibrated 3-D pose tracking or frame-exact imitation.',
            planned_tool_tilt_from_vertical_deg=abs(args.tilt_deg),
            approach_mode=args.approach_mode,
            cabinet_lateral_axis_w=lateral.tolist(),cabinet_front_axis_w=normal.tolist(),
            cabinet_root_pose_wxyz=cabinet.data.root_pose_w[0].tolist(),
            side_clearance_m=args.side_clearance,side_entry_tcp_height_m=args.side_entry_height,
            planned_tcp_poses_wxyz=planned,scene_modified=False),indent=2)+'\n')
        target=ee().clone();command=target.clone();scale=torch.tensor(cfg.actions.arm_action.scale,device=env.device)
        phase='rest';age=0;plan_index=-1;contact_count=0;first_collision=None;stop_frame=None;stop_phase=None
        failed=False;grasped=False;finished=False;stop_target=None
        rows=[];actions=[];phases=[];grasp_forces=[];drawer_forces=[];ee_poses=[];commands=[];points=[];tilts=[]
        process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output_dir/'demo.mp4')],stdin=subprocess.PIPE)
        for step in range(args.max_steps):
            current=ee()
            gf=env.scene['grasp_contact'].data.force_matrix_w_history[0].reshape(5,3,3)
            df=sensor.data.force_matrix_w_history[0].reshape(5,len(bodies)+1,3)
            magnitude=df.norm(dim=-1);drawer_force=float(magnitude.max())
            if phase not in ('rest','safety_hold'):
                contact_count=contact_count+1 if drawer_force>1. else 0
                if drawer_force>1. and first_collision is None:
                    first_collision=step
                    Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output_dir/'first_collision.png')
                if contact_count>=8:
                    failed=True;stop_phase=phase;stop_frame=step;stop_target=target.clone()
                    phase='safety_hold';target=current.clone();command=current.clone();age=0
                    Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output_dir/'collision_stop.png')
            speed=.08 if phase in ('side_high','outside') else .045
            command=advance_pose_toward_waypoint(command,target,speed*.05,.05)
            closed=phase in ('grasp','extract','lift','above_cabinet','place') or (phase=='safety_hold' and grasped)
            action=absolute_pose_to_relative_action(current,command,torch.tensor([-1. if closed else 1.],device=env.device),scale,max_translation_step_m=.05,max_rotation_step_rad=.08)
            frame=camera.data.output['rgb'][0,:,:,:3].cpu().numpy();process.stdin.write(frame.tobytes())
            rows.append(obs['critic'][0].cpu().numpy().copy());actions.append(action[0].cpu().numpy().copy());phases.append(phase)
            grasp_forces.append(gf.cpu().numpy());drawer_forces.append(df.cpu().numpy());ee_poses.append(current[0].cpu().numpy());commands.append(target[0].cpu().numpy())
            local=sensor.contact_pos_w_history[0].reshape(-1,3)
            local=quat_apply_inverse(cabinet.data.body_quat_w[0,drawer_id].expand(len(local),-1),local-cabinet.data.body_pos_w[0,drawer_id])
            points.append(local.reshape(5,len(bodies)+1,3).cpu().numpy())
            tool_z=quat_apply(current[:,3:],torch.tensor([[0.,0.,1.]],device=env.device))[0]
            tilts.append(float(torch.rad2deg(torch.acos((-tool_z[2]).clamp(-1,1)))))
            if step%60==0:
                Image.fromarray(frame).save(args.output_dir/f'frame_{step:04d}.png')
                print(json.dumps(dict(step=step,phase=phase,ee=current[0,:3].tolist(),target=target[0,:3].tolist(),tool_tilt_deg=tilts[-1],drawer_forces=magnitude.max(0).values.tolist())),flush=True)
            if phase=='safety_hold' and age>=30:break
            obs,_,terminated,truncated,_=env.step(action);age+=1
            if bool(terminated[0]|truncated[0]):raise RuntimeError('Unexpected simulation termination')
            error=float((ee()[0,:3]-target[0,:3]).norm())
            angular=float(2*torch.acos((ee()[0,3:]*target[0,3:]).sum().abs().clamp(0,1)))
            advance=False
            if phase=='rest':advance=age>=15
            elif phase=='grasp':
                if age>=20:
                    grasped=bool((gf.norm(dim=-1).max(0).values[:2]>1.).all())
                    if not grasped:break
                    advance=True
            elif phase=='release':
                if age>=30:finished=True;break
            elif phase!='safety_hold':advance=error<.006 and angular<.06
            if advance:
                plan_index+=1;next_phase,target=plan[plan_index]
                print(json.dumps(dict(transition=phase+'->'+next_phase,step=step)),flush=True)
                phase=next_phase;age=0
        process.stdin.close();code=process.wait();process=None
        if code:raise RuntimeError('Video encoding failed')
        force=np.linalg.norm(np.asarray(drawer_forces),axis=-1)
        collision_mask=force.max((1,2))>1.
        names=['cup']+bodies;peaks=force.max((0,1))
        actual_tilt=float(np.median(np.asarray(tilts)[collision_mask])) if collision_mask.any() else None
        cup_delta=float((cup.data.root_pos_w[0]-initial[:3]).norm())
        report=dict(failure_verified=failed,task='below_drawer_cup_to_cabinet_top',failure_stage=stop_phase,
            approach_mode=args.approach_mode,
            physical_grasp_verified=grasped,task_completed=finished,first_collision_frame=first_collision,
            safety_stop_frame=stop_frame,consecutive_contact_frames_trigger=8,contact_threshold_N=1.,
            blocking_body_names=[name for name,peak in zip(names,peaks) if peak>1.],
            peak_drawer_forces_N=dict(zip(names,peaks.tolist())),
            desired_tool_tilt_from_vertical_deg=abs(args.tilt_deg),actual_tool_tilt_during_collision_deg=actual_tilt,
            desired_pose_at_collision_stop=None if stop_target is None else stop_target[0].tolist(),
            actual_pose_at_collision_stop=None if stop_frame is None else ee_poses[stop_frame].tolist(),
            cup_displacement_m=cup_delta,initial_cup_pose=initial.tolist(),final_cup_pose=cup.data.root_pose_w[0].tolist(),
            frames=len(rows),fps=20,resolution=[1280,720],scene_or_target_modified=False,
            camera_eye_m=args.camera_eye,camera_target_m=args.camera_target,
            source_human_video=str(args.human_video.resolve()),
            interpretation='A manually interpreted human-inspired top-grasp motion candidate tested in the existing robot scene. A collision before grasp is an approach failure, not a post-grasp transfer failure. This is not an exact 3-D human trajectory replay or a universal impossibility proof.',
            physics_contract='No object attachment, object/robot state overwrites, extra obstacles, or auxiliary forces after reset.')
        np.savez_compressed(args.output_dir/'episode.npz',oracle=rows,action=actions,phase=phases,
            grasp_contact_forces=grasp_forces,drawer_contact_forces=drawer_forces,
            contact_points_drawer=points,contact_body_names=names,ee_pose=ee_poses,target_pose=commands,tool_tilt_deg=tilts)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
        if not failed:raise RuntimeError('No verified physical collision failure; do not publish as failure.')
    finally:
        if process:process.stdin.close();process.wait()
        env.close()


if __name__=='__main__':
    try:main()
    finally:app.close()
