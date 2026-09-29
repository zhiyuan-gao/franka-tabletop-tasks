"""Test a top-oriented grasp below the drawer, then a direct upward transfer.

An approach collision is NOT labeled a successful-grasp/transfer failure.
All object and robot motion after reset is driven by physical simulation.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--pregrasp-height',type=float,default=.050)
parser.add_argument('--grasp-height',type=float,default=.040)
parser.add_argument('--tilt-deg',type=float,default=0.)
parser.add_argument('--max-steps',type=int,default=900)
parser.add_argument('--camera-eye',type=float,nargs=3,default=[.60,.62,.58])
parser.add_argument('--camera-target',type=float,nargs=3,default=[.52,-.15,.25])
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.output_dir.exists():raise FileExistsError(args.output_dir)
args.enable_cameras=True
app=AppLauncher(args).app


def main():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCameraCfg,ContactSensorCfg
    from isaaclab.utils.math import quat_mul,quat_apply
    from track_b_envs.envs.cup_cabinet import CupToCabinetTopEnvCfg
    from track_b_envs.envs.microwave_closing.microwave_env_cfg import Terminations
    from track_b_envs.collection.experts.common import absolute_pose_to_relative_action,advance_pose_toward_waypoint,cup_cabinet_task_state
    cfg=CupToCabinetTopEnvCfg();cfg.sim.device=args.device;cfg.terminations=Terminations();cfg.episode_length_s=65.
    cfg.observations.policy.rgb=None;cfg.scene.policy_camera=None
    cfg.scene.paper_camera=TiledCameraCfg(prim_path='{ENV_REGEX_NS}/PaperCamera',
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.,horizontal_aperture=36.,clipping_range=(.01,20.)),
        width=1280,height=720,data_types=['rgb'],update_latest_camera_pose=True)
    cfg.scene.platform.spawn.visible=True
    cfg.scene.platform.spawn.visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12,.14,.16),roughness=.92)
    cfg.scene.real_background_key_light.spawn.intensity=1150.;cfg.scene.light.spawn.intensity=350.
    cfg.scene.object.spawn.activate_contact_sensors=True;cfg.scene.cabinet.spawn.activate_contact_sensors=True
    cfg.scene.grasp_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/BluePouringCup',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/panda_leftfinger','{ENV_REGEX_NS}/Robot/panda_rightfinger','{ENV_REGEX_NS}/Platform'],
        history_length=5,max_contact_data_count_per_prim=512)
    bodies=['panda_leftfinger','panda_rightfinger','panda_hand','panda_link7','panda_link6','panda_link5','panda_link4','panda_link3','panda_link2','panda_link1','panda_link0']
    cfg.scene.blocking_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Cabinet/UpperDrawer',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/BluePouringCup']+['{ENV_REGEX_NS}/Robot/'+b for b in bodies],
        history_length=5,max_contact_data_count_per_prim=1024)
    env=ManagerBasedRLEnv(cfg=cfg);process=None;args.output_dir.mkdir(parents=True)
    try:
        env.reset(seed=20260827)
        for _ in range(10):env.step(torch.zeros(env.action_space.shape,device=env.device))
        obs,_=env.reset(seed=20260827)
        camera=env.scene['paper_camera']
        camera.set_world_poses_from_view(torch.tensor([args.camera_eye],device=env.device),torch.tensor([args.camera_target],device=env.device))
        env.sim.render();camera.update(0.,force_recompute=True)
        cup=env.scene['object'];initial=cup.data.root_pose_w[0].clone()
        cabinet=env.scene['cabinet'];cabq=cabinet.data.root_quat_w[0]
        normal=quat_apply(cabq[None],torch.tensor([[0.,1.,0.]],device=env.device))[0]
        # Finger opening axis is cabinet-local X, allowing entry from the front
        # around the cup without sweeping a finger through its center.
        top=quat_mul(cabq[None],torch.tensor([[0.,math.sqrt(.5),math.sqrt(.5),0.]],device=env.device))
        # Optional top-approach tilt: tool remains mostly downward; NEGATIVE
        # tilt moves the wrist outward. Report it rather than calling it vertical.
        if args.tilt_deg:
            t=math.radians(args.tilt_deg)/2
            top=quat_mul(top,torch.tensor([[math.cos(t),0.,math.sin(t),0.]],device=env.device))
        def ee():return cup_cabinet_task_state(env)['ee_pose']
        def pose(distance,height):
            p=initial[:3]+normal*distance;p[2]=height
            return torch.cat((p[None],top),dim=-1)
        command=ee().clone();target=command.clone();scale=torch.tensor(cfg.actions.arm_action.scale,device=env.device)
        phase='rest';age=0;contact_count=0;verified=False;grasp_verified_frame=None;blocked=False;approach_blocked=False;hold=0
        rows=[];actions=[];phases=[];grasp_forces=[];drawer_forces=[];ee_poses=[];commands=[]
        lift_start=None;collision_frame=None;peak_before_lift=0.;first_lift_free=None
        process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output_dir/'demo.mp4')],stdin=subprocess.PIPE)
        for step in range(args.max_steps):
            current=ee()
            gf=env.scene['grasp_contact'].data.force_matrix_w_history[0].reshape(5,3,3).norm(dim=-1)
            df=env.scene['blocking_contact'].data.force_matrix_w_history[0].reshape(5,len(bodies)+1,3).norm(dim=-1)
            bilateral=bool((gf[:,:2].max(0).values>1.).all())
            lift=float(cup.data.root_pos_w[0,2]-initial[2]);drawer_force=float(df.max())
            up=quat_apply(cup.data.root_quat_w,torch.tensor([[0.,0.,1.]],device=env.device))[0]
            # Conservative lowest point of the tapered cup, including tilt.
            sine=float(up[:2].norm());cosine=float(up[2])
            lowest=float(cup.data.root_pos_w[0,2])+min(-.026*sine,.087*cosine-.039*sine)
            if phase not in ('lift','collision_hold'):peak_before_lift=max(peak_before_lift,drawer_force)
            if phase=='lift' and bilateral and lift>.010 and lowest>.005 and float(gf[:,2].max())<.05 and drawer_force<.5:
                if not verified:
                    verified=True;grasp_verified_frame=step;first_lift_free=lift
                    Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output_dir/'verified_top_grasp.png')
            if phase=='lift' and drawer_force>.5:
                contact_count+=1
                if collision_frame is None:collision_frame=step
                if verified and bilateral and contact_count>=10:
                    blocked=True;phase='collision_hold';target=current.clone();command=current.clone();age=0
                    Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output_dir/'lift_collision.png')
            elif phase=='lift':contact_count=0
            elif phase in ('enter','descend'):
                contact_count=contact_count+1 if drawer_force>2. else 0
                if contact_count>=20:
                    approach_blocked=True;phase='approach_blocked';target=current.clone();command=current.clone();age=0
            speed=.10 if phase in ('high_front','low_front') else (.025 if phase in ('enter','descend','lift') else .01)
            command=advance_pose_toward_waypoint(command,target,speed*.05,.05)
            closed=phase in ('grasp','lift','collision_hold')
            action=absolute_pose_to_relative_action(current,command,torch.tensor([-1. if closed else 1.],device=env.device),scale,max_translation_step_m=.05,max_rotation_step_rad=.08)
            frame=camera.data.output['rgb'][0,:,:,:3].cpu().numpy();process.stdin.write(frame.tobytes())
            rows.append(obs['critic'][0].cpu().numpy().copy());actions.append(action[0].cpu().numpy().copy());phases.append(phase)
            grasp_forces.append(gf.cpu().numpy());drawer_forces.append(df.cpu().numpy());ee_poses.append(current[0].cpu().numpy());commands.append(target[0].cpu().numpy())
            if step%60==0:
                Image.fromarray(frame).save(args.output_dir/f'frame_{step:04d}.png')
                print(json.dumps(dict(step=step,phase=phase,ee=current[0,:3].tolist(),target=target[0,:3].tolist(),cup=cup.data.root_pos_w[0].tolist(),grasp_N=gf.max(0).values.tolist(),drawer_N=df.max(0).values.tolist())),flush=True)
            if phase in ('collision_hold','approach_blocked') and age>=20:break
            if phase in ('enter','descend') and age>=220:
                print('Approach waypoint timed out without verified grasp',flush=True)
                break
            obs,_,terminated,truncated,_=env.step(action);age+=1
            if bool(terminated[0]|truncated[0]):break
            error=float((ee()[0,:3]-target[0,:3]).norm())
            angular=2*torch.acos((ee()[0,3:]*target[0,3:]).sum().abs().clamp(0,1))
            next_phase=None
            if phase=='rest' and age>=10:next_phase='high_front';target=pose(.22,.40)
            elif phase=='high_front' and error<.015 and angular<.08:next_phase='low_front';target=pose(.16,args.pregrasp_height)
            elif phase=='low_front' and error<.010 and angular<.06:next_phase='enter';target=pose(0.,args.pregrasp_height)
            elif phase=='enter' and error<.006 and angular<.06:next_phase='descend';target=pose(0.,args.grasp_height)
            elif phase=='descend' and error<.008 and angular<.08:next_phase='grasp';target=ee().clone();command=target.clone()
            elif phase=='grasp' and age>=24:
                if not bilateral:break
                next_phase='lift';lift_start=step;target=ee().clone();target[0,2]=.50
            elif phase=='lift' and error<.01:break
            if next_phase:
                print(json.dumps(dict(transition=f'{phase}->{next_phase}',step=step)),flush=True)
                phase=next_phase;age=0;contact_count=0
        process.stdin.close();code=process.wait();process=None
        if code:raise RuntimeError('Encoding failed')
        report=dict(task='cup_below_open_drawer_to_cabinet_top',top_grasp_and_free_lift_verified=verified,
            correct_failure_verified=bool(verified and blocked and peak_before_lift<.5),
            approach_blocked=approach_blocked,grasp_verification_frame=grasp_verified_frame,
            free_lift_at_verification_m=first_lift_free,lift_start_frame=lift_start,
            first_lift_collision_frame=collision_frame,collision_safety_stop=blocked,
            peak_drawer_force_before_lift_N=peak_before_lift,drawer_filter_order=['cup']+bodies,
            top_grasp_tilt_deg=args.tilt_deg,pregrasp_tcp_height_m=args.pregrasp_height,grasp_tcp_height_m=args.grasp_height,
            initial_cup_pose=initial.tolist(),final_cup_pose=cup.data.root_pose_w[0].tolist(),
            target_cabinet_top_pose=cup_cabinet_task_state(env)['target_pose'][0].tolist(),
            final_phase=phase,frames=len(rows),fps=20,resolution=[1280,720],scene_or_target_modified=False,
            camera_eye_m=args.camera_eye,camera_target_m=args.camera_target,
            interpretation='Tests a top-oriented grasp reached from the front below the drawer, followed by a direct upward transfer. A lift collision only counts after free physical lift and bilateral grasp verification. Not a proof that all top-grasp transfer paths fail.',
            physics_contract='No state writes, object attachments or auxiliary forces after reset.')
        np.savez_compressed(args.output_dir/'episode.npz',oracle=rows,action=actions,phase=phases,grasp_force=grasp_forces,drawer_force=drawer_forces,ee_pose=ee_poses,target_pose=commands)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    finally:
        if process:process.stdin.close();process.wait()
        env.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc();raise
    finally:app.close()
