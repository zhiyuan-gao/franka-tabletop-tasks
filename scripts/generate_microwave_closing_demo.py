"""Physically close the calibrated microwave at one of two contact heights."""

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--contact', choices=['upper','lower'], required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--max-steps', type=int, default=1000)
parser.add_argument('--contact-height', type=float, default=None)
parser.add_argument('--along', type=float, default=.265)
parser.add_argument('--push-lead', type=float, default=.05)
parser.add_argument('--effector', choices=['finger','palm'], default='finger')
parser.add_argument('--palm-offset', type=float, default=.065)
parser.add_argument('--touch-speed', type=float, default=.035)
parser.add_argument('--hinge-damping', type=float, default=.10)
parser.add_argument('--keep-gripper-open', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_dir.exists():
    raise FileExistsError(args.output_dir)
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def main():
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply_inverse
    from build_microwave_physics_asset import build_asset
    from track_b_envs.envs.microwave_closing.microwave_env_cfg import (
        MicrowaveClosingEnvCfg, ASSET, ROOT_POS, ROOT_YAW, HINGE_LOCAL, INITIAL_ANGLE,
    )
    from track_b_envs.envs.common.constants import FRANKA_HOME_JOINTS_RAD
    from track_b_envs.envs.common.compositing import RealBackgroundCompositor
    from track_b_envs.collection.experts.common import absolute_pose_to_relative_action, advance_pose_toward_waypoint

    build_asset(ASSET)
    args.output_dir.mkdir(parents=True)
    height = args.contact_height if args.contact_height is not None else (.090 if args.effector=='palm' else (.155 if args.contact=='upper' else .110))
    cfg = MicrowaveClosingEnvCfg()
    cfg.scene.microwave.actuators['hinge'].damping=args.hinge_damping
    contact_bodies=['panda_leftfinger','panda_rightfinger','panda_hand']
    if args.effector=='palm':
        # The original preview-derived USD has no DriveAPI. Merely setting an
        # actuator damping value cannot create the missing USD drive schema.
        # Author a zero-stiffness, zero-velocity passive brake at spawn time;
        # do not alter the shared asset or apply any closing position target.
        original_spawn=cfg.scene.microwave.spawn.func
        def spawn_with_passive_brake(prim_path, spawn_cfg, *positional, **kwargs):
            from pxr import UsdPhysics
            prim=original_spawn(prim_path,spawn_cfg,*positional,**kwargs)
            joint=prim.GetStage().GetPrimAtPath(str(prim.GetPath())+'/DoorHingeJoint')
            drive=UsdPhysics.DriveAPI.Apply(joint,'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(0.)
            drive.CreateDampingAttr(args.hinge_damping)
            drive.CreateTargetVelocityAttr(0.)
            drive.CreateMaxForceAttr(5.)
            return prim
        cfg.scene.microwave.spawn.func=spawn_with_passive_brake
        contact_bodies += ['panda_link7','panda_link6','panda_link5','panda_link4',
            'panda_link3','panda_link2','panda_link1','panda_link0']
        cfg.scene.door_contact.filter_prim_paths_expr=[
            '{ENV_REGEX_NS}/Robot/'+name for name in contact_bodies]
    cfg.sim.device = args.device
    cfg.seed = 20260827
    env = ManagerBasedRLEnv(cfg=cfg)
    process = None
    try:
        env.reset(seed=cfg.seed)
        for _ in range(10):
            env.step(torch.zeros(env.action_space.shape, device=env.device))
        obs, _ = env.reset(seed=cfg.seed)
        robot = env.scene['robot']
        door = env.scene['microwave']
        body_ids, _ = door.find_bodies('Door')
        door_id = body_ids[0]
        print('Door physics:',dict(masses=door.root_physx_view.get_masses().tolist(),
            inertias=door.root_physx_view.get_inertias().tolist(),
            damping=door.data.joint_damping.tolist()),flush=True)
        assert torch.allclose(robot.data.joint_pos[0,:7],torch.tensor(FRANKA_HOME_JOINTS_RAD,device=env.device),atol=1e-5,rtol=0)
        camera = env.scene['policy_camera']
        sensor = env.scene['door_contact']
        compositor = RealBackgroundCompositor(env.device)
        env.sim.render()
        camera.update(0.0, force_recompute=True)
        scale = torch.tensor(cfg.actions.arm_action.scale, device=env.device)
        def ee_pose():
            data = env.scene['ee_frame'].data
            return torch.cat((data.target_pos_w[:,0],data.target_quat_w[:,0]),dim=-1)
        hinge = np.array(ROOT_POS)+np.array([
            math.cos(ROOT_YAW)*HINGE_LOCAL[0]-math.sin(ROOT_YAW)*HINGE_LOCAL[1],
            math.sin(ROOT_YAW)*HINGE_LOCAL[0]+math.cos(ROOT_YAW)*HINGE_LOCAL[1],0.])
        def waypoint(q, offset, z):
            yaw = ROOT_YAW+q
            along = np.array([-math.cos(yaw),-math.sin(yaw),0.])
            outward = np.array([-math.sin(yaw),math.cos(yaw),0.])
            point = hinge+args.along*along+offset*outward
            point[2] = z
            # Rz(yaw) Rx(pi/2): fingers close vertically; tool +Z points into door.
            quat = [math.cos(yaw/2)/math.sqrt(2),math.cos(yaw/2)/math.sqrt(2),
                math.sin(yaw/2)/math.sqrt(2),math.sin(yaw/2)/math.sqrt(2)]
            if args.effector=='palm':
                # Downward gripper: local X is the door's outward normal.
                # The palm's side, not the finger tips, should lead the push.
                angle=yaw+math.pi/2
                quat=[0.,math.cos(angle/2),math.sin(angle/2),0.]
            return torch.tensor([[*point,*quat]],device=env.device,dtype=torch.float32)
        video = args.output_dir/'demo.mp4'
        process = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pix_fmt','rgb24','-video_size','1280x720','-framerate','20',
            '-i','pipe:0','-an','-c:v','libx264','-preset','slow','-crf','16',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        phase = 'rest'
        phase_steps = 0
        command = ee_pose().clone()
        retreat = None
        rows, actions, forces, contacts, phase_history, finger_positions = [], [], [], [], [], []
        finger_ids,_=robot.find_joints('panda_finger_joint.*')
        success = False
        contact_height_samples = []
        for step in range(args.max_steps):
            ee = ee_pose()
            q = float(door.data.joint_pos[0,0])
            qd = float(door.data.joint_vel[0,0])
            force = sensor.data.force_matrix_w_history[0].reshape(-1,3)
            contact_points = sensor.contact_pos_w_history[0].reshape(-1,3)
            force_norm = float(force.norm(dim=-1).max())
            finite = torch.isfinite(contact_points).all(dim=-1) & (force.norm(dim=-1)>.05)
            local_contacts = torch.full_like(contact_points, float('nan'))
            if finite.any():
                local_contacts[finite] = quat_apply_inverse(
                    door.data.body_quat_w[0,door_id].expand(int(finite.sum()),-1),
                    contact_points[finite]-door.data.body_pos_w[0,door_id])
                if phase in ('touch','push'):
                    contact_height_samples.extend(local_contacts[finite,2].cpu().tolist())
            if phase == 'rest':
                target = command.clone()
            elif phase == 'above':
                target = waypoint(INITIAL_ANGLE,.12 if args.effector=='palm' else .065,.32)
            elif phase == 'outside':
                # The lower approach is closer to the door to stay within the
                # arm workspace while retaining clearance from the handle.
                target = waypoint(q,.12 if args.effector=='palm' else (.045 if height<.12 else .075),height)
            elif phase == 'touch':
                target = waypoint(q,args.palm_offset if args.effector=='palm' else .006,height)
            elif phase == 'push':
                target = waypoint(min(q+args.push_lead,0.),args.palm_offset if args.effector=='palm' else .006,height)
            else:
                target = retreat
            speed = .12 if phase in ('above','outside','retreat') else .035
            if phase=='touch':
                speed=args.touch_speed
            command = advance_pose_toward_waypoint(command,target,speed*.05,1.0*.05)
            action = absolute_pose_to_relative_action(ee,command,
                torch.tensor([1. if args.keep_gripper_open or (phase=='rest' and phase_steps<6) else -1.],device=env.device),
                scale,max_translation_step_m=.05,max_rotation_step_rad=.08)
            frame = compositor.compose(camera.data.output['rgb'],camera.data.output['semantic_segmentation'],camera.data.intrinsic_matrices)[0].cpu().numpy()
            process.stdin.write(frame.tobytes())
            if step%80==0:
                Image.fromarray(frame).save(args.output_dir/f'frame_{step:04d}.png')
                print(json.dumps(dict(step=step,phase=phase,door_angle_deg=math.degrees(q),
                    ee=ee[0,:3].tolist(),target=target[0,:3].tolist(),contact_force_N=force_norm,
                    robot_joints=robot.data.joint_pos[0,:7].tolist())),flush=True)
            rows.append(obs['critic'][0].cpu().numpy().copy())
            actions.append(action[0].cpu().numpy().copy())
            forces.append(force.cpu().numpy().copy())
            contacts.append(local_contacts.cpu().numpy().copy())
            phase_history.append(phase)
            finger_positions.append(robot.data.joint_pos[0,finger_ids].cpu().numpy().copy())
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0] | truncated[0]):
                raise RuntimeError('Unexpected environment termination')
            phase_steps += 1
            error = float(torch.linalg.vector_norm(ee_pose()[0,:3]-target[0,:3]))
            new_q = float(door.data.joint_pos[0,0])
            new_qd = float(door.data.joint_vel[0,0])
            next_phase = None
            if phase=='rest' and phase_steps>=16:
                if abs(new_q-INITIAL_ANGLE)>math.radians(1):
                    raise RuntimeError('Door moved before robot approach')
                next_phase='above'
            elif phase=='above' and error<.015:
                next_phase='outside'
            elif phase=='outside' and error<.008:
                next_phase='touch'
            elif phase=='touch' and (force_norm>.10 or error<.004):
                next_phase='push'
            elif phase=='push' and new_q>=-math.radians(1.0):
                retreat=ee_pose().clone()
                yaw=ROOT_YAW+new_q
                retreat[0,:3]+=torch.tensor([-.10*math.sin(yaw),.10*math.cos(yaw),.14],device=env.device)
                next_phase='retreat'
            elif phase=='retreat' and error<.010:
                next_phase='hold'
            elif phase=='hold' and phase_steps>=20:
                recent=np.asarray(rows[-10:])
                success=(abs(new_q)<math.radians(2) and abs(new_qd)<.10
                    and np.max(np.abs(recent[:,-2]))<math.radians(2)
                    and np.max(np.abs(recent[:,-1]))<.10 and force_norm<.5)
                break
            if next_phase:
                print(json.dumps(dict(transition=f'{phase}->{next_phase}',step=step,door_angle_deg=math.degrees(new_q))),flush=True)
                phase=next_phase
                phase_steps=0
        process.stdin.close()
        stderr=process.stderr.read().decode(errors='replace')
        code=process.wait()
        process=None
        if code:
            raise RuntimeError(stderr)
        np.savez_compressed(args.output_dir/'episode.npz',action=actions,oracle=rows,
            contact_forces=forces,contact_points_door=contacts,phase=phase_history,
            finger_joint_pos=finger_positions)
        contact_norm=np.linalg.norm(np.asarray(forces).reshape(-1,5,len(contact_bodies),3),axis=-1)
        pushing=np.isin(phase_history,['touch','push'])
        contact_impulse=contact_norm[pushing].sum(axis=(0,1))*.01
        palm_fraction=float(contact_impulse[2]/max(contact_impulse.sum(),1e-12))
        push_mask=np.asarray(phase_history)=='push'
        palm_frames=contact_norm[:,:,2].max(axis=1)>.05
        palm_coverage=float(palm_frames[push_mask].mean()) if push_mask.any() else 0.
        measured_angles=np.rad2deg(np.asarray(rows)[:,-2])[push_mask&palm_frames]
        palm_angle_range=([float(measured_angles.min()),float(measured_angles.max())]
            if measured_angles.size else None)
        palm_verified=(palm_fraction>.95 and palm_coverage>.5
            and palm_angle_range is not None and palm_angle_range[0]<-70 and palm_angle_range[1]>-10
            and float(contact_norm[:,:,:2].max())<.05)
        if args.effector=='palm':
            success=success and palm_verified
        aperture=np.asarray(finger_positions).sum(axis=1)
        if args.keep_gripper_open:
            success=success and bool(aperture.min()>.075)
        report=dict(contact=args.contact,effector=args.effector,planned_tcp_height_m=height,
            planned_tcp_door_xyz=[-args.along,args.palm_offset if args.effector=='palm' else .006,height],
            contact_body_order=contact_bodies,
            push_contact_impulse_Ns=contact_impulse.tolist(),palm_contact_impulse_fraction=palm_fraction,
            palm_contact_verified=bool(palm_verified),
            push_frames_with_palm_contact_fraction=palm_coverage,
            palm_contact_door_angle_range_deg=palm_angle_range,
            passive_hinge_damping=args.hinge_damping,
            passive_brake_drive_schema_authored=(args.effector=='palm'),
            gripper_command='always_open' if args.keep_gripper_open else 'closed_during_push',
            min_gripper_aperture_m=float(aperture.min()),
            contact_height_median_m=float(np.median(contact_height_samples)) if contact_height_samples else None,
            physical_contact_samples=len(contact_height_samples),success=bool(success),
            final_door_angle_deg=math.degrees(new_q),final_door_velocity_rad_s=new_qd,
            initial_door_angle_deg=-86.,resolution=[1280,720],fps=20,frames=len(rows),
            source_layout='render_close_microwave_initial_state.py',
            physics_contract='passive hinge; zero position stiffness; no state overwrites after reset; robot contacts close door',
            final_phase=phase,video=str(video))
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not success or len(contact_height_samples)<5:
            raise RuntimeError('Did not verify a physically contacted and settled door closure')
        if args.effector=='finger' and abs(report['contact_height_median_m']-height)>.035:
            raise RuntimeError('Actual contact height differs from the intended contact region')
    finally:
        if process is not None:
            if not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
            process.wait()
        env.close()


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
