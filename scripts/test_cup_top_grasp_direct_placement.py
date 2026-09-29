"""Test a direct descending transfer after a verified physical top grasp.

The scene and target are unchanged. This tests ONE transfer path, not universal
infeasibility of top grasping. Contact-triggered safety stops are labeled as such.
"""
import argparse
import json
from pathlib import Path
import subprocess
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--max-steps',type=int,default=700)
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
    from track_b_envs.envs.cup_cabinet import CupToDrawerEnvCfg
    from track_b_envs.envs.microwave_closing.microwave_env_cfg import Terminations
    from track_b_envs.collection.experts.cup_cabinet import CupToDrawerExpert
    from track_b_envs.collection.experts.common import cup_cabinet_task_state,absolute_pose_to_relative_action
    cfg=CupToDrawerEnvCfg();cfg.sim.device=args.device;cfg.terminations=Terminations();cfg.episode_length_s=60.
    cfg.observations.policy.rgb=None;cfg.scene.policy_camera=None
    cfg.scene.paper_camera=TiledCameraCfg(prim_path='{ENV_REGEX_NS}/PaperCamera',
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.,horizontal_aperture=36.,clipping_range=(.01,20.)),
        width=1280,height=720,data_types=['rgb'],update_latest_camera_pose=True)
    cfg.scene.platform.spawn.visible=True
    cfg.scene.platform.spawn.visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12,.14,.16),roughness=.92)
    cfg.scene.real_background_key_light.spawn.intensity=1150.;cfg.scene.light.spawn.intensity=350.
    cfg.scene.object.spawn.activate_contact_sensors=True;cfg.scene.cabinet.spawn.activate_contact_sensors=True
    cfg.scene.grasp_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/BluePouringCup',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/panda_leftfinger','{ENV_REGEX_NS}/Robot/panda_rightfinger'],
        history_length=5,max_contact_data_count_per_prim=256)
    cfg.scene.body_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Cabinet/CabinetBody',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/BluePouringCup','{ENV_REGEX_NS}/Robot/panda_leftfinger',
            '{ENV_REGEX_NS}/Robot/panda_rightfinger','{ENV_REGEX_NS}/Robot/panda_hand'],
        history_length=5,max_contact_data_count_per_prim=512)
    env=ManagerBasedRLEnv(cfg=cfg);process=None;args.output_dir.mkdir(parents=True)
    try:
        env.reset(seed=20260827)
        for _ in range(10):env.step(torch.zeros(env.action_space.shape,device=env.device))
        obs,_=env.reset(seed=20260827)
        camera=env.scene['paper_camera']
        camera.set_world_poses_from_view(torch.tensor([[1.12,.02,.48]],device=env.device),torch.tensor([[.54,-.25,.32]],device=env.device))
        env.sim.render();camera.update(0.,force_recompute=True)
        expert=CupToDrawerExpert(env);expert.reset(env)
        cup=env.scene['object'];initial=cup.data.root_pose_w[0].cpu().numpy().copy()
        process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output_dir/'demo.mp4')],stdin=subprocess.PIPE)
        states=[];actions=[];phases=[];grasp_forces=[];cabinet_forces=[]
        verified=False;transfer=False;safety_stop=False;hold_pose=None;hold_steps=0;collision_frames=0;first_contact=None;placed=False
        verified_frame=None;verified_lift=None;verified_forces=None
        for step in range(args.max_steps):
            task=cup_cabinet_task_state(env);ee=task['ee_pose']
            gf=env.scene['grasp_contact'].data.force_matrix_w_history[0].reshape(5,2,3).norm(dim=-1)
            cf=env.scene['body_contact'].data.force_matrix_w_history[0].reshape(5,4,3).norm(dim=-1)
            cup_height=float(cup.data.root_pos_w[0,2]);bilateral=bool((gf.max(dim=0).values>.1).all())
            phase=expert.STATE_NAMES[int(expert.state[0])]
            if not verified and cup_height-float(initial[2])>.075 and bilateral:
                verified=True;verified_frame=step;verified_lift=cup_height-float(initial[2]);verified_forces=gf.max(dim=0).values.tolist()
                Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output_dir/'verified_top_grasp.png')
            if verified and not transfer and int(expert.state[0])==expert.ABOVE_TARGET:
                # Only replace the transfer path: skip the high waypoint above
                # the drawer, keeping the same final cup target and grasp.
                expert.state[:]=expert.LOWER;expert.wait.zero_();transfer=True
                expert.command_pose.copy_(ee)
                print('Starting direct descending transfer',flush=True)
            if transfer and not safety_stop and float(cf.max())>.5:
                if first_contact is None:first_contact=step
                collision_frames+=1
                # Stop unsafe contact, not a fabricated IK/physical failure.
                if collision_frames>=5 and bilateral:
                    safety_stop=True;hold_pose=ee.clone()
                    Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output_dir/'placement_collision.png')
            elif not safety_stop:
                collision_frames=0
            if safety_stop:
                phase='collision_safety_hold';hold_steps+=1
                action=absolute_pose_to_relative_action(ee,hold_pose,-torch.ones(1,device=env.device),expert.controller_scale,max_translation_step_m=.05,max_rotation_step_rad=.08)
            else:
                action=expert.compute(env)
            frame=camera.data.output['rgb'][0,:,:,:3].cpu().numpy();process.stdin.write(frame.tobytes())
            states.append(obs['critic'][0].cpu().numpy().copy());actions.append(action[0].cpu().numpy().copy());phases.append(phase)
            grasp_forces.append(gf.cpu().numpy());cabinet_forces.append(cf.cpu().numpy())
            if step%60==0:
                Image.fromarray(frame).save(args.output_dir/f'frame_{step:04d}.png')
                print(json.dumps(dict(step=step,phase=phase,cup=cup.data.root_pos_w[0].tolist(),target=task['target_pose'][0,:3].tolist(),grasp_N=gf.max(dim=0).values.tolist(),body_contact_N=cf.max(dim=0).values.tolist())),flush=True)
            if safety_stop and hold_steps>=20:break
            obs,_,terminated,truncated,_=env.step(action)
            if bool(terminated[0]|truncated[0]):break
            if int(expert.state[0])==expert.SETTLE:
                error=float((cup.data.root_pos_w[0]-task['target_pose'][0,:3]).norm())
                placed=error<.035
                if placed:break
        process.stdin.close();code=process.wait();process=None
        if code:raise RuntimeError('Encoding failed')
        task=cup_cabinet_task_state(env)
        report=dict(top_grasp_and_lift_verified=verified,lift_verification_frame=verified_frame,lift_at_verification_m=verified_lift,
            finger_forces_at_lift_N=verified_forces,placement_collision_safety_stop=safety_stop,placed=placed,
            first_transfer_collision_frame=first_contact,final_cup_position=cup.data.root_pos_w[0].tolist(),target_position=task['target_pose'][0,:3].tolist(),
            final_target_error_m=float((cup.data.root_pos_w[0]-task['target_pose'][0,:3]).norm()),
            cabinet_contact_filter_order=['cup','leftfinger','rightfinger','palm'],
            interpretation=('One direct descending transfer candidate after a physical top grasp; stopped on cabinet collision. This does not prove that a top grasp cannot place the cup. The existing high-clearance transfer succeeds.' if safety_stop else 'No collision-stopped failure verified; inspect placed and grasp verification flags.'),
            scene_or_target_modified=False,physics_contract='No state overwrites, attachments, or auxiliary object forces after reset.',frames=len(states),fps=20,resolution=[1280,720])
        np.savez_compressed(args.output_dir/'episode.npz',oracle=states,action=actions,phase=phases,grasp_force=grasp_forces,cabinet_force=cabinet_forces)
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
