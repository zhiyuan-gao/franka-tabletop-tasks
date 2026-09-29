"""Physically test one vertical grasp candidate below the open upper drawer.

This is a controlled candidate test, not evidence that every top grasp fails.
"""
import argparse
import json
from pathlib import Path
import subprocess
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
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
    from track_b_envs.envs.cup_cabinet import CupToCabinetTopEnvCfg
    from track_b_envs.envs.microwave_closing.microwave_env_cfg import Terminations
    from track_b_envs.collection.experts.common import absolute_pose_to_relative_action,advance_pose_toward_waypoint
    cfg=CupToCabinetTopEnvCfg();cfg.sim.device=args.device
    cfg.terminations=Terminations();cfg.episode_length_s=60.
    cfg.observations.policy.rgb=None;cfg.scene.policy_camera=None
    cfg.scene.paper_camera=TiledCameraCfg(prim_path='{ENV_REGEX_NS}/PaperCamera',
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.,horizontal_aperture=36.,clipping_range=(.01,20.)),
        width=1280,height=720,data_types=['rgb'],update_latest_camera_pose=True)
    cfg.scene.platform.spawn.visible=True
    cfg.scene.platform.spawn.visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12,.14,.16),roughness=.92)
    cfg.scene.real_background_key_light.spawn.intensity=1150.
    cfg.scene.light.spawn.intensity=350.
    cfg.scene.cabinet.spawn.activate_contact_sensors=True
    cfg.scene.blocking_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Cabinet/UpperDrawer',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/panda_leftfinger','{ENV_REGEX_NS}/Robot/panda_rightfinger','{ENV_REGEX_NS}/Robot/panda_hand'],history_length=5,max_contact_data_count_per_prim=256)
    print('Creating candidate-test environment',flush=True)
    env=ManagerBasedRLEnv(cfg=cfg);process=None;args.output_dir.mkdir(parents=True)
    try:
        env.reset(seed=20260827)
        for _ in range(10):env.step(torch.zeros(env.action_space.shape,device=env.device))
        env.reset(seed=20260827)
        camera=env.scene['paper_camera']
        camera.set_world_poses_from_view(torch.tensor([[.60,.62,.58]],device=env.device),torch.tensor([[.52,-.15,.25]],device=env.device))
        env.sim.render();camera.update(0.,force_recompute=True)
        def ee():
            d=env.scene['ee_frame'].data
            return torch.cat((d.target_pos_w[:,0],d.target_quat_w[:,0]),-1)
        cup=env.scene['object'];initial=cup.data.root_pose_w[0].cpu().numpy().copy()
        target=ee().clone();target[:,:3]=cup.data.root_pos_w;target[:,2]=.50
        target[:,3:]=torch.tensor([0.,1.,0.,0.],device=env.device)
        command=ee().clone();scale=torch.tensor(cfg.actions.arm_action.scale,device=env.device)
        process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output_dir/'blocked_top_grasp.mp4')],stdin=subprocess.PIPE)
        phase='above';wait=0;contact_steps=0;peak_force=0.;blocked=False
        rows=[];actions=[];forces=[];phases=[]
        for step in range(500):
            force=float(env.scene['blocking_contact'].data.force_matrix_w_history.norm(dim=-1).max())
            peak_force=max(peak_force,force)
            command=advance_pose_toward_waypoint(command,target,(.10 if phase=='above' else .025)*.05,.05)
            action=absolute_pose_to_relative_action(ee(),command,torch.ones(1,device=env.device),scale,max_translation_step_m=.05,max_rotation_step_rad=.08)
            frame=camera.data.output['rgb'][0,:,:,:3].cpu().numpy()
            process.stdin.write(frame.tobytes())
            rows.append(np.r_[ee()[0].cpu().numpy(),cup.data.root_state_w[0].cpu().numpy()]);actions.append(action[0].cpu().numpy());forces.append(force);phases.append(phase)
            if step%60==0:
                Image.fromarray(frame).save(args.output_dir/f'frame_{step:04d}.png')
                print(json.dumps(dict(step=step,phase=phase,ee=ee()[0,:3].tolist(),target=target[0,:3].tolist(),drawer_contact_N=force)),flush=True)
            if phase=='descend' and force>.5:
                contact_steps+=1
                if contact_steps==1:Image.fromarray(frame).save(args.output_dir/'first_contact.png')
                if contact_steps>=20 and float(ee()[0,2])>.15:
                    blocked=True
                    Image.fromarray(frame).save(args.output_dir/'blocked_candidate.png')
                    break
            env.step(action);wait+=1
            if phase=='above' and float((ee()[0,:3]-target[0,:3]).norm())<.01:
                phase='descend';target[:,2]=initial[2]+.045;wait=0
        process.stdin.close();code=process.wait();process=None
        if code:raise RuntimeError('Encoding failed')
        cup_delta=float(np.linalg.norm(cup.data.root_pos_w[0].cpu().numpy()-initial[:3]))
        report=dict(blocked_candidate_verified=blocked,contact_steps=contact_steps,peak_drawer_contact_N=peak_force,cup_displacement_m=cup_delta,
            actual_tcp_position_m=ee()[0,:3].tolist(),desired_tcp_position_m=target[0,:3].tolist(),
            interpretation='This vertical approach contacts the open upper drawer before reaching the cup; not a universal infeasibility proof.',
            provenance='New controlled physical simulation, not an archived planner rejection.',frames=len(rows),fps=20,resolution=[1280,720])
        np.savez_compressed(args.output_dir/'candidate.npz',state=rows,action=actions,contact_force=forces,phase=phases)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        if not blocked:raise RuntimeError('Did not verify the intended obstructed candidate')
    finally:
        if process:process.stdin.close();process.wait()
        env.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc();raise
    finally:app.close()
