"""Publish a verified human-inspired grasp-approach failure, without overstating it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from build_paper_sim_video720p import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=Path('/raid/home/than/icra_video_materials'))
    parser.add_argument('--variant-label',choices=['previous_camera'],default=None)
    args=parser.parse_args();src=args.source.resolve();dst=args.destination.resolve()
    report=json.loads((src/'report.json').read_text());p=np.load(src/'episode.npz')
    mapping=json.loads((src/'human_motion_mapping.json').read_text())
    side=mapping.get('approach_mode')=='side_lateral'
    assert report['failure_verified'] and report['failure_stage'] in (['side_entry','rim_approach'] if side else ['rim_approach'])
    assert not report['physical_grasp_verified'] and not report['task_completed']
    assert not report['scene_or_target_modified']
    reference=Path(__file__).resolve().parents[1]/'outputs/cup_cabinet_tasks/below_upper_drawer_to_cabinet_top_side_grasp_v11/episodes/episode_000000.npz'
    with np.load(reference) as original:
        assert np.array_equal(p['oracle'][0],original['observation.oracle'][0])
        assert bool(original['success'][-1])
    stop=int(report['safety_stop_frame']);first=int(report['first_collision_frame'])
    f=np.linalg.norm(p['drawer_contact_forces'],axis=-1);peak=f.max((1,2))
    assert np.all(peak[stop-7:stop+1]>1.) and peak[:first].max()<=1.
    assert p['phase'][first] in (['side_entry','rim_approach'] if side else ['rim_approach'])
    assert np.all(p['phase'][stop:]=='safety_hold') and len(p['phase'][stop:])>=20
    assert np.all(p['action'][:,-1]==1.)
    assert report['cup_displacement_m']<.005
    displacement=np.linalg.norm(p['oracle'][:,25:28]-p['oracle'][0,25:28],axis=1)
    assert displacement.max()<.005
    assert np.abs(p['tool_tilt_deg'][first:stop+1]).max()<5., 'This publisher labels a vertical robot grasp'
    desired=np.asarray(report['desired_pose_at_collision_stop'])
    actual=np.asarray(report['actual_pose_at_collision_stop'])
    gap=float(np.linalg.norm(actual[:3]-desired[:3]));assert gap>.015
    assert f[:,:,0].max()<.1, 'Cup contact is not the expected pregrasp failure'
    assert not np.isin(p['phase'],['grasp','extract','lift','above_cabinet','place']).any()
    human=Path(mapping['source_video']);assert hashlib.sha256(human.read_bytes()).hexdigest()==mapping['source_sha256']
    validation=validate(src/'demo.mp4',fps=20,expected_frames=len(p['oracle']))
    if side:
        lateral=np.asarray(mapping['cabinet_lateral_axis_w']);normal=np.asarray(mapping['cabinet_front_axis_w'])
        entry=np.flatnonzero(p['phase']=='side_entry');assert len(entry)>10
        movement=p['ee_pose'][entry[-1],:3]-p['ee_pose'][entry[0],:3]
        local_movement=np.array([movement@lateral,movement@normal,movement[2]])
        measured_path=p['ee_pose'][entry,:3]
        normal_span=float(np.ptp(measured_path@normal));vertical_span=float(np.ptp(measured_path[:,2]))
        # Verify approach direction BEFORE collision. Contact reaction can
        # deflect the wrist afterwards; a very early block need not permit
        # an arbitrary minimum of 25 mm lateral progress.
        free_entry=entry[entry<first];assert len(free_entry)>=4
        free_path=p['ee_pose'][free_entry,:3]
        free_delta=free_path[-1]-free_path[0]
        free_local=np.array([free_delta@lateral,free_delta@normal,free_delta[2]])
        assert free_local[0]*np.sign(mapping['side_clearance_m'])<-.005
        assert abs(free_local[0])>3*abs(free_local[1])
        assert np.ptp(free_path@normal)<.005 and np.ptp(free_path[:,2])<.01
        planned=mapping['planned_tcp_poses_wxyz']
        delta=np.asarray(planned['side_entry'][:3])-planned['outside'][:3]
        assert abs(delta@normal)<1e-6 and abs(delta[2])<1e-6
        report['measured_side_entry_delta_lateral_front_vertical_m']=local_movement.tolist()
        report['side_entry_direction_verified']=True
        report['side_entry_front_axis_span_m']=normal_span
        report['side_entry_vertical_span_m']=vertical_span
        report['contact_free_side_entry_delta_lateral_front_vertical_m']=free_local.tolist()
        report['direction_validation_note']='Lateral commanded path and contact-free measured approach verified; post-contact deflection recorded separately. An early collision is retained rather than forced past the obstacle.'
    report.update(initial_state_matches_successful_side_grasp=True,
        maximum_cup_displacement_m=float(displacement.max()),unreached_grasp_tcp_distance_m=gap,
        failure_category='human_inspired_lateral_entry_top_grasp_collision_before_grasp' if side else 'human_inspired_top_grasp_approach_collision_before_grasp',
        human_motion_mapping=mapping,
        paper_caption=('A manually retargeted human-inspired lateral approach from the SIDE of the drawer, with the gripper pointing down, is blocked before grasping. ' if side else 'A manually retargeted human-inspired top-grasp approach is blocked by the open drawer before grasping. ')+
            'The same initial scene admits the demonstrated side-grasp solution. This does not establish that every top-grasp trajectory is infeasible.')
    name='cup_human_side_entry_top_grasp_blocked' if side else 'cup_human_top_grasp_transfer_blocked'
    if args.variant_label:
        assert report.get('camera_only_replay',{}).get('recorded_actions_replayed')
        name+='_'+args.variant_label
    for source,target in {src/'demo.mp4':dst/(name+'.mp4'),src/'episode.npz':dst/'trajectories'/(name+'.npz'),
            src/'human_motion_mapping.json':dst/'metadata'/(name+'_mapping.json')}.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
        else:shutil.copy2(source,target)
    (dst/'metadata'/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    capture=cv2.VideoCapture(str(src/'demo.mp4'))
    screenshots=[('initial',0),('collision',stop)]
    if side:screenshots.insert(1,('side_start',int(entry[0])))
    for suffix,index in screenshots:
        capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read();assert ok
        assert cv2.imwrite(str(dst/'images'/f'{name}_{suffix}.png'),frame)
    capture.release()
    manifest_path=dst/'metadata'/'manifest.json';manifest=json.loads(manifest_path.read_text())
    entry=dict(file=name+'.mp4',source=str(src),validation=validation,report=report)
    previous=[item for item in manifest['videos'] if item['file']==entry['file']]
    if previous:assert previous==[entry]
    else:manifest['videos'].append(entry)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(validation=validation,report=report),indent=2))


if __name__=='__main__':main()
