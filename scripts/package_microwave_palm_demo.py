"""Validate and add a palm-push demo, preserving all existing/user videos."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil

import cv2
import numpy as np
from build_paper_sim_video720p import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=Path('/raid/home/than/icra_video_materials'))
    parser.add_argument('--variant-label',default='')
    parser.add_argument('--reference-source',type=Path)
    args=parser.parse_args()
    src=args.source.resolve();dst=args.destination.resolve()
    report=json.loads((src/'report.json').read_text())
    assert report['success'] and report['effector']=='palm'
    data=np.load(src/'episode.npz')
    bodies=report['contact_body_order']
    forces=np.linalg.norm(data['contact_forces'].reshape(-1,5,len(bodies),3),axis=-1)
    push=data['phase']=='push'
    palm=forces[:,:,bodies.index('panda_hand')].max(axis=1)>.05
    angles=np.rad2deg(data['oracle'][:,-2])
    contact_angles=angles[push&palm]
    coverage=float(palm[push].mean())
    assert coverage>.5, ('Insufficient sustained palm contact',coverage)
    assert contact_angles.min()<-70 and contact_angles.max()>-10, contact_angles
    assert report['palm_contact_impulse_fraction']>.95
    assert forces[:,:,:2].max()<.05, 'Finger contact detected'
    assert np.abs(angles[-10:]).max()<2
    assert np.abs(data['oracle'][-10:,-1]).max()<.1
    assert forces[-10:].max()<.5
    assert abs(angles[0]+86)<.01
    assert np.allclose(data['oracle'][0,:7],[0,-np.pi/4,0,-3*np.pi/4,0,np.pi/2,np.pi/4],atol=1e-5)
    validation=validate(src/'demo.mp4',fps=20,expected_frames=len(angles))
    additions=dict(push_frames_with_palm_contact_fraction=coverage,
        palm_contact_door_angle_range_deg=[float(contact_angles.min()),float(contact_angles.max())],
        peak_finger_force_N=float(forces[:,:,:2].max()),
        final_withdrawn_max_contact_N=float(forces[-10:].max()))
    report.update(additions)
    if args.reference_source:
        assert args.variant_label, 'A comparison variant needs its own filename'
        reference=args.reference_source.resolve()
        ref_report=json.loads((reference/'report.json').read_text())
        ref=np.load(reference/'episode.npz')
        assert np.array_equal(data['oracle'][0],ref['oracle'][0]), 'Different initial state'
        assert report['gripper_command']==ref_report['gripper_command']=='always_open'
        assert report['passive_hinge_damping']==ref_report['passive_hinge_damping']
        def palm_points(episode, episode_bodies):
            n=len(episode_bodies);i=episode_bodies.index('panda_hand')
            f=np.linalg.norm(episode['contact_forces'].reshape(-1,5,n,3),axis=-1)[:,:,i]
            points=episode['contact_points_door'].reshape(-1,5,n,3)[:,:,i]
            mask=np.isin(episode['phase'],['touch','push'])[:,None] & (f>.05) & np.isfinite(points).all(axis=-1)
            assert mask.sum()>=5
            return points[mask]
        current_points=palm_points(data,bodies)
        previous_points=palm_points(ref,ref_report['contact_body_order'])
        center=np.median(current_points,axis=0);previous=np.median(previous_points,axis=0)
        separation=float(np.linalg.norm(center-previous))
        assert separation>.06, ('Contact positions not sufficiently different',separation)
        additions.update(reference_source=str(reference),initial_state_identical=True,
            palm_contact_median_door_xyz_m=center.tolist(),
            reference_palm_contact_median_door_xyz_m=previous.tolist(),
            palm_contact_median_separation_m=separation,
            palm_contact_door_xyz_5_95_percentile_m=np.percentile(current_points,[5,95],axis=0).tolist())
        report.update(additions)
    name='microwave_close_palm_push_success'
    if report.get('gripper_command')=='always_open':
        assert np.all(data['action'][:,-1]==1.)
        assert np.min(data['finger_joint_pos'].sum(axis=1))>.075
        name='microwave_close_palm_push_open_gripper_success'
    if args.variant_label:
        assert re.fullmatch('[a-z0-9_]+',args.variant_label), 'Invalid variant label'
        name=name.removesuffix('_success')+'_'+args.variant_label+'_success'
    copies={src/'demo.mp4':dst/(name+'.mp4'),src/'episode.npz':dst/'trajectories'/(name+'.npz')}
    for source,target in copies.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest(),target
        else:shutil.copy2(source,target)
    metadata=dst/'metadata'/(name+'.json')
    metadata.write_text(json.dumps(report,indent=2)+'\n')
    # Choose a physically measured mid-sweep contact, not just an attractive pose.
    indices=np.flatnonzero(push&palm)
    idx=int(indices[np.argmin(np.abs(angles[indices]+45))])
    capture=cv2.VideoCapture(str(src/'demo.mp4'));capture.set(cv2.CAP_PROP_POS_FRAMES,idx)
    ok,frame=capture.read();capture.release();assert ok
    (dst/'images').mkdir(exist_ok=True)
    assert cv2.imwrite(str(dst/'images'/(name+'.png')),frame)
    manifest_path=dst/'metadata'/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    entry=dict(file=name+'.mp4',source=str(src),validation=validation,report=report,
        contact_image_frame=idx,contact_image='images/'+name+'.png')
    previous=[x for x in manifest['videos'] if x['file']==entry['file']]
    if previous:assert previous==[entry]
    else:manifest['videos'].append(entry)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(validation=validation,**additions,contact_image_frame=idx),indent=2))


if __name__=='__main__':main()
