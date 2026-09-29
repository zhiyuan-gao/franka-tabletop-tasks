"""Publish verified grasp-before-collision evidence for the BELOW-drawer task."""
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
    args=parser.parse_args();src=args.source.resolve();dst=args.destination.resolve()
    report=json.loads((src/'report.json').read_text());p=np.load(src/'episode.npz')
    assert report['task']=='cup_below_open_drawer_to_cabinet_top'
    assert report['correct_failure_verified'] and not report['scene_or_target_modified']
    root=Path(__file__).resolve().parents[1]
    reference=root/'outputs/cup_cabinet_tasks/below_upper_drawer_to_cabinet_top_side_grasp_v11/episodes/episode_000000.npz'
    with np.load(reference) as original:
        assert np.array_equal(p['oracle'][0],original['observation.oracle'][0])
    gf=p['grasp_force'].max(axis=1);df=p['drawer_force'].max(axis=1)
    peak=df.max(axis=1);hit=int(report['first_lift_collision_frame'])
    q=p['oracle'][:,28:32]
    cosine=1-2*(q[:,1]**2+q[:,2]**2)
    sine=np.sqrt(np.maximum(0,1-cosine**2))
    # Conservative geometric table-clearance bound for the actual tilted cup.
    lowest=p['oracle'][:,27]+np.minimum(-.026*sine,.087*cosine-.039*sine)
    free=(p['phase']=='lift')&(gf[:,:2].min(axis=1)>1.)&(lowest>.005)&(peak<.5)
    verified=np.flatnonzero(free&(np.arange(len(free))<hit));assert len(verified)>=3
    grasp=int(verified[-1])
    assert peak[:hit].max()<.5, 'An earlier drawer collision invalidates the sequence'
    assert np.all(gf[verified[0]:,:2]>1.), 'Cup grip was lost'
    hold=np.flatnonzero(p['phase']=='collision_hold');assert len(hold)>=20
    assert np.all(peak[hold[0]-9:hold[0]+1]>.5)
    tcpq=p['ee_pose'][grasp,3:]
    angle=float(np.rad2deg(np.arccos(np.clip(-(1-2*(tcpq[1]**2+tcpq[2]**2)),-1,1))))
    assert 40<angle<50, 'Publish with the measured oblique-grasp label'
    validation=validate(src/'demo.mp4',fps=20,expected_frames=len(p['oracle']))
    report.update(verified_free_grasp_frame=grasp,verified_lowest_cup_clearance_m=float(lowest[grasp]),
        actual_tool_angle_from_vertical_deg=angle,peak_drawer_contact_N=float(peak.max()),
        measured_contact_body=report['drawer_filter_order'][int(df.max(axis=0).argmax())],
        min_finger_force_after_free_grasp_N=gf[verified[0]:,:2].min(axis=0).tolist(),
        failure_category='oblique_top_grasp_success_then_upward_transfer_collision',
        interpretation='Same below-drawer cup scene and cabinet-top target. A 45-degree oblique top grasp succeeds without preceding drawer contact; the cup clears the table before upward transfer causes wrist contact with the upper drawer. Explicit collision safety stop. Not a strictly vertical grasp, not a cup-to-drawer task, and not a universal transfer-infeasibility proof.')
    name='cup_below_drawer_oblique_top_grasp_then_lift_collision'
    for source,target in {src/'demo.mp4':dst/(name+'.mp4'),src/'episode.npz':dst/'trajectories'/(name+'.npz')}.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
        else:shutil.copy2(source,target)
    (dst/'metadata'/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    capture=cv2.VideoCapture(str(src/'demo.mp4'))
    for suffix,index in [('grasp',grasp),('collision',int(hold[0]))]:
        capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read();assert ok
        assert cv2.imwrite(str(dst/'images'/f'{name}_{suffix}.png'),frame)
    capture.release()
    path=dst/'metadata'/'manifest.json';manifest=json.loads(path.read_text())
    entry=dict(file=name+'.mp4',source=str(src),validation=validation,report=report)
    previous=[x for x in manifest['videos'] if x['file']==entry['file']]
    if previous:assert previous==[entry]
    else:manifest['videos'].append(entry)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(validation=validation,report=report),indent=2))


if __name__=='__main__':main()
