"""Publish a collision-stopped transfer candidate, without claiming infeasibility."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from build_paper_sim_video720p import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=Path('/raid/home/than/icra_video_materials'))
    args=parser.parse_args();src=args.source.resolve();dst=args.destination.resolve()
    report=json.loads((src/'report.json').read_text());data=np.load(src/'episode.npz')
    assert report['top_grasp_and_lift_verified'] and report['placement_collision_safety_stop'] and not report['placed']
    grasp=int(report['lift_verification_frame']);hit=int(report['first_transfer_collision_frame'])
    assert grasp<hit and data['oracle'][grasp,27]-data['oracle'][0,27]>.075
    gf=data['grasp_force'].max(axis=1);cf=data['cabinet_force'].max(axis=1)
    assert np.all(gf[grasp:,0]>.1) and np.all(gf[grasp:,1]>.1), 'Lost cup grip'
    hold=np.flatnonzero(data['phase']=='collision_safety_hold');assert len(hold)>=20
    assert np.all(cf[hold[0]-4:hold[0]+1,0]>.5), 'No sustained cup-cabinet collision'
    assert np.max(cf[grasp:hit])<.5, 'Cup not clear before transfer'
    assert report['final_target_error_m']>.05
    assert np.allclose(data['oracle'][0,:7],[0,-np.pi/4,0,-3*np.pi/4,0,np.pi/2,np.pi/4],atol=1e-5)
    reference=Path(__file__).resolve().parents[1]/'outputs/cup_cabinet_tasks/cup_on_cabinet_to_upper_drawer_top_grasp_v8/episodes/episode_000000.npz'
    with np.load(reference) as original:
        assert np.array_equal(data['oracle'][0],original['observation.oracle'][0]), 'Initial scene differs from successful reference'
    validation=validate(src/'demo.mp4',fps=20,expected_frames=len(data['oracle']))
    report.update(peak_transfer_cup_cabinet_force_N=float(cf[hit:,0].max()),
        min_finger_force_after_lift_N=gf[grasp:].min(axis=0).tolist(),
        failure_category='transfer_path_collision_with_safety_stop',
        not_evidence_of_grasp_infeasibility=True)
    name='cup_top_grasp_direct_transfer_collision_stop'
    mapping={src/'demo.mp4':dst/(name+'.mp4'),src/'episode.npz':dst/'trajectories'/(name+'.npz'),
        src/'verified_top_grasp.png':dst/'images'/(name+'_grasp.png'),
        src/'placement_collision.png':dst/'images'/(name+'_collision.png')}
    for source,target in mapping.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
        else:shutil.copy2(source,target)
    (dst/'metadata'/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    path=dst/'metadata'/'manifest.json';manifest=json.loads(path.read_text())
    entry=dict(file=name+'.mp4',source=str(src),validation=validation,report=report)
    previous=[x for x in manifest['videos'] if x['file']==entry['file']]
    if previous:assert previous==[entry]
    else:manifest['videos'].append(entry)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(validation=validation,report=report),indent=2))


if __name__=='__main__':main()
