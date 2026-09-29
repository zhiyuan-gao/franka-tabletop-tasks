"""Verify and publish a contact-driven, open-gripper palm drawer closing demo."""
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
    parser.add_argument('--variant-label',choices=['alternate_contact'],default=None)
    parser.add_argument('--reference-source',type=Path,default=None)
    parser.add_argument('--video-only-destination',type=Path,default=None)
    args=parser.parse_args();src=args.source.resolve();dst=args.destination.resolve()
    report=json.loads((src/'report.json').read_text());p=np.load(src/'episode.npz')
    assert report['success'] and report['method']=='palm_push'
    root=Path(__file__).resolve().parents[1]
    with np.load(root/'outputs/icra_materials_build/drawer_panel_v2/episode.npz') as original:
        assert np.array_equal(p['oracle'][0],original['oracle'][0])
    assert p['contact_body_names'][2]=='panda_hand'
    f=np.linalg.norm(p['contact_forces'],axis=-1)
    others=[i for i in range(f.shape[-1]) if i!=2]
    assert f[:,:,others].max()<.1
    assert np.all(p['action'][:,-1]==1.) and p['oracle'][:,7:9].min()>.0375
    close=p['phase']=='close';palm=close&(f[:,:,2].max(1)>.1)
    assert palm.sum()>=20 and palm.sum()/close.sum()>.7
    opening=p['drawer_opening'];assert opening[palm].max()>.20 and opening[palm].min()<.02
    assert opening[-10:].max()<.01 and np.abs(p['oracle'][-10:,-1]).max()<.02
    assert f[-10:].max()<.1 and np.all(p['phase'][-10:]=='hold')
    point=p['contact_points_drawer'][:,:,2]
    mask=close[:,None]&(f[:,:,2]>.1)&np.isfinite(point).all(-1)
    samples=point[mask];assert len(samples)>20
    assert np.abs(samples[:,0]).max()<.177 and np.abs(samples[:,2]).max()<.069
    assert np.abs(samples[:,1]-.170).max()<.015
    validation=validate(src/'demo.mp4',fps=20,expected_frames=len(opening))
    report.update(initial_state_matches_original=True,
        min_measured_gripper_aperture_m=float(p['oracle'][:,7:9].sum(1).min()),
        palm_contact_median_drawer_xyz_m=np.median(samples,axis=0).tolist(),
        palm_contact_5_95_percentile_xyz_m=np.percentile(samples,[5,95],axis=0).tolist(),
        interpretation='Open gripper, downward fingers; palm side physically pushes drawer panel. No finger or other monitored arm-link contact with upper drawer above 0.1 N; passive drawer remains closed after withdrawal.')
    name='drawer_close_palm_push_open_gripper_success'
    if args.variant_label:
        assert args.reference_source is not None, 'Alternate contact requires measured reference comparison'
        with np.load(args.reference_source/'episode.npz') as reference:
            reference_report=json.loads((args.reference_source/'report.json').read_text())
            assert reference_report['success'] and reference_report['method']=='palm_push'
            assert np.array_equal(p['oracle'][0],reference['oracle'][0])
            assert np.all(reference['action'][:,-1]==1.)
            rf=np.linalg.norm(reference['contact_forces'],axis=-1)
            rp=reference['contact_points_drawer'][:,:,2]
            rm=(reference['phase']=='close')[:,None]&(rf[:,:,2]>.1)&np.isfinite(rp).all(-1)
            rs=rp[rm];assert len(rs)>20
            median=np.median(samples,axis=0);old_median=np.median(rs,axis=0)
            separation=float(np.linalg.norm(median-old_median))
            assert separation>.06, 'Measured palm points are not sufficiently distinct'
            report['contact_position_comparison']=dict(reference_source=str(args.reference_source.resolve()),
                reference_median_drawer_xyz_m=old_median.tolist(),new_median_drawer_xyz_m=median.tolist(),
                median_separation_m=separation,initial_state_identical=True,
                reference_contact_5_95_percentile_xyz_m=np.percentile(rs,[5,95],axis=0).tolist())
        name='drawer_close_palm_push_open_gripper_'+args.variant_label+'_success'
    for source,target in {src/'demo.mp4':dst/(name+'.mp4'),src/'episode.npz':dst/'trajectories'/(name+'.npz'),
            src/'runtime_bindings.json':dst/'metadata'/(name+'_bindings.json')}.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
        else:shutil.copy2(source,target)
    (dst/'metadata'/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    candidates=np.flatnonzero(palm)
    frame_index=int(candidates[np.argmin(np.abs(opening[candidates]-.11))])
    capture=cv2.VideoCapture(str(src/'demo.mp4'));capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
    ok,frame=capture.read();capture.release();assert ok
    (dst/'images').mkdir(exist_ok=True)
    assert cv2.imwrite(str(dst/'images'/(name+'.png')),frame)
    path=dst/'metadata'/'manifest.json';manifest=json.loads(path.read_text())
    entry=dict(file=name+'.mp4',source=str(src),validation=validation,report=report,contact_image_frame=frame_index)
    previous=[x for x in manifest['videos'] if x['file']==entry['file']]
    if previous:assert previous==[entry]
    else:manifest['videos'].append(entry)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    if args.video_only_destination:
        args.video_only_destination.mkdir(parents=True,exist_ok=True)
        target=args.video_only_destination/(name+'.mp4')
        if target.exists():assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256((src/'demo.mp4').read_bytes()).digest()
        else:shutil.copy2(src/'demo.mp4',target)
    print(json.dumps(dict(validation=validation,report=report),indent=2))


if __name__=='__main__':main()
