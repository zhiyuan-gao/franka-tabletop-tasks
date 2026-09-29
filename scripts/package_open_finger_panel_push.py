"""Publish an open-finger drawer panel push with audited handle clearance."""
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
    assert report['success'] and report['method']=='panel_push' and report['gripper_command']=='always_open'
    root=Path(__file__).resolve().parents[1]
    reference=root/'outputs/icra_materials_build/drawer_panel_v2/episode.npz'
    with np.load(reference) as original:
        assert np.array_equal(p['oracle'][0],original['oracle'][0])
    f=np.linalg.norm(p['contact_forces'],axis=-1)
    close=p['phase']=='close';active=close&(f[:,:,:2].max((1,2))>.1)
    dual=(f[:,:,:2].max(1)>.1).all(1)
    fraction=float(dual[active].mean());assert active.sum()>=20 and fraction>.9
    assert np.all(p['action'][:,-1]==1.)
    assert p['oracle'][:,7:9].min()>.0375
    assert f[:,:,2].max()<.1
    assert p['drawer_opening'][-10:].max()<.01
    assert np.abs(p['oracle'][-10:,-1]).max()<.02
    assert f[-10:].max()<.5
    # Runtime asset's handle bar and two attachment blocks, in UpperDrawer
    # coordinates. Clearance is measured to the nearest box surface, not center.
    boxes=[([0.,.190,-.004],[.055,.0055,.0055]),
        ([-.046,.177,-.004],[.006,.013,.008]),([.046,.177,-.004],[.006,.013,.008])]
    medians=[];clearances=[];ranges=[]
    for finger in range(2):
        point=p['contact_points_drawer'][:,:,finger]
        mask=close[:,None]&(f[:,:,finger]>.1)&np.isfinite(point).all(-1)
        samples=point[mask];assert len(samples)>20
        assert np.abs(samples[:,0]).max()<.177
        assert np.abs(samples[:,1]-.170).max()<.015
        assert np.abs(samples[:,2]).max()<.069
        distance=np.min([np.linalg.norm(np.maximum(np.abs(samples-center)-half,0),axis=1) for center,half in boxes],axis=0)
        assert distance.min()>.045, ('Finger too close to handle',finger,distance.min())
        medians.append(np.median(samples,axis=0).tolist());clearances.append(float(distance.min()))
        ranges.append(np.percentile(samples,[5,95],axis=0).tolist())
    assert np.linalg.norm(np.asarray(medians[0])-medians[1])>.075
    validation=validate(src/'demo.mp4',fps=20,expected_frames=len(p['oracle']))
    report.update(initial_state_matches_original=True,min_measured_gripper_aperture_m=float(p['oracle'][:,7:9].sum(1).min()),
        fingertip_contact_median_drawer_xyz_m=medians,fingertip_contact_5_95_percentile_xyz_m=ranges,
        fingertip_min_handle_surface_clearance_m=clearances,
        bilateral_contact_among_contacting_close_frames=fraction,
        simultaneous_bilateral_substep_fraction=float((f[active,:,:2]>.1).all(2).mean()),
        fingertip_contact_separation_m=float(np.linalg.norm(np.asarray(medians[0])-medians[1])),
        interpretation='Fully open gripper pushes the front panel with two fingertips away from the handle; original scene layout and initial state preserved. Fingertip contact compliance and optional feedback settings are recorded explicitly in this report.')
    name='drawer_close_panel_push_open_gripper_far_handle_success'
    for source,target in {src/'demo.mp4':dst/(name+'.mp4'),src/'episode.npz':dst/'trajectories'/(name+'.npz'),
            src/'runtime_bindings.json':dst/'metadata'/(name+'_bindings.json')}.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
        else:shutil.copy2(source,target)
    (dst/'metadata'/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    candidates=np.flatnonzero(active&dual)
    frame_index=int(candidates[np.argmin(np.abs(p['drawer_opening'][candidates]-.11))])
    capture=cv2.VideoCapture(str(src/'demo.mp4'));capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
    ok,frame=capture.read();capture.release();assert ok
    assert cv2.imwrite(str(dst/'images'/(name+'.png')),frame)
    path=dst/'metadata'/'manifest.json';manifest=json.loads(path.read_text())
    entry=dict(file=name+'.mp4',source=str(src),validation=validation,report=report,contact_image_frame=frame_index)
    previous=[x for x in manifest['videos'] if x['file']==entry['file']]
    if previous:assert previous==[entry]
    else:manifest['videos'].append(entry)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(validation=validation,contact_medians=medians,handle_clearance=clearances,bilateral_fraction=fraction),indent=2))


if __name__=='__main__':main()
