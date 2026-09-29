"""Validate a physical two-contact demo pair and publish only its MP4 files."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from build_paper_sim_video720p import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upper', type=Path, required=True)
    parser.add_argument('--lower', type=Path, required=True)
    parser.add_argument('--destination', type=Path,
                        default=Path('/raid/home/than/paper_sim_video720p'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    reports, initial_states, heights, copies = [], [], [], []
    for region, folder in [('upper', args.upper), ('lower', args.lower)]:
        report = json.loads((folder / 'report.json').read_text())
        with np.load(folder / 'episode.npz') as episode:
            state = episode['oracle']
            force = episode['contact_forces']
            points = episode['contact_points_door']
            contact = np.isfinite(points).all(-1) & (np.linalg.norm(force, axis=-1) > .05)
            pushing = np.isin(episode['phase'], ['touch', 'push'])
            actual = points[contact & pushing[:, None]]
            assert report['success'] and report['contact'] == region, report
            assert len(actual) >= 5
            median = np.median(actual, axis=0)
            assert abs(median[2] - report['planned_contact_door_xyz'][2]) < .035
            assert abs(median[2] - report['contact_height_median_m']) < 1e-6
            assert episode['phase'][-1] == 'hold'
            assert np.max(abs(state[-10:, -2])) < np.deg2rad(2)
            assert np.max(abs(state[-10:, -1])) < .1
            assert np.max(np.linalg.norm(force[-1], axis=-1)) < .5
            initial_states.append(state[0].copy())
            heights.append(float(median[2]))
            video = validate(folder / 'demo.mp4', fps=20, expected_frames=len(state))
        destination = args.destination / f'microwave_close_{region}_contact_success.mp4'
        if destination.exists():
            raise FileExistsError(destination)
        reports.append(dict(region=region, source=str(folder.resolve()),
                            destination=str(destination), median_contact_xyz_m=median.tolist(),
                            physical_contact_samples=len(actual), video=video))
        copies.append((folder / 'demo.mp4', destination))
    np.testing.assert_allclose(initial_states[0], initial_states[1], atol=1e-6, rtol=0)
    assert heights[0] - heights[1] > .03, heights
    args.destination.mkdir(parents=True, exist_ok=True)
    for source, destination in copies:
        shutil.copy2(source, destination)
    result = dict(success=True, identical_initial_state=True,
                  contact_height_separation_m=heights[0] - heights[1], episodes=reports)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
