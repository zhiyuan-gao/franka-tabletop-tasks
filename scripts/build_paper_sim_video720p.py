"""Build the 20 canonical 720p paper videos without modifying the originals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/raid/home/than/paper_sim_video')
DEST = Path('/raid/home/than/paper_sim_video720p')
BUILD = ROOT / 'outputs/paper_sim_video720p_build'
PYTHON = sys.executable
TASKS = [f'open_{level}_drawer' for level in ('upper', 'lower')] + [
    f'put_object_in_{level}_drawer{suffix}'
    for level in ('upper', 'lower') for suffix in ('', '_and_close')
]
ENV = os.environ.copy()
CMEEL = str(Path(PYTHON).resolve().parents[1] / 'lib/python3.11/site-packages/cmeel.prefix/lib')
ENV['LD_LIBRARY_PATH'] = CMEEL + (':' + ENV['LD_LIBRARY_PATH'] if ENV.get('LD_LIBRARY_PATH') else '')


def validate(path, fps=None, expected_frames=None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot decode {path}')
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rate = capture.get(cv2.CAP_PROP_FPS)
    advertised = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if (width, height) != (1280, 720) or count != advertised or count <= 0:
        raise RuntimeError(f'Invalid 720p video {path}: {width}x{height}, {count}/{advertised}')
    if fps is not None and abs(rate - fps) > 1e-5:
        raise RuntimeError(f'Wrong frame rate {path}: {rate}')
    if expected_frames is not None and count != expected_frames:
        raise RuntimeError(f'Wrong frame count {path}: {count}/{expected_frames}')
    return dict(file=path.name, width=width, height=height, fps=rate, frames=count,
        duration_s=count/rate, bytes=path.stat().st_size)


def publish(path, **checks):
    report = validate(path, **checks)
    target = DEST / path.name
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(path, target)
    print(json.dumps(dict(stage='published', **report)), flush=True)
    return report


def run_logged(command, log):
    with log.open('w') as handle:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); inspect {log}')


def extract_validation(log, key):
    text = log.read_text()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get(key) is True:
            return obj
    raise RuntimeError(f'Missing {key} validation in {log}')


def drawer_videos():
    reports = []
    # This native render already uses the new 21 cm success criterion and ends
    # at the 22 cm travel limit, with one verified physical handle grasp.
    name = 'drawer_open_upper_drawer_success.mp4'
    if not (DEST / name).exists():
        existing = SOURCE / 'drawer_open_upper_drawer_success_21cm_720p.mp4'
        staged = BUILD / name
        shutil.copy2(existing, staged)
        reports.append(publish(staged, fps=20, expected_frames=226))
    for task in TASKS:
        for outcome in ('success', 'failure'):
            name = f'drawer_{task}_{outcome}.mp4'
            if (DEST / name).exists():
                continue
            # Preserve a failed diagnostic render rather than silently
            # overwriting it if a build is resumed after an error.
            stage = BUILD / 'renders'
            attempt = 0
            while (stage / name).exists():
                attempt += 1
                stage = BUILD / f'renders_retry{attempt}'
            stage.mkdir(parents=True, exist_ok=True)
            path = stage / name
            log = BUILD / f'{name.removesuffix(".mp4")}_attempt{attempt}.log'
            script = 'generate_drawer_success_video.py' if outcome == 'success' else 'generate_drawer_grasp_failure_video.py'
            cmd = [PYTHON, str(ROOT/'scripts'/script), '--task', task,
                '--output', str(path), '--headless']
            if outcome == 'failure':
                cmd += ['--full-resolution']
            else:
                reference = (Path('/tmp/drawer_single_grasp_sVhwKC') if task == 'open_lower_drawer'
                    else ROOT/'outputs/drawer_subtasks_demo_bundle') / task / 'episodes/episode_000000.npz'
                cmd += ['--reference-episode', str(reference)]
            print(json.dumps(dict(stage='rendering', task=task, outcome=outcome, log=str(log))), flush=True)
            run_logged(cmd, log)
            validation = extract_validation(log, 'validated_success' if outcome == 'success' else 'validated_failure')
            report = publish(path, fps=20, expected_frames=validation.get('frames', validation.get('steps')))
            reports.append(dict(**report, validation=validation))
    return reports


def fruit_videos():
    sources = sorted(SOURCE.glob('two_fruit_*.mp4'))
    if len(sources) != 8:
        raise RuntimeError(f'Expected 8 fruit videos, got {len(sources)}')
    reports = []
    for source in sources:
        if (DEST/source.name).exists():
            continue
        path = BUILD/source.name
        capture = cv2.VideoCapture(str(source))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rate = capture.get(cv2.CAP_PROP_FPS)
        if (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != (1920,1080):
            raise RuntimeError(f'Expected native 1080p fruit source: {source}')
        capture.release()
        run_logged(['ffmpeg','-hide_banner','-loglevel','error','-n','-threads','2',
            '-i',str(source),'-vf','scale=1280:720:flags=lanczos','-an',
            '-c:v','libx264','-threads','2','-preset','slow','-crf','16',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],
            BUILD/f'{source.stem}.log')
        reports.append(publish(path, fps=rate, expected_frames=count))
    return reports


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fruit = pool.submit(fruit_videos)
        drawer = pool.submit(drawer_videos)
        reports = fruit.result() + drawer.result()
    expected = {f'drawer_{task}_{outcome}.mp4' for task in TASKS for outcome in ('success','failure')}
    expected.update(path.name for path in SOURCE.glob('two_fruit_*.mp4'))
    actual = {path.name for path in DEST.iterdir()}
    if actual != expected:
        raise RuntimeError(f'Output inventory mismatch: missing={expected-actual}, extra={actual-expected}')
    checked = [validate(DEST/name, fps=10 if name.startswith('two_fruit_') else 20) for name in sorted(expected)]
    (BUILD/'validation.json').write_text(json.dumps(dict(videos=checked, new_renders=reports), indent=2)+'\n')
    print(json.dumps(dict(stage='complete', count=len(checked), destination=str(DEST))), flush=True)


if __name__ == '__main__':
    main()
