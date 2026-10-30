#!/usr/bin/env python3
"""Measure RGB tracking against the programmed ground-plane projection.

This is a sparse, planar diagnostic, not a camera-pose certificate. Generated
texture correspondences may be wrong and the output scene may be nonplanar.
No output from this script labels a neural video as strict geometric GT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image


def ground_homography(k, world_to_camera_a, world_to_camera_b):
    """For the known world plane Z=0, map image A pixel centers to image B."""
    a, b = np.asarray(world_to_camera_a), np.asarray(world_to_camera_b)
    k = np.asarray(k)
    pa = k @ a[:3, [0, 1, 3]]
    pb = k @ b[:3, [0, 1, 3]]
    h = pb @ np.linalg.inv(pa)
    return h / h[2, 2]


def project_points(h, xy):
    p = np.column_stack((xy, np.ones(len(xy)))) @ np.asarray(h).T
    return p[:, :2] / p[:, 2:3]


def static_mask(conditions, index, size):
    """Use only near/midrange ground, away from the moving aircraft boundary."""
    with Image.open(conditions / f'{index:06d}.semantic_id.png') as image:
        sem = np.asarray(image)
    depth = np.fromfile(conditions / f'{index:06d}.depth.f32', dtype='<f4').reshape(sem.shape)
    mask = np.isin(sem, [3, 4]) & (depth > .3) & (depth < 240)
    mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    aircraft = cv2.resize((sem == 10).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    aircraft = cv2.dilate(aircraft, np.ones((31, 31), np.uint8))
    mask[aircraft != 0] = 0
    return cv2.erode(mask, np.ones((5, 5), np.uint8)) * 255


def track_pair(a, b, mask_a, mask_b):
    points = cv2.goodFeaturesToTrack(a, maxCorners=1200, qualityLevel=.01,
                                    minDistance=7, mask=mask_a, blockSize=7)
    if points is None:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    lk = dict(winSize=(31, 31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
    following, forward_ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **lk)
    if following is None:
        return np.empty((0, 2)), np.empty((0, 2)), len(points)
    finite = np.isfinite(following).all(axis=(1, 2))
    following[~finite] = points[~finite]
    back, backward_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, following, None, **lk)
    if back is None:
        return np.empty((0, 2)), np.empty((0, 2)), len(points)
    fb = np.linalg.norm(back[:, 0] - points[:, 0], axis=1)
    xy = np.rint(following[:, 0]).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < b.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < b.shape[0])
    allowed = np.zeros(len(points), bool)
    allowed[inside] = mask_b[xy[inside, 1], xy[inside, 0]] > 0
    good = finite & forward_ok[:, 0].astype(bool) & backward_ok[:, 0].astype(bool) & (fb <= 1.) & allowed
    # Actual RGB flow is measured independently; the prescribed homography is
    # never used to initialize tracking or reject tracks by geometric residual.
    return points[good, 0], following[good, 0], len(points)


def flat_green_fraction(rgb, conditions, index):
    """Screen flat green blocks, not a general photorealism score."""
    with Image.open(conditions / f'{index:06d}.semantic_id.png') as image:
        mask = cv2.resize((np.asarray(image) == 3).astype(np.uint8),
                          (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    count = flat = 0
    for y in range(0, rgb.shape[0] - 15, 16):
        for x in range(0, rgb.shape[1] - 15, 16):
            roi = rgb[y:y+16, x:x+16]
            if mask[y:y+16, x:x+16].mean() < .9:
                continue
            count += 1
            color = roi.mean(axis=(0, 1))
            flat += int(roi.std(axis=(0, 1)).mean() < 3 and color[1] > color[0] + 8 and color[1] > color[2] + 8)
    return float(flat / count) if count else None


def audit(video, camera_path, conditions, output, stride=6, analysis_width=672):
    cv2.setNumThreads(4)
    camera = json.loads(camera_path.read_text())
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    if len(camera['frames']) < len(frames):
        raise ValueError('Camera sequence is shorter than decoded video')
    target_size = camera['target_size_wh']
    if [width, height] != target_size:
        raise ValueError('This diagnostic requires the recorded target image size; no implicit crop/rescale')
    size = (analysis_width, round(height * analysis_width / width))
    scale = width / size[0]
    k = np.asarray(camera['K_target'], float).copy()
    k[0] *= size[0] / width
    k[1] *= size[1] / height
    small = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in frames]
    indices = sorted(set([0, 1, *range(stride, len(frames), stride), len(frames)-1]))
    reports = []
    output.mkdir(parents=True, exist_ok=True)
    for ia, ib in zip(indices[:-1], indices[1:]):
        a, b = (cv2.cvtColor(small[i], cv2.COLOR_RGB2GRAY) for i in [ia, ib])
        ma, mb = (static_mask(conditions, i, size) for i in [ia, ib])
        ca, cb = (camera['frames'][i]['T_camera_from_world'] for i in [ia, ib])
        h = ground_homography(k, ca, cb)
        pa, pb, candidates = track_pair(a, b, ma, mb)
        # Camera calibration uses centers at col+.5,row+.5. OpenCV feature
        # coordinates use col,row, so convert explicitly in both directions.
        predicted = project_points(h, pa + .5) - .5
        errors = np.linalg.norm(pb - predicted, axis=1) * scale
        row = {'frame_a': ia, 'frame_b': ib, 'corner_candidates': candidates,
               'forward_backward_valid_tracks': len(pa), 'units': 'target_resolution_pixels',
               'diagnostic_support': 'measured' if len(pa) >= 40 else 'insufficient_tracks'}
        if len(pa):
            row.update(median_residual_px=float(np.median(errors)), p90_residual_px=float(np.percentile(errors, 90)),
                       fraction_within_4px=float(np.mean(errors <= 4)))
        if len(pa) >= 8:
            _, inliers = cv2.findHomography(pa, pb, cv2.RANSAC, 2.)
            row['observed_homography_ransac_inlier_fraction'] = float(inliers.mean()) if inliers is not None else None
        reports.append(row)
        if ia in {0, stride, stride*5}:
            overlay = cv2.cvtColor(small[ib], cv2.COLOR_RGB2BGR)
            for actual, expected in zip(pb[::6], predicted[::6]):
                if not np.isfinite(expected).all() or np.max(np.abs(expected)) > 10000:
                    continue
                cv2.line(overlay, tuple(np.rint(actual).astype(int)), tuple(np.rint(expected).astype(int)), (0, 0, 255), 1)
                cv2.circle(overlay, tuple(np.rint(actual).astype(int)), 2, (0, 170, 255), -1)
                cv2.circle(overlay, tuple(np.rint(expected).astype(int)), 2, (0, 255, 0), 1)
            cv2.putText(overlay, f'{ia}->{ib}: green=programmed, orange=RGB track', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.imwrite(str(output / f'geometry_overlay_{ia:06d}.jpg'), overlay)
    supported = [r for r in reports if r['diagnostic_support'] == 'measured']
    # Keep 0->1 separate: it tests the anchor transition rather than later motion.
    motion = [r for r in supported if r['frame_a'] != 0]
    transition = [float(np.mean(np.abs(small[i].astype(float)-small[i-1].astype(float)))) for i in range(1, len(small))]
    flatness = [{'frame': i, 'flat_green_ground_fraction': flat_green_fraction(small[i], conditions, i)} for i in indices]
    support_fraction = len(supported) / len(reports) if reports else 0
    summary = {'decoded_frames': len(frames), 'supported_pairs': len(supported), 'sampled_pairs': len(reports),
               'supported_motion_pairs': len(motion), 'track_support_fraction': support_fraction,
               'diagnostic_status': 'measured' if len(motion) >= 3 and support_fraction >= .6 else 'insufficient_track_support',
               'motion_pair_median_residual_px': float(np.median([r['median_residual_px'] for r in motion])) if motion else None,
               'first_transition_rgb_mae_0_255': transition[0],
               'later_transition_median_rgb_mae_0_255': float(np.median(transition[1:])),
               'median_flat_green_ground_fraction_after_first_frame': float(np.median([r['flat_green_ground_fraction'] for r in flatness[1:] if r['flat_green_ground_fraction'] is not None]))}
    result = {'schema': 'cwm_static_plane_diagnostic_v1', 'video': str(video.resolve()),
              'camera': str(camera_path.resolve()), 'conditions': str(conditions.resolve()),
              'analysis_size_wh': list(size), 'summary': summary, 'pairs': reports, 'flat_green_samples': flatness,
              'strict_neural_pixel_gt': False, 'camera_pose_verdict': 'not_certified',
              'limitations': ['Only the known world ground plane Z=0 is tested; no independent 6DoF pose is recovered.',
                             'RGB tracks can follow hallucinated or repeated textures, or miss nonrigid regions.',
                             'Forward/backward checking measures track reliability, not geometric truth.',
                             'Four-pixel tolerance is a comparison diagnostic, not a validated dataset acceptance criterion.',
                             'Flat-green screening is a narrow artifact indicator, not an overall realism metric.']}
    (output / 'geometry_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['video', 'camera', 'conditions', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stride', type=int, default=6)
    args = parser.parse_args()
    audit(args.video, args.camera, args.conditions, args.output, args.stride)
