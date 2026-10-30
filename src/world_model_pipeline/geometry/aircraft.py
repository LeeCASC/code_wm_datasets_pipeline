#!/usr/bin/env python3
"""CPU-only coarse aircraft world, calibrated views, and CWM condition producer.

No Blender, model weights, remote API, or learned 3D reconstruction is used.
RGB outputs are diagnostic proxy renders, not CWM-generated observations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw


from world_model_pipeline.paths import ROOT
VIEWS = ("side", "rear", "fixed")
PALETTE = np.array([
    [0, 0, 0], [126, 184, 225], [46, 120, 180], [114, 145, 86],
    [85, 88, 94], [40, 105, 45], [172, 163, 146], [126, 111, 96],
    [218, 111, 107], [181, 147, 104], [230, 181, 63], [145, 92, 160],
], dtype=np.uint8)


@dataclass(frozen=True)
class Scene:
    fps: int = 24
    windows: int = 2
    initial_speed_mps: float = 8.0
    acceleration_mps2: float = 4.0
    rotation_start_s: float = 3.5
    rotation_end_s: float = 5.5
    liftoff_s: float = 4.8
    climb_acceleration_mps2: float = 1.8
    pitch_deg: float = 10.0
    horizontal_fov_deg: float = 58.0
    description: str = "A white airplane accelerates, raises its nose, and takes off."

    @property
    def frames(self):
        return 124 + 90 * (self.windows - 1)

    def validate(self):
        if self.fps != 24 or type(self.windows) is not int or self.windows < 1:
            raise ValueError("CWM requires 24 fps and a positive integer window count")
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError("Scene values must be finite")
        if not 0 <= self.rotation_start_s < self.rotation_end_s:
            raise ValueError("rotation_end_s must follow rotation_start_s")
        if not 0 < self.liftoff_s < (self.frames - 1) / self.fps:
            raise ValueError("The takeoff must occur inside the sampled sequence")
        if min(self.initial_speed_mps, self.acceleration_mps2, self.climb_acceleration_mps2) <= 0:
            raise ValueError("Speed and accelerations must be positive")
        if not 5 < self.horizontal_fov_deg < 150 or not 0 <= self.pitch_deg <= 25:
            raise ValueError("Unsupported FOV or pitch")


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray


def aircraft_mesh() -> Mesh:
    """Editable approximate 18 m long, 20 m span plane; no reference fitting."""
    vertices, faces, colors = [], [], []

    def add(v, f, color):
        offset = len(vertices)
        vertices.extend(v)
        faces.extend([[i + offset for i in face] for face in f])
        colors.extend([color] * len(f))

    def prism(polygon, delta, color):
        p = np.asarray(polygon, dtype=float)
        n = len(p)
        v = np.concatenate([p, p + np.asarray(delta)])
        f = []
        for i in range(1, n - 1):
            f.extend([(0, i, i + 1), (n, n + i + 1, n + i)])
        for i in range(n):
            j = (i + 1) % n
            f.extend([(i, j, n + j), (i, n + j, n + i)])
        add(v, f, color)

    def box(center, extent, color):
        x, y, z = np.asarray(center) - np.asarray(extent) / 2
        a, b, c = extent
        prism([(x, y, z), (x+a, y, z), (x+a, y+b, z), (x, y+b, z)], (0, 0, c), color)

    # Low-poly tapered body; details deliberately omitted.
    rings = [(-9, .06), (-7, .5), (-4, .95), (3.8, 1), (6.5, .72), (9, .04)]
    segments = 12
    v = [[x, .85*r*np.cos(a), .9*r*np.sin(a)]
         for x, r in rings for a in np.arange(segments) * (2*np.pi/segments)]
    f = []
    for j in range(len(rings)-1):
        for i in range(segments):
            a, b = j*segments+i, j*segments+(i+1)%segments
            f.extend([(a, b, b+segments), (a, b+segments, a+segments)])
    for i in range(1, segments-1):
        f.extend([(0, i+1, i), ((len(rings)-1)*segments,
                 (len(rings)-1)*segments+i, (len(rings)-1)*segments+i+1)])
    add(v, f, (224, 228, 234))
    for side in (-1, 1):
        prism([(2, side*.65, -.05), (-2, side*.65, -.05),
               (-3.8, side*10, .32), (-1.9, side*10, .32)], (0, 0, .18), (194, 204, 217))
        prism([(-5.9, side*.4, .3), (-8.4, side*.4, .3),
               (-8.7, side*3.7, .6), (-7.3, side*3.7, .6)], (0, 0, .15), (173, 191, 213))
        box((.1, side*3.2, -.65), (3, 1.0, 1.0), (139, 155, 175))
        box((-1.5, side*1.65, -1.15), (.16, .16, .8), (121, 130, 141))
        box((-1.5, side*1.65, -1.55), (.75, .34, .5), (33, 37, 43))
    prism([(-8.6, -.09, .45), (-5.7, -.09, .45), (-7.8, -.09, 3.4)],
          (0, .18, 0), (58, 99, 160))
    box((5.8, 0, -1.15), (.16, .16, .8), (121, 130, 141))
    box((5.8, 0, -1.55), (.7, .3, .5), (33, 37, 43))
    return Mesh(np.asarray(vertices, dtype=float), np.asarray(faces), np.asarray(colors, dtype=np.uint8))


def world_state(scene: Scene, t: float, mesh: Mesh):
    u = np.clip((t-scene.rotation_start_s)/(scene.rotation_end_s-scene.rotation_start_s), 0, 1)
    pitch = np.deg2rad(scene.pitch_deg) * u*u*(3-2*u)
    c, s = np.cos(pitch), np.sin(pitch)
    # Positive nose-up pitch is a NEGATIVE right-handed rotation around +Y.
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    rotated = mesh.vertices @ rotation.T
    climb_t = max(0.0, t-scene.liftoff_s)
    clearance = .5 * scene.climb_acceleration_mps2 * climb_t**2
    position = np.array([scene.initial_speed_mps*t + .5*scene.acceleration_mps2*t*t,
                         0, -rotated[:, 2].min() + clearance])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, position
    phase = "climb" if t > scene.liftoff_s else "rotation" if t >= scene.rotation_start_s else "ground_roll"
    return transform, float(pitch), float(clearance), phase


def camera_matrix(eye, target):
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, forward], axis=1)
    c2w[:3, 3] = eye
    return c2w


def camera_for(view, position):
    p = np.asarray(position)
    if view == "side":
        return camera_matrix(p+[-14, -55, 11], p+[1.5, 0, 0])
    if view == "rear":
        return camera_matrix(p+[-52, -24, 13], p+[0, 0, 0])
    if view == "fixed":
        # Both pose and intrinsics remain constant for this camera.
        return camera_matrix([115, -225, 75], [115, 0, 10])
    raise ValueError(view)


def intrinsics(width, height, fov_deg):
    f = width/(2*np.tan(np.deg2rad(fov_deg)/2))
    # Coordinates refer to pixel centers (x+.5,y+.5).
    return np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1.]])


def clip_near(triangle, near=.05):
    """Clip in camera coordinates before perspective divide."""
    output = []
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        inside_a, inside_b = a[2] >= near, b[2] >= near
        if inside_a:
            output.append(a)
        if inside_a != inside_b:
            output.append(a+(b-a)*((near-a[2])/(b[2]-a[2])))
    return [np.asarray([output[0], output[i], output[i+1]]) for i in range(1, len(output)-1)]


def raster_triangle(triangle, color, depth, semantic, rgb, k):
    h, w = depth.shape
    xy = triangle[:, :2]/triangle[:, 2:3]
    xy = xy * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    lo = np.maximum(np.floor(xy.min(axis=0)-.5).astype(int), 0)
    hi = np.minimum(np.ceil(xy.max(axis=0)-.5).astype(int), [w-1, h-1])
    if (hi < lo).any():
        return
    (ax, ay), (bx, by), (cx, cy) = xy
    denominator = (by-cy)*(ax-cx) + (cx-bx)*(ay-cy)
    if abs(denominator) < 1e-10:
        return
    yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    xx, yy = xx+.5, yy+.5
    a = ((by-cy)*(xx-cx)+(cx-bx)*(yy-cy))/denominator
    b = ((cy-ay)*(xx-cx)+(ax-cx)*(yy-cy))/denominator
    c = 1-a-b
    inside = (a >= -1e-8) & (b >= -1e-8) & (c >= -1e-8)
    inv_z = a/triangle[0, 2] + b/triangle[1, 2] + c/triangle[2, 2]
    z = np.divide(1., inv_z, out=np.full_like(inv_z, np.inf), where=inv_z > 0)
    sl = np.s_[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    mask = inside & (z < depth[sl])
    depth[sl][mask] = z[mask]
    semantic[sl][mask] = 10
    rgb[sl][mask] = color


def render(mesh, transform, c2w, width, height, fov_deg):
    k = intrinsics(width, height, fov_deg)
    yy, xx = np.mgrid[:height, :width]
    rays_c = np.stack([(xx+.5-k[0, 2])/k[0, 0],
                       (yy+.5-k[1, 2])/k[1, 1], np.ones_like(xx)], axis=-1)
    rays_w = rays_c @ c2w[:3, :3].T
    eye = c2w[:3, 3]
    # Intersection with infinite ground z=0. Parameter equals camera-Z depth
    # because camera ray z is 1; it is NOT Euclidean distance along the ray.
    depth = np.divide(-eye[2], rays_w[..., 2], out=np.full((height, width), np.inf),
                      where=rays_w[..., 2] < -1e-9)
    ground = np.isfinite(depth) & (depth > .05)
    depth[~ground] = np.inf
    hit = eye + rays_w * np.where(ground, depth, 0)[..., None]
    road = ground & (np.abs(hit[..., 1]) < 12) & (hit[..., 0] > -120) & (hit[..., 0] < 1200)
    semantic = np.full((height, width), 1, dtype=np.uint8)
    semantic[ground], semantic[road] = 3, 4
    rgb = PALETTE[semantic].copy()
    markings = road & (((np.abs(hit[..., 1]) < .22) & ((hit[..., 0] % 22) < 9)) |
                       (np.abs(np.abs(hit[..., 1])-11.2) < .12))
    rgb[markings] = [224, 225, 221]
    world_vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
    camera_vertices = (world_vertices-eye) @ c2w[:3, :3]
    sun = np.array([.2, -.45, 1.])
    sun /= np.linalg.norm(sun)
    for face, base in zip(mesh.faces, mesh.colors):
        world_tri = world_vertices[face]
        normal = np.cross(world_tri[1]-world_tri[0], world_tri[2]-world_tri[0])
        normal /= max(np.linalg.norm(normal), 1e-12)
        color = np.clip(base * (.52+.48*abs(normal @ sun)), 0, 255).astype(np.uint8)
        for triangle in clip_near(camera_vertices[face]):
            raster_triangle(triangle, color, depth, semantic, rgb, k)
    depth[~np.isfinite(depth)] = 0
    return depth.astype("<f4"), semantic, rgb


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def export_obj(mesh, path):
    with Path(path).open("w") as f:
        f.write("# Procedural coarse aircraft; meters; X forward, Y left, Z up\n")
        for v in mesh.vertices:
            f.write("v " + " ".join(f"{x:.7f}" for x in v) + "\n")
        for face in mesh.faces:
            f.write("f " + " ".join(str(int(x)+1) for x in face) + "\n")


def open_video(path, width, height, fps):
    import imageio_ffmpeg
    process = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)
    return process


def close_video(process):
    try:
        process.stdin.close()
    finally:
        code = process.wait()
    if code:
        raise RuntimeError(f"Video encoder failed with exit code {code}")


def panel(rgb, label, t):
    im = Image.fromarray(rgb).resize((672, 384), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 672, 29), fill=(16, 22, 30))
    draw.text((10, 9), f"{label.upper()} | t={t:.3f}s | CPU PROXY - NOT CWM", fill="white")
    return np.asarray(im)


def cwm_template(view_dir, scene, view, models=None):
    raw = json.loads((ROOT/"third_party/code_world_model/examples/config.example.json").read_text())
    model_root = ROOT/"models"
    defaults = {
        "dit": model_root/"diffusion_models/minimax_h3_ref2va_bf16.safetensors",
        "video_vae": model_root/"vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": model_root/"vae/minimax_h3_audio_vae_fp32.safetensors",
        "text_encoder": model_root/"text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
        "processor": model_root/"processors/Qwen3-VL-32B-Instruct",
        "lora": model_root/"lora/minimax-h3-cwm.safetensors",
    }
    raw["models"] = {key: str(value) for key, value in (models or defaults).items()}
    raw.update(anchor_image=str(view_dir/"anchor_proxy.png"), condition_root=str(view_dir/"condition"),
               cache_dir=str(view_dir/"cwm_cache"), output=str(view_dir/"cwm_generated.mp4"))
    view_text = {"side": "A camera tracks the airplane from its left side.",
                 "rear": "A camera follows behind and to the left of the airplane.",
                 "fixed": "A stationary camera beside the runway observes the airplane passing and climbing."}[view]
    raw["windows"] = [{"start_frame": i*90,
                       "prompt": f"[{i*90/24:.2f}s-{(i*90+124)/24:.2f}s] {scene.description} {view_text} "
                                 "Follow the supplied depth and semantic reference for the airplane position, orientation, and scene layout."}
                      for i in range(scene.windows)]
    return raw


def generate(output, scene, views=VIEWS):
    scene.validate()
    output = Path(output).resolve()
    # Refuse reuse so changed inputs cannot inherit stale conditions or caches.
    output.mkdir(parents=True, exist_ok=False)
    mesh = aircraft_mesh()
    export_obj(mesh, output/"aircraft_proxy.obj")
    write_json(output/"scene.json", asdict(scene))
    states = []
    for frame in range(scene.frames):
        t = frame/scene.fps
        transform, pitch, clearance, phase = world_state(scene, t, mesh)
        states.append({"frame": frame, "time_s": t, "phase": phase,
                       "T_world_from_body": transform.tolist(), "pitch_nose_up_rad": pitch,
                       "ground_clearance_m": clearance})
    positions = np.array([s["T_world_from_body"] for s in states])[:, :3, 3]
    velocities = np.gradient(positions, 1/scene.fps, axis=0, edge_order=2)
    for state, velocity in zip(states, velocities):
        state["linear_velocity_world_mps"] = velocity.tolist()
    write_json(output/"flight_trajectory.json", {
        "world_frame": "right-handed: X runway forward, Y left, Z up; meters",
        "body_frame": "X nose forward, Y left wing, Z up",
        "transform_convention": "column vectors; p_world = T_world_from_body @ p_body",
        "provenance": "scripted kinematics, NOT a calibrated aerodynamic simulation or measured real motion",
        "velocity_method": "second-order finite differences of sampled body-origin position",
        "fps": scene.fps, "frames": states,
    })
    cameras = {view: [] for view in views}
    writers, montage = {}, None
    start_time = time.monotonic()
    try:
        for view in views:
            view_dir = output/view
            (view_dir/"condition").mkdir(parents=True)
            writers[view] = open_video(view_dir/"proxy.mp4", 672, 384, scene.fps)
            write_json(view_dir/"cwm_config.template.json", cwm_template(view_dir, scene, view))
        montage = open_video(output/"multiview_proxy.mp4", 672*len(views), 384, scene.fps)
        for state in states:
            frame, t = state["frame"], state["time_s"]
            transform = np.asarray(state["T_world_from_body"])
            panels = []
            for view in views:
                view_dir = output/view
                c2w = camera_for(view, transform[:3, 3])
                depth, semantic, rgb = render(mesh, transform, c2w, 336, 192, scene.horizontal_fov_deg)
                depth.tofile(view_dir/"condition"/f"{frame:06d}.depth.f32")
                Image.fromarray(semantic).save(view_dir/"condition"/f"{frame:06d}.semantic_id.png")
                camera = {"frame": frame, "time_s": t,
                          "T_world_from_camera": c2w.tolist(), "T_camera_from_world": np.linalg.inv(c2w).tolist(),
                          "aircraft_visible_pixels": int((semantic == 10).sum())}
                cameras[view].append(camera)
                preview = panel(rgb, view, t)
                writers[view].stdin.write(preview.tobytes())
                panels.append(preview)
                if frame in {0, round(scene.liftoff_s*scene.fps), scene.frames-1}:
                    Image.fromarray(preview).save(view_dir/f"preview_{frame:06d}.png")
                    Image.fromarray(PALETTE[semantic]).resize((672,384), Image.Resampling.NEAREST).save(view_dir/f"semantic_{frame:06d}.png")
                if frame == 0:
                    # No photo was supplied. This is intentionally a proxy placeholder.
                    Image.fromarray(rgb).resize((1344,768), Image.Resampling.NEAREST).save(view_dir/"anchor_proxy.png")
            combined = np.concatenate(panels, axis=1)
            montage.stdin.write(combined.tobytes())
            if frame in {0, scene.frames-1}:
                Image.fromarray(combined).save(output/f"multiview_{frame:06d}.png")
            if frame % 48 == 0:
                print(f"rendered {frame+1}/{scene.frames} shared states, {len(views)} views", flush=True)
    finally:
        errors = []
        for process in [*writers.values(), montage]:
            if process is not None:
                try:
                    close_video(process)
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))
    for view in views:
        write_json(output/view/"camera.json", {
            "camera_frame": "OpenCV: X right, Y down, Z forward; meters",
            "pixel_coordinates": "pixel centers at (column+0.5,row+0.5); image origin at top left",
            "K_condition": intrinsics(336, 192, scene.horizontal_fov_deg).tolist(),
            "condition_size_wh": [336, 192],
            "K_preview": intrinsics(672, 384, scene.horizontal_fov_deg).tolist(),
            "preview_size_wh": [672, 384],
            "K_target": intrinsics(1344, 768, scene.horizontal_fov_deg).tolist(),
            "target_size_wh": [1344, 768], "fps": scene.fps, "frames": cameras[view],
        })
    receipt = {
        "status": "proxy_generated", "cwm_video_generated": False,
        "frame_count": scene.frames, "fps": scene.fps, "encoded_duration_s": scene.frames/scene.fps,
        "last_timestamp_s": (scene.frames-1)/scene.fps, "views": list(views),
        "depth_convention": "camera-Z in meters, little-endian float32, C-order 192x336; sky invalid=0",
        "depth_training_convention_verified": False,
        "anchor_status": "synthetic coarse proxy placeholder; no user reference image provided",
        "limitations": ["scripted trajectory, not aerodynamic ground truth", "custom aircraft domain unvalidated for CWM",
                        "cross-view neural generation consistency not established", "CWM weights/runtime not used"],
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds": round(time.monotonic()-start_time, 2),
    }
    write_json(output/"manifest.json", receipt)
    report = audit(output)
    write_json(output/"audit.json", report)
    print(json.dumps({"output": str(output), "manifest": receipt, "audit": report}, indent=2))
    return output


def audit(output):
    output = Path(output).resolve()
    manifest = json.loads((output/"manifest.json").read_text())
    states = json.loads((output/"flight_trajectory.json").read_text())["frames"]
    n = manifest["frame_count"]
    if len(states) != n:
        raise ValueError("Incorrect shared trajectory length")
    result = {"status": "pass", "checked_condition_pairs": 0, "views": {}}
    for view in manifest["views"]:
        directory = output/view/"condition"
        if len(list(directory.glob("*.depth.f32"))) != n or len(list(directory.glob("*.semantic_id.png"))) != n:
            raise ValueError(f"{view}: incorrect condition count")
        camera = json.loads((output/view/"camera.json").read_text())
        if len(camera["frames"]) != n:
            raise ValueError(f"{view}: incorrect camera length")
        visible, beyond_far, aircraft_total = [], 0, 0
        for frame, c, state in zip(range(n), camera["frames"], states):
            path = directory/f"{frame:06d}.depth.f32"
            if path.stat().st_size != 258048:
                raise ValueError(f"Incorrect depth size: {path}")
            depth = np.fromfile(path, dtype="<f4").reshape(192, 336)
            with Image.open(directory/f"{frame:06d}.semantic_id.png") as image:
                if image.mode != "L" or image.size != (336,192):
                    raise ValueError("Semantic PNG must be grayscale L at 336x192")
                semantic = np.asarray(image)
            if not np.isfinite(depth).all() or (depth < 0).any() or semantic.max() > 11:
                raise ValueError("Invalid depth or semantic range")
            if not np.all(depth[semantic == 1] == 0) or not np.all(depth[semantic == 10] > .001):
                raise ValueError("Incorrect sky or aircraft depth")
            visible.append(int((semantic == 10).sum()))
            aircraft_total += visible[-1]
            beyond_far += int(((semantic == 10) & (depth > 256)).sum())
            if visible[-1] == 0:
                raise ValueError(f"{view}: aircraft outside frustum at {frame}")
            c2w, w2c = np.array(c["T_world_from_camera"]), np.array(c["T_camera_from_world"])
            if not np.allclose(c2w @ w2c, np.eye(4), atol=1e-8):
                raise ValueError("Invalid inverse camera transforms")
            if c["frame"] != frame or c["time_s"] != state["time_s"]:
                raise ValueError("Camera/world timestamps do not match")
            result["checked_condition_pairs"] += 1
        if view == "fixed" and not all(np.allclose(c["T_world_from_camera"], camera["frames"][0]["T_world_from_camera"]) for c in camera["frames"]):
            raise ValueError("Fixed camera moved")
        result["views"][view] = {"frames": n, "min_aircraft_pixels": min(visible), "max_aircraft_pixels": max(visible)}
        result["views"][view]["aircraft_depth_beyond_cwm_256m_fraction"] = beyond_far/aircraft_total
        if beyond_far:
            raise ValueError(f"{view}: aircraft exceeds CWM depth range")
    return result


def preflight(config):
    """Read-only upstream parser/path check without importing torch or loading weights."""
    sys.path.insert(0, str(ROOT/"third_party/code_world_model/src"))
    from cwm_h3_inference.config import load_config
    parsed = load_config(config, check_files=False)
    resources = {key: {"path": str(path), "exists": path.exists()} for key, path in vars(parsed.models).items()}
    packages = {}
    for name in ("torch", "safetensors", "transformers", "av", "flash_attn_interface", "musubi_tuner"):
        try:
            packages[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            packages[name] = False
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    missing = [key for key, entry in resources.items() if not entry["exists"]]
    blockers = []
    if missing:
        blockers.append("Missing configured resources: " + ", ".join(missing))
    if "A100" in gpu.stdout and not any(x in gpu.stdout for x in ("H100", "H800")):
        blockers.append("Detected A100; released CWM requires flash3. A100 needs an implemented and verified attention port.")
    if not all(packages.values()):
        blockers.append("Missing runtime packages in this interpreter")
    return {"upstream_config_parser": "pass", "output_frames": parsed.output_frames,
            "resources": resources, "packages": packages, "python": sys.version,
            "gpu": gpu.stdout.strip().splitlines(), "blockers": blockers,
            "inference_attempted": False, "weights_loaded": False,
            "note": "Read-only preflight; existence checks do not establish runtime readiness"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate_p = sub.add_parser("generate")
    generate_p.add_argument("--scene", type=Path, default=ROOT/"configs/aircraft_scene.json")
    generate_p.add_argument("--output", type=Path, required=True)
    generate_p.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    generate_p.add_argument("--windows", type=int)
    audit_p = sub.add_parser("audit")
    audit_p.add_argument("--output", type=Path, required=True)
    preflight_p = sub.add_parser("preflight")
    preflight_p.add_argument("--config", type=Path, required=True)
    preflight_p.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        scene = Scene(**json.loads(args.scene.read_text()))
        if args.windows is not None:
            scene = replace(scene, windows=args.windows)
        if len(set(args.views)) != len(args.views):
            parser.error("Duplicate views are not supported")
        generate(args.output, scene, tuple(args.views))
    elif args.command == "audit":
        print(json.dumps(audit(args.output), indent=2))
    else:
        report = preflight(args.config)
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
