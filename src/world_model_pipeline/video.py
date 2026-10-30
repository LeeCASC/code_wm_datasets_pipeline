"""Join silent, compatible RGB clips without a generation or encoding pass."""
from fractions import Fraction
import hashlib
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg

from .io import write_json


def concatenate(inputs, output):
    paths = [Path(value).resolve() for value in inputs]
    output = Path(output).resolve()
    if not paths:
        raise ValueError('At least one input video is required')
    if output.exists():
        raise FileExistsError(output)
    geometry = None
    timeline = []
    expected_hashes = {}
    count = 0
    for path in paths:
        if any(character in str(path) for character in "'\n\r"):
            raise ValueError('Concat input paths must not contain quotes or newlines')
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            current = (stream.width, stream.height, stream.average_rate, stream.codec_context.name)
            if geometry is not None and current != geometry:
                raise ValueError('Stream-copy inputs must have the same dimensions, frame rate and codec')
            if container.streams.audio:
                raise ValueError('This RGB concat command requires silent input videos')
            geometry = current
            local = 0
            last_hash = None
            for frame in container.decode(video=0):
                if local == 0:
                    expected_hashes[count] = hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest()
                last_hash = hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest()
                local += 1
            if local == 0:
                raise ValueError(f'Input contains no frames: {path}')
            expected_hashes[count + local - 1] = last_hash
            timeline.append({'source': str(path), 'start_frame': count, 'frames': local,
                             'start_time_s': float(count / geometry[2])})
            count += local
    output.parent.mkdir(parents=True, exist_ok=True)
    listing = output.with_suffix('.ffconcat')
    listing.write_text('ffconcat version 1.0\n' + ''.join("file '" + str(path) + "'\n" for path in paths))
    temporary = output.with_name('.' + output.stem + '.partial.mp4')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-loglevel', 'error', '-n',
               '-f', 'concat', '-safe', '0', '-i', str(listing), '-map', '0:v:0', '-c:v', 'copy',
               '-an', '-movflags', '+faststart', str(temporary)]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        times = []
        matched = 0
        with av.open(str(temporary)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                times.append(Fraction(frame.pts) * frame.time_base)
                if index in expected_hashes:
                    actual = hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest()
                    if actual != expected_hashes[index]:
                        raise ValueError(f'Concat decoded pixels differ at frame {index}')
                    matched += 1
        if len(times) != count or times[0] != 0 or any(b-a != 1/geometry[2] for a, b in zip(times, times[1:])):
            raise ValueError('Concat frame count or timestamps are incorrect')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {'status': 'pass', 'output': str(output), 'frames': count,
              'fps': str(geometry[2]), 'size_wh': list(geometry[:2]),
              'duration_s': float(count / geometry[2]), 'stream_copy': True,
              'source_boundary_hashes_matched': matched, 'sources': timeline}
    write_json(output.with_suffix('.json'), result)
    return result
