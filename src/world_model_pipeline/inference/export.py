"""Project-owned CWM export policy, leaving vendored inference code unchanged."""
from contextlib import contextmanager
import gc
import os
import uuid

import av
from PIL import Image

from world_model_pipeline.io import write_json


@contextmanager
def quality_export(crf=18):
    if not 0 <= crf <= 51:
        raise ValueError('CRF must be between 0 and 51')
    from cwm_h3_inference.engine import Ref2VAEngine
    original = Ref2VAEngine.write_video

    def write_video(self):
        import torch
        output = self.config.output
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name('.' + output.stem + '.' + uuid.uuid4().hex + '.mp4')
        count = 0
        try:
            with av.open(str(temporary), 'w', options={'movflags': '+faststart'}) as container:
                stream = container.add_stream('libx264', rate=24)
                stream.width, stream.height, stream.pix_fmt = 1344, 768, 'yuv420p'
                stream.options = {'crf': str(crf), 'preset': 'medium', 'threads': '4'}
                for index in range(len(self.config.windows)):
                    video, _audio = self._load_recovery(index)
                    with torch.inference_mode():
                        decoded = self.video_vae.decode(video.unsqueeze(0).to(self.device, torch.float16))[0, :, :124].cpu()
                    pixels = ((decoded.float().clamp(-1, 1)+1)*127.5).round().to(torch.uint8).permute(1, 2, 3, 0).contiguous()
                    if index:
                        pixels = pixels[34:]
                    for pixel in pixels:
                        rgb = pixel.numpy()
                        if count in {0, self.config.output_frames//2, self.config.output_frames-1}:
                            Image.fromarray(rgb).save(output.parent/f'decoded_{count:06d}.png')
                        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                            container.mux(packet)
                        count += 1
                    del video, decoded, pixels
                    gc.collect()
                    torch.cuda.empty_cache()
                for packet in stream.encode():
                    container.mux(packet)
            if count != self.config.output_frames:
                raise ValueError(f'Unexpected export frame count: {count}')
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        receipt = {'output': str(output), 'output_frames': count, 'fps': 24,
                   'resolution': [1344, 768], 'audio': False, 'crf': crf,
                   'codec': 'libx264', 'decoded_pngs_are_pre_compression': True,
                   'bytes': output.stat().st_size, 'seed': self.config.seed, 'steps': self.config.steps,
                   'lora_gate': self.lora_gate, 'strict_neural_pixel_gt': False}
        write_json(output.with_suffix('.mp4.json'), receipt)
        return receipt

    Ref2VAEngine.write_video = write_video
    try:
        yield
    finally:
        Ref2VAEngine.write_video = original
