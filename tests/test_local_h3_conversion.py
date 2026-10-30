import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT=Path(__file__).resolve().parents[1]
import world_model_pipeline.tools.convert as convert


def write_reference(folder,name,specs):
    tensors={};offset=0
    for key,(dtype,shape) in specs.items():
        size=int(np.prod(shape))*convert.DTYPES[dtype][1]
        tensors[key]={'dtype':dtype,'shape':shape,'data_offsets':[offset,offset+size]};offset+=size
    (folder/(name+'.json')).write_text(json.dumps({'metadata':{},'tensors':tensors}))


class ConversionTests(unittest.TestCase):
    def test_video_cast_adds_stats_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';headers=root/'headers';headers.mkdir()
            folder=src/'video_vae';(folder/'source').mkdir(parents=True)
            source=folder/'source/model.safetensors'
            values=np.array([[1.5,-2.25],[.00314159,0]],dtype=np.float32)
            save_file({'decoder.weight':values},str(source));before=source.read_bytes()
            (folder/'config.json').write_text(json.dumps({'latents_mean':[.1,.2],'latents_std':[1.,2.]}))
            write_reference(headers,'minimax_h3_video_vae_fp16',{'latents_mean':('F16',[2]),'latents_std':('F16',[2]),'decoder.weight':('F16',[2,2])})
            with patch.object(convert,'HEADERS',headers):receipt=convert.convert(src,root/'release','video_vae')
            actual=load_file(receipt['target'])
            np.testing.assert_array_equal(actual['decoder.weight'],values.astype(np.float16))
            np.testing.assert_array_equal(actual['latents_mean'],np.array([.1,.2],dtype=np.float16))
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(hashlib.sha256(Path(receipt['target']).read_bytes()).hexdigest(),receipt['sha256'])

    def test_audio_weight_norm_matches_independent_channel_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';headers=root/'headers';headers.mkdir()
            folder=src/'audio_vae';folder.mkdir(parents=True)
            values=np.arange(1,25,dtype=np.float32).reshape(2,3,4)
            gain=np.array([2,3],dtype=np.float32).reshape(2,1,1)
            save_file({'conv.weight_g':gain,'conv.weight_v':values},str(folder/'model.safetensors'))
            (folder/'config.json').write_text(json.dumps({'latents_mean':[0.,0.],'latents_std':[1.,1.]}))
            write_reference(headers,'minimax_h3_audio_vae_fp32',{'latents_mean':('F32',[2]),'latents_std':('F32',[2]),'conv.weight':('F32',[2,3,4])})
            with patch.object(convert,'HEADERS',headers):receipt=convert.convert(src,root/'release','audio_vae')
            actual=load_file(receipt['target'])
            expected=values*gain/np.sqrt(np.square(values).sum(axis=(1,2),keepdims=True))
            np.testing.assert_allclose(actual['conv.weight'],expected,rtol=1e-6,atol=1e-6)
            self.assertNotIn('conv.weight_g',actual)


if __name__=='__main__':unittest.main()
