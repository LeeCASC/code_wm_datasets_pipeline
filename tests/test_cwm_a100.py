from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import world_model_pipeline.inference.cwm as runner


class A100ConfigTests(unittest.TestCase):
    def test_progress_preserves_sampling_arguments_callback_result_and_restores_function(self):
        seen = []
        forwarded = []
        token = object()

        def sample(*args, **kwargs):
            forwarded.append((args, kwargs['seed'], kwargs['initial_video']))
            kwargs['step_callback'](1, 20)
            return token

        module = SimpleNamespace(sample_joint_av=sample)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'import_module', return_value=module):
            path = Path(tmp)/'progress.json'
            with runner.sampling_progress(path):
                result = module.sample_joint_av('model', seed=1234, initial_video=token,
                                                step_callback=lambda step, total: seen.append((step, total)))
            self.assertIs(result, token)
            self.assertIs(module.sample_joint_av, sample)
            self.assertEqual(forwarded, [(('model',), 1234, token)])
            self.assertEqual(seen, [(1, 20)])
            self.assertEqual(json.loads(path.read_text())['step'], 1)

    def test_override_changes_only_backend_and_isolated_output_paths(self):
        from cwm_h3_inference.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'config.json'
            data = json.loads((ROOT/'third_party/code_world_model/examples/config.example.json').read_text())
            source.write_text(json.dumps(data))
            original_bytes = source.read_bytes()
            base = load_config(source)
            runtime = runner.runtime_config(source, check_files=False)
            self.assertEqual(runtime.attention_mode, 'torch')
            self.assertNotEqual(runtime.cache_dir, base.cache_dir)
            self.assertNotEqual(runtime.output, base.output)
            self.assertEqual(runtime.output.name, 'output.a100_sdpa.mp4')
            for field in fields(base):
                if field.name not in {'attention_mode', 'cache_dir', 'output'}:
                    self.assertEqual(getattr(runtime, field.name), getattr(base, field.name))
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_official_config_remains_flash3_only(self):
        from cwm_h3_inference.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'config.json'
            data = json.loads((ROOT/'third_party/code_world_model/examples/config.example.json').read_text())
            data['attention_mode'] = 'torch'
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'flash3'):
                load_config(source)


if __name__ == '__main__':
    unittest.main()
