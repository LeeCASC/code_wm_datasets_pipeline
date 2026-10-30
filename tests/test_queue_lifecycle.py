import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from world_model_pipeline.orchestration import aircraft, observers


def test_worker_started_by_queue_is_stopped_if_gpu_query_fails(tmp_path):
    work = tmp_path/'job'
    work.mkdir()
    config = work/'config.json'
    config.write_text('{}')
    manifest = {'jobs': [{'id': '01_terminal_day', 'config': str(config), 'output': str(work/'video.mp4')}],
                'allowed_gpus': [4], 'minimum_free_memory_mib': 77000}
    (tmp_path/'suite_manifest.json').write_text(json.dumps(manifest))
    worker = Mock(pid=123)
    worker.poll.return_value = None
    with patch.object(aircraft, 'gpu_snapshot', side_effect=[
        [{'index': 4, 'free_mib': 80000, 'utilization': 0}], RuntimeError('query failed')
    ]), patch.object(aircraft.subprocess, 'Popen', return_value=worker), patch.object(aircraft.time, 'sleep'):
        with pytest.raises(RuntimeError, match='query failed'):
            aircraft.run(tmp_path)
    worker.terminate.assert_called_once()
    worker.wait.assert_called_once_with(timeout=15)
    assert json.loads((tmp_path/'batch_status.json').read_text())['phase'] == 'failed'


def test_observer_queue_resumes_completed_jobs_without_waiting_for_gpu(tmp_path):
    job = tmp_path/'job'
    job.mkdir()
    anchor = job/'anchor.png'; anchor.write_bytes(b'present')
    output = job/'enhanced.mp4'; output.write_bytes(b'present')
    (job/'status.json').write_text(json.dumps({'phase': 'complete'}))
    record = {'id': 'Observer07_seg01', 'observer': 'Observer07', 'config': str(job/'config.json'),
              'anchor': str(anchor), 'output': str(output)}
    (tmp_path/'suite_manifest.json').write_text(json.dumps({'jobs': [record]}))
    report = {'phase': 'complete'}
    with patch.object(observers, 'gpu_snapshot', side_effect=AssertionError('Should not query GPU')), \
         patch.object(observers, 'export', return_value=report) as export:
        observers.run(tmp_path)
    assert export.call_args.args[2][record['id']]['phase'] == 'complete'
    assert json.loads((tmp_path/'batch_status.json').read_text())['phase'] == 'complete'
