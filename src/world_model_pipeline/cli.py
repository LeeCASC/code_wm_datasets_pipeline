"""One CLI for geometry, CWM generation, UE references, and video assembly."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess

from .paths import ROOT
from .runtime import clean_environment, module_command


def gpu_list(value):
    try:
        result = [int(item) for item in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('GPU list must contain comma-separated integers') from exc
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError('GPU IDs must be unique nonnegative integers')
    return result


def show(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--version', action='version', version='0.1.0')
    commands = cli.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Report dependency locations and local model availability')
    models = commands.add_parser('models').add_subparsers(dest='action', required=True)
    verify = models.add_parser('verify')
    verify.add_argument('--model-root', type=Path)
    verify.add_argument('--full', action='store_true', help='Read and SHA-256 every model file')
    proxy = commands.add_parser('proxy').add_subparsers(dest='action', required=True)
    generate = proxy.add_parser('generate')
    generate.add_argument('--scene', type=Path, default=ROOT/'configs/aircraft_scene.json')
    generate.add_argument('--output', type=Path, required=True)
    generate.add_argument('--views', nargs='+', choices=['side', 'rear', 'fixed'], default=['side', 'rear', 'fixed'])
    audit = proxy.add_parser('audit')
    audit.add_argument('--directory', type=Path, required=True)
    obb = commands.add_parser('obb')
    obb.add_argument('--source', type=Path, required=True)
    obb.add_argument('--output', type=Path, required=True)
    aircraft = commands.add_parser('aircraft').add_subparsers(dest='action', required=True)
    for action in ('prepare', 'prepare-suite'):
        p = aircraft.add_parser(action)
        p.add_argument('--proxy', type=Path, default=ROOT/'outputs/aircraft_proxy')
        p.add_argument('--anchor', type=Path, default=ROOT/'assets/examples/aircraft/anchor_photoreal.png')
        p.add_argument('--model-root', type=Path)
        p.add_argument('--output', type=Path, required=True)
        if action == 'prepare':
            p.add_argument('--prompt', help='A custom English scene/motion caption')
        else:
            p.add_argument('--reuse-from', type=Path, help='Optional compatible existing CWM config; reuse visual caches only')
            p.add_argument('--gpus', type=gpu_list, default=[4, 5, 6, 7])
            p.add_argument('--minimum-free-mib', type=int, default=77000)
    run = aircraft.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--gpu', type=int, required=True)
    suite = aircraft.add_parser('run-suite')
    suite.add_argument('--directory', type=Path, required=True)
    cwm = commands.add_parser('cwm')
    cwm.add_argument('action', choices=['validate', 'prepare', 'generate'])
    cwm.add_argument('--config', type=Path, required=True)
    cwm.add_argument('--gpu', type=int)
    cwm.add_argument('--crf', type=int, default=18)
    ue = commands.add_parser('ue').add_subparsers(dest='action', required=True)
    prep = ue.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True, help='Directory containing Observer*/rgb/rgb.mp4 and metadata')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--anchors-dir', type=Path, default=ROOT/'assets/examples/observers')
    prep.add_argument('--model-root', type=Path)
    prep.add_argument('--observers', nargs='+', default=['Observer07_OrbitRiseSlow', 'Observer08_OrbitFallSlow'])
    prep.add_argument('--gpus', type=gpu_list, default=[4, 5, 6, 7])
    prep.add_argument('--minimum-free-mib', type=int, default=77000)
    run = ue.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--gpu', type=int, required=True)
    suite = ue.add_parser('run-suite')
    suite.add_argument('--directory', type=Path, required=True)
    video = commands.add_parser('video').add_subparsers(dest='action', required=True)
    concat = video.add_parser('concat')
    concat.add_argument('--inputs', type=Path, nargs='+', required=True)
    concat.add_argument('--output', type=Path, required=True)
    return cli


def doctor():
    from .models import model_paths
    dependencies = {}
    for module, distribution in [('numpy', 'numpy'), ('av', 'av'), ('torch', 'torch'),
                                 ('transformers', 'transformers'), ('imageio_ffmpeg', 'imageio-ffmpeg'),
                                 ('cwm_h3_inference', 'cwm-minimax-h3-inference'), ('musubi_tuner', 'musubi-tuner')]:
        spec = importlib.util.find_spec(module)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = 'vendored source' if spec else None
        dependencies[module] = {'available': spec is not None, 'version': version,
                                'source': spec.origin if spec else None}
    return {'root': str(ROOT), 'dependencies': dependencies,
            'models': {name: {'path': path, 'exists': Path(path).exists()} for name, path in model_paths().items()},
            'neural_camera_gt_certified': False}


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == 'doctor':
        show(doctor())
    elif args.command == 'models':
        from .models import verify_models
        report = verify_models(args.model_root, full=args.full)
        show(report)
        return 0 if report['status'] == 'pass' else 1
    elif args.command == 'proxy':
        from .geometry.aircraft import Scene, generate, audit
        if args.action == 'generate':
            show(generate(args.output, Scene(**json.loads(args.scene.read_text())), args.views))
        else:
            show(audit(args.directory.resolve()))
    elif args.command == 'obb':
        from .geometry.obb import generate
        show(generate(args.source, args.output))
    elif args.command == 'aircraft':
        if args.action in ('prepare', 'prepare-suite'):
            from .workflows.aircraft import prepare, prepare_single
            if args.action == 'prepare':
                show(prepare_single(args.output, args.proxy, args.anchor, prompt=args.prompt, model_root=args.model_root))
            else:
                show(prepare(args.output, args.proxy, args.anchor, model_root=args.model_root,
                             reuse_from=args.reuse_from, allowed_gpus=args.gpus,
                             minimum_free_memory_mib=args.minimum_free_mib))
        elif args.action == 'run':
            from .orchestration.supervisor import run
            show(run(args.config, args.gpu))
        else:
            return subprocess.call(module_command('orchestration.aircraft', '--directory', args.directory.resolve()),
                                   env=clean_environment(), cwd=ROOT)
    elif args.command == 'cwm':
        if args.action != 'validate' and args.gpu is None:
            cli.error('cwm prepare/generate requires an explicit --gpu')
        return subprocess.call(module_command('inference.cwm', args.action, '--config', args.config.resolve(), '--crf', args.crf),
                               env=clean_environment(args.gpu), cwd=ROOT)
    elif args.command == 'ue':
        if args.action == 'prepare':
            from .workflows.observers import prepare
            show(prepare(args.output, args.source, model_root=args.model_root, anchors_dir=args.anchors_dir,
                         observers=args.observers, allowed_gpus=args.gpus,
                         minimum_free_memory_mib=args.minimum_free_mib))
        elif args.action == 'run':
            return subprocess.call(module_command('inference.rgb', '--config', args.config.resolve()),
                                   env=clean_environment(args.gpu), cwd=ROOT)
        else:
            return subprocess.call(module_command('orchestration.observers', '--directory', args.directory.resolve()),
                                   env=clean_environment(), cwd=ROOT)
    elif args.command == 'video':
        from .video import concatenate
        show(concatenate(args.inputs, args.output))
    return 0
