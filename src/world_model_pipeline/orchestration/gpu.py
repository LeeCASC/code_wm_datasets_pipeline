"""Read-only GPU availability checks shared by both queues."""
import subprocess


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return [dict(zip(['index', 'free_mib', 'utilization'], map(int, line.split(','))))
            for line in result.stdout.splitlines() if line.strip()]


def eligible_gpus(snapshot, occupied, allowed, minimum_free):
    return [row['index'] for row in sorted(snapshot, key=lambda row: -row['free_mib'])
            if row['index'] in allowed and row['index'] not in occupied
            and row['free_mib'] >= minimum_free and row['utilization'] <= 5]
