"""Execute a fixed P45 engineering shard; module bindings belong to its freeze."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    root, rank = Path(sys.argv[1]), int(sys.argv[2])
    frozen = json.loads((root / 'FREEZE.json').read_text())
    assert frozen['status'] == 'frozen' and frozen['root_authorized']
    assert frozen['steps'] == 240 and 0 <= rank < 4
    for index, cell in enumerate(frozen['cells']):
        if index % 4 != rank:
            continue
        for path, digest in frozen['sources'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        module = frozen['runners'][cell['task']]
        command = [sys.executable, '-m', module, '--seed', str(cell['seed']),
                   '--split', cell['split'], '--steps', str(frozen['steps']), '--out', cell['out']]
        if cell['task'] != 'slider':
            command += ['--task', cell['task']]
        result = subprocess.run(command)
        print(json.dumps(dict(cell=index, returncode=result.returncode)), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    (root / f'WORKER_{rank}_DONE.json').write_text(json.dumps(dict(rank=rank, completed=True)))


if __name__ == '__main__':
    main()
