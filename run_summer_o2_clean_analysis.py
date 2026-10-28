"""Run the read-only O2 figures with a compatible, isolated Matplotlib Python."""
import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
ANACONDA = Path('C:/ProgramData/anaconda3/python.exe')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run')
    parser.add_argument('--check-env', action='store_true')
    args = parser.parse_args()
    if not ANACONDA.is_file():
        raise FileNotFoundError(ANACONDA)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    if args.check_env:
        command = [str(ANACONDA), '-c',
                   'import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)']
    else:
        if not args.run:
            parser.error('--run is required unless --check-env is used')
        command = [str(ANACONDA), str(ROOT/'analyze_summer_o2.py'),
                   '--run', str(Path(args.run).resolve())]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
