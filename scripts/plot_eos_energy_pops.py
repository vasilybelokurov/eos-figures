#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).with_name("plot_eos_figure.py")
    raise SystemExit(subprocess.call([sys.executable, str(script), "eos_energy_pops", *sys.argv[1:]]))

