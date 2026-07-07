# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SWARM_PARENT = REPO_ROOT if (REPO_ROOT / "swarm").exists() else REPO_ROOT.parent

if str(SWARM_PARENT) not in sys.path:
    sys.path.insert(0, str(SWARM_PARENT))
