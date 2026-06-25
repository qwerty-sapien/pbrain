# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Theme registry and active theme loader.

3-tier load order: custom TOML file > per-key overrides > preset > catppuccin fallback.
Per D-09, D-10, D-11, D-12.
"""

from .catppuccin import CATPPUCCIN
from .nord import NORD

PRESETS: dict[str, dict[str, str]] = {
    "catppuccin": CATPPUCCIN,
    "nord": NORD,
}


def load_active_theme() -> dict[str, str]:
    """Return the active theme role dict for Console(theme=Theme(...)).

    Load order per D-12:
    1. Custom TOML file (if path set and file exists)
    2. Per-key overrides merged onto preset
    3. Preset by name
    4. Catppuccin fallback
    """
    try:
        from pb.storage.config import get_config
        config = get_config()
        ui = getattr(config, "ui", None)
    except Exception:
        return dict(CATPPUCCIN)

    theme_preset = getattr(ui, "theme", "catppuccin") if ui else "catppuccin"
    theme_overrides = getattr(ui, "theme_overrides", {}) if ui else {}
    theme_file = getattr(ui, "theme_file", "") if ui else ""

    if theme_file:
        import tomli
        from pathlib import Path

        custom = Path(theme_file).expanduser()
        if custom.exists():
            try:
                with open(custom, "rb") as f:
                    return tomli.load(f)
            except Exception:
                # T-16-03: Fall back to catppuccin on malformed TOML
                pass

    base = dict(PRESETS.get(theme_preset, CATPPUCCIN))
    base.update(theme_overrides)
    return base
