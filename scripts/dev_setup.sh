#!/bin/bash
set -e

echo "Setting up pbrain development environment..."

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Installing dependencies..."
uv sync

echo "Installing dev dependencies..."
uv sync --extra dev

echo "Initializing database..."
mkdir -p ~/.local/share/pbrain
mkdir -p ~/.local/state/pbrain
mkdir -p ~/.config/pbrain

if [ ! -f ~/.config/pbrain/config.toml ]; then
    echo "Creating default config..."
    cat > ~/.config/pbrain/config.toml << 'EOF'
[general]
vault_path = "~/Documents/pbrain-vault"
verbose = false

[storage]
data_dir = "~/.local/share/pbrain"
log_dir = "~/.local/state/pbrain"

[adapters]
taskwarrior_enabled = false
timewarrior_enabled = false
EOF
    echo "Config created at ~/.config/pbrain/config.toml"
    echo "Edit vault_path to point to your Obsidian vault."
fi

echo "Running tests..."
uv run pytest

echo ""
echo "Setup complete!"
echo "Run 'uv run pb --help' to get started."
