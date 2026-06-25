#!/bin/bash
set -e

echo "Setting up ProductiveBrain development environment..."

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Installing dependencies..."
uv sync

echo "Installing dev dependencies..."
uv sync --extra dev

echo "Initializing database..."
mkdir -p ~/.local/share/productivebrain
mkdir -p ~/.local/state/productivebrain
mkdir -p ~/.config/productivebrain

if [ ! -f ~/.config/productivebrain/config.toml ]; then
    echo "Creating default config..."
    cat > ~/.config/productivebrain/config.toml << 'EOF'
[general]
vault_path = "~/Documents/productivebrain-vault"
verbose = false

[storage]
data_dir = "~/.local/share/productivebrain"
log_dir = "~/.local/state/productivebrain"

[adapters]
taskwarrior_enabled = false
timewarrior_enabled = false
EOF
    echo "Config created at ~/.config/productivebrain/config.toml"
    echo "Edit vault_path to point to your Obsidian vault."
fi

echo "Running tests..."
uv run pytest

echo ""
echo "Setup complete!"
echo "Run 'uv run pb --help' to get started."
