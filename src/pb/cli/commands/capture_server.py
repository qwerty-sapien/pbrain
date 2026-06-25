# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Capture server CLI -- local HTTP server for browser bookmarklet capture (D-10)."""
from __future__ import annotations

import typer


app = typer.Typer(no_args_is_help=True)


@app.command("start")
def server_start(
    port: int = typer.Option(9421, "--port", "-p", help="Port to listen on"),
):
    """Start the local capture server for browser bookmarklet.

    The server listens on localhost only (127.0.0.1) and accepts POST
    requests from the bookmarklet to save web captures to your vault inbox.

    Press Ctrl+C to stop.
    """
    from pb.capture.server import CaptureServer

    server = CaptureServer(port=port)
    typer.echo(f"Capture server listening on http://127.0.0.1:{port}/capture")
    typer.echo("Press Ctrl+C to stop.")
    typer.echo("")
    typer.echo(f"Tip: Run 'pb capture-server bookmarklet --port {port}' to get the bookmarklet code.")
    server.start()  # blocking


@app.command("bookmarklet")
def server_bookmarklet(
    port: int = typer.Option(9421, "--port", "-p", help="Port the server listens on"),
):
    """Output the bookmarklet JavaScript for your browser bookmark bar.

    Copy the output and create a new bookmark in your browser.
    Paste the JavaScript as the bookmark URL.
    """
    from pb.capture.server import generate_bookmarklet

    js = generate_bookmarklet(port=port)
    typer.echo("Add this as a bookmark URL in your browser:")
    typer.echo("")
    typer.echo(js)
    typer.echo("")
    typer.echo(f"The bookmarklet will POST captures to http://localhost:{port}/capture")
    typer.echo("Make sure the capture server is running first: pb capture-server start")
