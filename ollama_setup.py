"""Make sure Ollama is running and the embedding model is ready.

Usage:
    from ollama_setup import ensure_ollama
    ensure_ollama("embeddinggemma-2:latest")

Or run directly:
    python ollama_setup.py embeddinggemma-2:latest
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL


def _find_ollama_exe():
    """Return the path to the ollama executable, or None if not found."""
    exe = shutil.which("ollama")
    if exe:
        return exe

    # Default Windows install location (not always on PATH right after install)
    local = os.getenv("LOCALAPPDATA")
    if local:
        candidate = os.path.join(local, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(candidate):
            return candidate
    return None


def _request(path, payload=None, timeout=5):
    url = OLLAMA_URL + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def is_server_running():
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=2).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def start_server(exe, wait_seconds=30):
    """Start `ollama serve` in the background and wait until it answers."""
    print("Starting Ollama server...")
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        # Run without a console window and keep it alive after this script exits
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen([exe, "serve"], **kwargs)

    for _ in range(wait_seconds * 2):
        if is_server_running():
            print("Ollama server is up.")
            return
        time.sleep(0.5)
    raise RuntimeError(f"Ollama did not start within {wait_seconds} seconds.")


def model_installed(model):
    names = [m.get("name", "") for m in _request("/api/tags").get("models", [])]
    wanted = {model, model + ":latest"} if ":" not in model else {model}
    return any(n in wanted for n in names)


def pull_model(exe, model):
    print(f"Downloading model '{model}' (this can take a few minutes the first time)...")
    result = subprocess.run([exe, "pull", model])
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not pull '{model}'. Check the name with `ollama list` "
            f"or try a different embedding model (e.g. 'nomic-embed-text')."
        )


def warm_up(model):
    """Load the model into memory so the first request is fast. Failure is not fatal."""
    try:
        _request("/api/embed", {"model": model, "input": "warm up"}, timeout=120)
        print(f"Model '{model}' loaded and ready.")
    except Exception as e:
        print(f"Warm-up skipped: {e}")


def ensure_ollama(model):
    exe = _find_ollama_exe()

    if not is_server_running():
        if exe is None:
            raise RuntimeError(
                "Ollama is not installed. Download it from https://ollama.com/download "
                "and run this again."
            )
        start_server(exe)
    else:
        print("Ollama server already running.")

    if not model_installed(model):
        if exe is None:
            raise RuntimeError(f"Model '{model}' is missing and the ollama executable was not found.")
        pull_model(exe, model)
    else:
        print(f"Model '{model}' already downloaded.")

    warm_up(model)


if __name__ == "__main__":
    ensure_ollama(sys.argv[1] if len(sys.argv) > 1 else "embeddinggemma-2:latest")