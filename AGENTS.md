# Project Rules & Agent Guidelines

## 1. WSL & Path Conventions (Crucial)
- **Linux Native Environment:** You are executing commands inside a WSL2 Linux environment (`x86_64 Linux`).
- **Never use Windows Backslashes:** Always use forward slashes (`/`). Do NOT pass paths like `C:\Users\...` to Linux commands, bash tools, or file APIs.
- **Windows Drives Mount Point:** Windows drives are mounted under `/mnt/` (e.g. `C:\` is `/mnt/c/`).
- **Path Conversions (`wslpath`):**
  - Windows to Linux: `wslpath -u 'C:\path\to\file'` $\rightarrow$ `/mnt/c/path/to/file`
  - Linux to Windows: `wslpath -w '~/bxjudge'` $\rightarrow$ `\\wsl.localhost\Ubuntu\home\<user>\bxjudge`
- **Filesystem Placement:** Always keep the project, virtual environment (`.venv`), git repository, and PyTorch datasets in the native Linux ext4 filesystem (`~/...`) for maximum I/O performance. Never run heavy I/O operations directly on `/mnt/c/`.
- **Line Endings:** Ensure files use `LF` line endings. If encountering `^M: bad interpreter`, run `sed -i 's/\r$//' <file>`.

---

## 2. Python & AI/ML Development Guidelines
- **Modern Package Manager:** This project uses `uv`.
  - Run commands with `uv run <command>` (e.g. `uv run pytest`, `uv run python script.py`).
  - Add packages with `uv add <package>` or `uv add --dev <package>`.
- **Code Quality:**
  - Lint with `uv run ruff check . --fix`.
  - Format with `uv run ruff format .`.
  - Type-safety: Enforce modern Python 3.12+ typing (`list[str]`, `str | None`, `typing.Self`).
- **PyTorch & GPU Optimization:**
  - Leverage CUDA acceleration (WSL2 maps `/usr/lib/wsl/lib/libcuda.so.1`).
  - Use `pin_memory=True` and `persistent_workers=True` in DataLoaders.
  - Employ `torch.amp.autocast('cuda')` and `optimizer.zero_grad(set_to_none=True)`.
  - Detach scalar losses when tracking metrics (`loss.item()`).

---

## 3. Superpowers & Code Review Discipline
- **Systematic Debugging:** Never apply speculative or blind fixes. Always build a tight feedback loop to reproduce the failure (RED), identify the root cause, apply the fix (GREEN), and confirm no regressions.
- **Test-Driven Development (TDD):** Write tests before or alongside code changes (`uv run pytest`).
- **Pre-Commit Verification:** Before completing a feature, check the diff (`git diff`), run linter and tests, and ensure code is simplified and cleanly structured.
