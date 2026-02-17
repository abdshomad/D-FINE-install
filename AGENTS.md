# Agent Instructions

## Git Submodules

**⚠️ CRITICAL: NEVER UPDATE THE CONTENTS OF GIT SUBMODULES**

- **DO NOT** edit, modify, or commit changes to files within submodule directories (e.g., `D-FINE/`, `chicken-detection-labelme-format/`)
- **DO NOT** run git commands inside submodule directories
- **DO NOT** update submodule dependencies or configurations
- Submodules are external repositories and should remain unchanged
- If you need to modify submodule code, do so in a separate branch or fork, NOT in this repository
- Only update submodule references (the commit hash) if explicitly requested by the user

## Python Environment

**Always use `uv venv` when running Python scripts. Use `uv add` and `uv sync` to manage dependencies.**

### Setup Instructions

1. **Create virtual environment using uv:**
   ```bash
   uv venv
   ```

2. **Sync dependencies from pyproject.toml:**
   ```bash
   uv sync
   ```
   This will create the venv (if it doesn't exist) and install all dependencies from `pyproject.toml`.

3. **Activate the virtual environment (optional, uv can run commands directly):**
   ```bash
   source .venv/bin/activate
   ```
   (On Windows: `.venv\Scripts\activate`)

4. **Run Python scripts:**
   ```bash
   python script.py
   ```
   Or use uv to run directly without activation:
   ```bash
   uv run python script.py
   ```

### Dependency Management

**Always use `uv add` and `uv sync` to manage dependencies:**

1. **Adding a new dependency:**
   ```bash
   uv add <package-name>
   ```
   Example:
   ```bash
   uv add numpy
   uv add "pandas>=2.0.0"
   ```

2. **Adding a development dependency:**
   ```bash
   uv add --dev <package-name>
   ```

3. **Syncing dependencies:**
   ```bash
   uv sync
   ```
   This ensures the virtual environment matches `pyproject.toml` exactly.

4. **Removing a dependency:**
   ```bash
   uv remove <package-name>
   ```

### Important Notes

- **⚠️ NEVER UPDATE THE CONTENTS OF GIT SUBMODULES** - Submodules are read-only external repositories
- **Never run Python scripts without ensuring dependencies are synced with `uv sync`**
- Always use `uv add` to add new packages (it updates `pyproject.toml` automatically)
- Use `uv sync` to install/update dependencies from `pyproject.toml`
- Never manually edit `pyproject.toml` dependencies - use `uv add`/`uv remove` commands
- When executing Python commands, you can use `uv run python script.py` which automatically uses the venv
- The virtual environment is managed by `pyproject.toml` - always sync before running scripts

### Example Workflow

```bash
# 1. Initial setup - create venv and sync dependencies
uv sync

# 2. Add a new dependency (if needed)
uv add some-package

# 3. Run your script (uv will use the venv automatically)
uv run python train-dfine-s.py --use-amp

# Or activate venv and run normally
source .venv/bin/activate
python train-dfine-s.py --use-amp
```

### Updating Dependencies

When `pyproject.toml` changes (e.g., after pulling updates):
```bash
uv sync
```

This will update the virtual environment to match the current `pyproject.toml`.
