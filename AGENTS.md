# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the Python application entrypoint and runtime modules (`main.py`, UI, rendering, navigation, input, ZMQ bus helpers, config). `assets/` stores GLB models used by the Panda3D viewer. `octomap_raycast_service/` contains a separate C++ helper service (`main.cpp`) plus its `CMakeLists.txt` and sample map (`iss.bt`).

Top-level files of note: `pyproject.toml` (Python dependencies), `uv.lock` (locked versions), and `imgui.ini` (local ImGui UI state).

## Build, Test, and Development Commands
Install Python dependencies with `uv sync` (Python 3.12+).

Run the app:
```bash
uv run src/main.py
```

Build the OctoMap raycast service (required for occlusion queries):
```bash
cmake -S octomap_raycast_service -B octomap_raycast_service/build
cmake --build octomap_raycast_service/build
```

If you are iterating on Python modules only, rerun `uv run src/main.py` after changes. If the C++ service changes, rebuild before launching.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints on public methods, concise docstrings, and `snake_case` for functions/variables/modules. Classes use `PascalCase` (for example, `SpacebotLinkApp`, `InputController`).

Keep config constants in `src/config.py` uppercase (`OCTOMAP_QUERY_PERIOD_S`). Prefer small, focused modules over adding more logic to `src/main.py`.

For C++, keep C++17 compatibility (per `octomap_raycast_service/CMakeLists.txt`) and match the current straightforward style.

## Testing Guidelines
There is no automated test suite configured yet. For now, validate changes by:
- launching `uv run src/main.py`
- exercising keyboard/gamepad input and UI overlays
- confirming OctoMap service startup and query behavior after C++ changes

When adding tests, place them under `tests/` and use `test_<module>.py` naming.

## Commit & Pull Request Guidelines
Recent commits use short, direct subjects (often lowercase), e.g. `fixed preview for orientation`. Keep messages focused on one change.

PRs should include:
- a brief description of behavior changes
- setup/runtime notes (especially endpoint or asset changes)
- screenshots or short recordings for UI/visual updates
- linked issue/task when applicable
