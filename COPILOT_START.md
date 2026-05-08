# Starting Copilot in this repository

Open this directory in VS Code or start your coding assistant from the repository root.

Recommended first prompt:

```text
Read AGENTS.md, README.md, and the files under src/paired_void_finder. Implement the package incrementally. Start with periodic.py and mocks.py. After each module, run pytest. Do not implement the full project in one pass.
```

Recommended first terminal commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

First implementation milestone:

```text
Implement periodic.py and mocks.py, then make tests/test_periodic.py and tests/test_mock.py pass.
```
