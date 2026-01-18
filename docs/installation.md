# Installation Guide

## System Requirements

- **OS**: macOS, Linux, or Windows
- **Python**: Version 3.10 or newer

## Installing via pip

The easiest way to install Iris Agent is via `pip` from PyPI:

```bash
pip install iris-agent
```

## Installing from Source

If you want the latest development version or want to contribute:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mrgehlot/iris-agent.git
    cd iris-agent
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install in editable mode**:
    ```bash
    pip install -e .
    ```

## Installing Development Dependencies

If you plan to run tests or build documentation:

```bash
pip install -e ".[dev]"
```

This installs tools like `pytest`, `black`, `isort`, and `mkdocs`.

## Verifying Installation

To check if the installation was successful, run a quick import in Python:

```bash
python -c "import iris_agent; print(f'Iris Agent {iris_agent.__file__} installed successfully')"
```

## Troubleshooting Installation

**"Module not found" error**:
Ensure your virtual environment is activated.

**Asyncio errors**:
Iris Agent relies heavily on `asyncio`. Ensure you are not trying to nest `asyncio.run()` calls if you are using the synchronous `Agent` inside an already running event loop (e.g., in a Jupyter Notebook). For those cases, use `AsyncAgent`.
