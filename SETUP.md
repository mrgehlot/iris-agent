# Setup and Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip (latest version recommended)
- setuptools and wheel

## Installation

### 1. Install in Development Mode

```bash
# Navigate to the package directory
cd iris-agent-framework

# Install in editable mode
pip install -e .

```

### 2. Verify Installation

```bash
# Run the verification script
python verify_setup.py

# Or test imports manually
python -c "from iris_agent import Agent, AsyncAgent; print('✅ Installation successful!')"
```

### 3. Run Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run only unit tests
pytest -m "not integration"

# Run with coverage
pytest --cov=iris_agent --cov-report=html
```

## Building Distribution Packages

### Build wheel and source distribution

```bash
# Install build tools
pip install build twine

# Build packages
python -m build

# This creates:
# - dist/iris-agent-0.1.0.tar.gz (source distribution)
# - dist/iris-agent-0.1.0-py3-none-any.whl (wheel)
```

### Test the built package

```bash
# Install from the built wheel
pip install dist/iris-agent-0.1.0-py3-none-any.whl

# Or from source distribution
pip install dist/iris-agent-0.1.0.tar.gz
```

## Publishing to PyPI

### Test on TestPyPI first

```bash
# Build the package
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ iris-agent
```

### Publish to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# After publishing, install with:
pip install iris-agent
```

## Troubleshooting

### Issue: setuptools/distutils errors

If you encounter setuptools errors, try:

```bash
# Upgrade setuptools
pip install --upgrade setuptools wheel

# Or install with no build isolation
pip install -e . --no-build-isolation
```

### Issue: Import errors after installation

Make sure you're using the correct Python environment:

```bash
# Check Python version
python --version  # Should be 3.10+

# Check installation
pip show iris-agent

# Verify imports
python -c "import iris_agent; print(iris_agent.__file__)"
```

## Development Workflow

1. **Make changes** to the code
2. **Run tests**: `pytest`
3. **Format code**: `black . && isort .`
4. **Update version** in `pyproject.toml` if needed
5. **Build package**: `python -m build`
6. **Test locally**: `pip install dist/iris-agent-*.whl`
7. **Commit and push** changes

## Package Structure

```
iris-agent-framework/
├── src/
│   └── iris_agent/
│       ├── __init__.py
│       ├── agent.py
│       ├── async_agent.py
│       ├── llm.py
│       ├── messages.py
│       ├── prompts.py
│       ├── tools.py
│       ├── types.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── test_basic.py
│   └── test_integration.py
├── examples/
│   └── system_prompt_example.py
├── pyproject.toml
├── README.md
├── LICENSE
├── MANIFEST.in
├── .gitignore
└── verify_setup.py
```

## Next Steps

After setting up:

1. Update `pyproject.toml` with your GitHub repository URLs
2. Add your name/email to authors in `pyproject.toml`
3. Create a GitHub repository
4. Push the code
5. Update repository URLs in `pyproject.toml`
6. Build and publish to PyPI
