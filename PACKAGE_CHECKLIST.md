# Package Distribution Checklist

Use this checklist to ensure your package is ready for distribution.

## ✅ Package Structure

- [x] `pyproject.toml` with build system configuration
- [x] `src/iris_agent/` package directory
- [x] `__init__.py` with proper exports
- [x] `py.typed` marker file for type checking
- [x] `LICENSE` file (MIT)
- [x] `README.md` with installation instructions
- [x] `.gitignore` file
- [x] `MANIFEST.in` for distribution files

## ✅ Configuration Files

- [x] `pyproject.toml` with:
  - [x] Build system (setuptools)
  - [x] Project metadata (name, version, description)
  - [x] Dependencies (core packages required for runtime)
  - [x] Package discovery configuration
  - [x] Tool configurations (black, isort, pytest)

## ✅ Testing

- [x] `tests/` directory
- [x] `tests/__init__.py`
- [x] `tests/test_basic.py` (unit tests)
- [x] `tests/test_integration.py` (integration tests)
- [x] Pytest configuration in `pyproject.toml`

## ✅ Documentation

- [x] `README.md` with:
  - [x] Installation instructions
  - [x] Quick start examples
  - [x] Feature documentation
  - [x] Usage examples
- [x] `QUICKSTART.md` for quick reference
- [x] `SETUP.md` for development setup
- [x] `CONTRIBUTING.md` for contributors
- [x] `examples/` directory with example code

## ✅ Before Publishing

Before publishing to PyPI, make sure to:

1. **Update version** in `pyproject.toml`
   ```toml
   version = "0.1.0"  # Update as needed
   ```

2. **Update repository URLs** in `pyproject.toml`
   ```toml
   [project.urls]
   Homepage = "https://github.com/yourusername/iris-agent"
   Repository = "https://github.com/yourusername/iris-agent"
   ```

3. **Update author information** in `pyproject.toml`
   ```toml
   authors = [
       {name = "Your Name", email = "your.email@example.com"}
   ]
   ```

4. **Test installation locally**
   ```bash
   pip install -e .
   python verify_setup.py
   ```

5. **Run all tests**
   ```bash
   pytest
   ```

6. **Build package**
   ```bash
   python -m build
   ```

7. **Test built package**
   ```bash
   pip install dist/iris-agent-*.whl
   ```

8. **Check package contents**
   ```bash
   tar -tzf dist/iris-agent-*.tar.gz | head -20
   ```

## 📦 Publishing Steps

1. **Create GitHub repository**
   - Initialize git: `git init`
   - Add remote: `git remote add origin https://github.com/yourusername/iris-agent.git`
   - Commit and push

2. **Test on TestPyPI**
   ```bash
   python -m build
   twine upload --repository testpypi dist/*
   ```

3. **Install from TestPyPI to verify**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ iris-agent
   ```

4. **Publish to PyPI**
   ```bash
   twine upload dist/*
   ```

5. **Verify on PyPI**
   - Visit: https://pypi.org/project/iris-agent/
   - Test installation: `pip install iris-agent`

## 🎯 Post-Publishing

- [ ] Update README with PyPI installation instructions
- [ ] Create GitHub releases
- [ ] Add badges to README (build status, version, etc.)
- [ ] Set up CI/CD (GitHub Actions) for automated testing
- [ ] Add code coverage reporting
- [ ] Create documentation site (if needed)

## 📝 Version Bumping

When releasing a new version:

1. Update `version` in `pyproject.toml`
2. Update `CHANGELOG.md` (if you create one)
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. Build and publish

## 🔍 Verification Commands

```bash
# Check package structure
python -c "import iris_agent; print(iris_agent.__file__)"

# Verify all exports
python -c "from iris_agent import *; print(__all__)"

# Run verification script
python verify_setup.py

# Run tests
pytest

# Check package metadata
pip show iris-agent

# Build and check
python -m build
twine check dist/*
```
