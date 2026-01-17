# Contributing to Iris Agent

Thank you for your interest in contributing to Iris Agent! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/iris-agent.git
   cd iris-agent
   ```

2. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests**
   ```bash
   pytest
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting: `black .`
- Use `isort` for import sorting: `isort .`
- Maximum line length: 100 characters

## Testing

- Write tests for new features
- Ensure all tests pass: `pytest`
- Run with coverage: `pytest --cov=iris_agent --cov-report=html`
- Integration tests require API keys (set `OPENAI_API_KEY` environment variable)

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add/update tests
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request with a clear description

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` (if exists)
3. Create a git tag
4. Build and publish to PyPI

## Questions?

Open an issue for questions or discussions.
