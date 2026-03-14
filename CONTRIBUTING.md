# Contributing to mlflow-modal-deploy

Thank you for your interest in contributing to mlflow-modal-deploy! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Modal account ([modal.com](https://modal.com))

### Setup

```bash
# Clone the repository
git clone https://github.com/debu-sinha/mlflow-modal-deploy.git
cd mlflow-modal-deploy

# Install dependencies with uv
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"

# Set up Modal authentication
modal setup
```

### Pre-commit Hooks

We use pre-commit for code quality. Install hooks:

```bash
uv run pre-commit install
```

## Code Style

- **Line length**: 120 characters
- **Python version**: 3.10+
- **Linter/Formatter**: Ruff
- **Docstrings**: Google style (only when providing additional context)

Run linting:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=mlflow_modal --cov-report=term-missing

# Run specific test
uv run pytest tests/test_deployment.py::TestConfigValidation -v
```

### Integration Tests

Integration tests require Modal authentication:

```bash
# Set up Modal
modal setup

# Run integration tests
TEST_MODAL_INTEGRATION=1 uv run pytest tests/ -v -m integration
```

### Writing Tests

- Tests should **fail** if dependencies are missing, not skip
- Use minimal mocking - only mock actual external API calls
- Use real library classes where possible
- Use `@pytest.mark.parametrize` for similar test patterns
- Assert specific expected values, not just existence
- **Codegen changes**: Any change to generated Modal app code must pass the `TestGeneratedCodeSyntax` suite, which runs `ast.parse()` on all generated code variants to catch indentation and syntax errors that string assertions miss

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with DCO sign-off (`git commit -s -m "Add feature"`)
6. Push and create a Pull Request

### Commit Messages

- Use clear, descriptive commit messages
- Sign off commits for DCO compliance: `git commit -s`
- Reference related issues: `Fixes #123`

### PR Requirements

- All tests pass
- Linting passes
- Code coverage maintained or improved
- Documentation updated if needed

## Issue Guidelines

### Bug Reports

Include:
- MLflow and mlflow-modal-deploy versions
- Modal SDK version
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered

## Release Process

Releases are automated via GitHub Actions with trusted publishing to PyPI.

### Pre-Release Checklist

Before creating a release, verify:

- [ ] All tests pass locally: `uv run pytest tests/ -v`
- [ ] Pre-commit hooks pass: `uv run pre-commit run --all-files`
- [ ] Version bumped in both files:
  - `pyproject.toml`
  - `src/mlflow_modal/__init__.py`
- [ ] CHANGELOG.md updated with new version section
- [ ] README.md reflects current API and features
- [ ] No hardcoded version strings in tests

### Release Steps

```bash
# 1. Run pre-release validation
./scripts/pre-release.sh

# 2. Create release branch
git checkout -b release/vX.Y.Z

# 3. Bump version
# Edit pyproject.toml and src/mlflow_modal/__init__.py

# 4. Update CHANGELOG.md
# Add new version section with changes

# 5. Commit and push
git add -A
git commit -s -m "Release vX.Y.Z"
git push origin release/vX.Y.Z

# 6. Create PR and merge to main

# 7. Create and push tag (triggers PyPI publish)
git checkout main
git pull
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Questions?

- Open a GitHub issue for bugs/features
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
