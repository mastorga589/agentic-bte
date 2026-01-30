# Contributing to Agentic BTE

Thank you for your interest in contributing to Agentic BTE! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and inclusive
- Focus on constructive criticism
- Help create a welcoming environment for all contributors

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/agentic-bte.git
   cd agentic-bte
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/agentic-bte.git
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install in development mode**:
   ```bash
   pip install -e ".[dev,test-external,notebooks]"
   ```

3. **Install spaCy models**:
   ```bash
   python -m spacy download en_core_sci_lg
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Making Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** with clear, focused commits:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

3. **Keep your branch up to date**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m external      # Tests requiring external APIs (requires API keys)

# Run with coverage
pytest --cov=agentic_bte --cov-report=html

# Run specific test file
pytest tests/unit/test_entities.py -v
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place benchmark tests in `tests/benchmarks/`
- Use pytest fixtures from `tests/conftest.py`
- Mark tests that require external APIs with `@pytest.mark.external`

Example:
```python
import pytest

@pytest.mark.unit
def test_entity_extraction():
    # Your test code here
    pass

@pytest.mark.external
def test_bte_api_call():
    # Test requiring BTE API
    pass
```

## Code Style

### Python Style Guide

- **PEP 8**: Follow Python's PEP 8 style guide
- **Line length**: Maximum 100 characters (configured in black)
- **Type hints**: Use type hints for function parameters and return values
- **Docstrings**: Use Google-style docstrings

### Formatting Tools

```bash
# Format code with black
black agentic_bte/

# Sort imports with isort
isort agentic_bte/

# Type checking with mypy
mypy agentic_bte/

# Linting with flake8
flake8 agentic_bte/
```

### Example

```python
from typing import List, Dict, Optional

def extract_entities(query: str, confidence_threshold: float = 0.7) -> Dict[str, str]:
    """
    Extract biomedical entities from a query string.
    
    Args:
        query: The input query text
        confidence_threshold: Minimum confidence for entity extraction
        
    Returns:
        Dictionary mapping entity text to entity IDs
        
    Raises:
        ValueError: If query is empty or invalid
    """
    if not query:
        raise ValueError("Query cannot be empty")
    
    # Implementation here
    return {}
```

## Pull Request Process

1. **Update documentation** if you've changed APIs or added features

2. **Add tests** for new functionality

3. **Ensure all tests pass**:
   ```bash
   pytest
   ```

4. **Update CHANGELOG.md** with your changes under "Unreleased"

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to any related issues (e.g., "Fixes #123")
   - Screenshots (if applicable for UI changes)

7. **Address review feedback** promptly

### PR Title Format

- `feat: Add new entity linking strategy`
- `fix: Correct TRAPI query building for multi-hop queries`
- `docs: Update installation instructions`
- `test: Add benchmark tests for drug discovery`
- `refactor: Simplify query decomposition logic`
- `chore: Update dependencies`

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, relevant package versions
- **Error messages**: Full error traceback if applicable

Example:
```markdown
**Bug Description**
Entity extraction fails for queries containing parentheses.

**Steps to Reproduce**
1. Run: `extract_entities("What drugs treat Alzheimer's (AD)?")`
2. Observe error

**Expected**
Entities extracted successfully

**Actual**
ValueError: Unmatched parentheses

**Environment**
- Python 3.10.5
- macOS 13.2
- agentic-bte 1.0.0
```

### Feature Requests

For feature requests, please include:

- **Description**: Clear description of the proposed feature
- **Use case**: Why this feature would be useful
- **Alternatives**: Any alternative solutions you've considered
- **Examples**: Code examples or mockups if applicable

## Development Guidelines

### Branch Naming

- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/what-changed` - Documentation updates
- `test/what-tested` - Test additions/improvements
- `refactor/what-refactored` - Code refactoring

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

### Code Organization

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add comments for complex logic
- Organize imports: standard library, third-party, local
- Keep files under 500 lines when possible

## Questions?

- Check the [README.md](README.md) for general information
- Check existing [Issues](https://github.com/your-org/agentic-bte/issues) and [Pull Requests](https://github.com/your-org/agentic-bte/pulls)
- Open a new issue for questions or discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Agentic BTE! 🧬🤖
