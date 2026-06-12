# Contributing to SenseVoice Multimodal Emotion

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install dependencies

```bash
git clone https://github.com/your-username/sensevoice-multimodal-emotion.git
cd sensevoice-multimodal-emotion
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install -e .
```

## Development

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all public functions

### Project Structure

```
sensevoice-multimodal-emotion/
├── sensevoice_emotion/
│   ├── __init__.py
│   ├── __main__.py
│   └── sensevoice_emotion.py
├── tests/
│   └── test_emotion.py
├── examples/
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

## Submitting Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Add tests for new functionality
4. Run the test suite
5. Commit with a clear message
6. Push to your fork
7. Submit a pull request

### Commit Messages

Use conventional commits:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

Example:
```
feat: add Whisper large-v3 model support

- Add model selection parameter
- Update requirements.txt
- Add comparison test
```

## Reporting Issues

When reporting issues, please include:

1. Python version
2. Operating system
3. Steps to reproduce
4. Expected behavior
5. Actual behavior
6. Error messages (if any)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
