# D-J Project
**A Collaborative Python Project Template**

Created by Daniel + Jason for effective 2-person development collaboration.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/danyan90/D-J.git
cd D-J

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Basic Usage

```python
from dj_project import DJProject, hello_world

# Say hello
print(hello_world("World"))

# Create a new collaborative project
project = DJProject("My Awesome Project")

# Add some tasks
project.add_task("setup", {
    "title": "Project Setup",
    "description": "Initialize the development environment"
})

project.add_task("feature1", {
    "title": "Implement Feature 1",
    "description": "Add the main functionality"
})

# Assign tasks to collaborators
project.assign_task("setup", "Daniel")
project.assign_task("feature1", "Jason")

# Mark a task as completed
project.complete_task("setup")

# Get project summary
summary = project.get_project_summary()
print(f"Project: {summary['project_name']}")
print(f"Completion rate: {summary['completion_rate']:.1%}")
```

### Command Line Interface

```bash
# Say hello
dj-cli hello --name "Daniel"

# Create a new project
dj-cli create-project "My Project" --collaborators "Daniel,Jason"

# Run a demo to see all features
dj-cli demo
```

## 📁 Project Structure

```
D-J/
├── src/dj_project/           # Main source code
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Core project management classes
│   ├── utils.py             # Utility functions
│   └── cli.py               # Command-line interface
├── tests/                   # Test suite
│   ├── test_core.py         # Tests for core functionality
│   └── test_utils.py        # Tests for utilities
├── docs/                    # Documentation
│   ├── README.md            # Documentation overview
│   └── collaboration.md     # Collaboration guidelines
├── pyproject.toml           # Project configuration
├── requirements.txt         # Core dependencies
├── requirements-dev.txt     # Development dependencies
├── .pre-commit-config.yaml  # Code quality hooks
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=dj_project

# Run specific test file
pytest tests/test_core.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Using Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📋 Features

### Core Functionality
- **Project Management**: Create and manage collaborative projects
- **Task Assignment**: Assign tasks to team members
- **Progress Tracking**: Monitor completion rates and project status
- **Data Export**: Export project data to JSON format

### Development Tools
- **Modern Python Setup**: Uses pyproject.toml for configuration
- **Code Quality**: Black, isort, flake8, mypy for code quality
- **Testing**: Comprehensive test suite with pytest
- **Pre-commit Hooks**: Automated code quality checks
- **CLI Interface**: Command-line tools for project management

### Collaboration Features
- **Two-Person Workflow**: Optimized for pair programming and collaboration
- **Documentation**: Comprehensive guides for effective teamwork
- **Git Workflow**: Structured branching and review process
- **Task Management**: Built-in system for tracking work

## 🤝 Collaboration Guidelines

This project is designed for effective collaboration between two developers. See [docs/collaboration.md](docs/collaboration.md) for detailed guidelines including:

- Git workflow and branching strategy
- Code review process
- Task assignment and tracking
- Communication protocols
- Quality standards
- Conflict resolution

## 📚 Documentation

- [Getting Started Guide](docs/README.md)
- [Collaboration Guidelines](docs/collaboration.md)
- [API Documentation](#) (Coming soon)

## 🧪 Testing

The project includes comprehensive tests covering:
- Core functionality (project management, task assignment)
- Utility functions (formatting, validation, statistics)
- CLI interface
- Edge cases and error handling

Run tests with: `pytest`

## 🔧 Configuration

### Project Configuration (pyproject.toml)
- Package metadata and dependencies
- Development tool settings (black, isort, mypy, pytest)
- Build system configuration

### Code Quality Settings
- **Black**: Line length 88, Python 3.8+ target
- **isort**: Black-compatible import sorting
- **Flake8**: PEP 8 compliance checking
- **MyPy**: Static type checking with strict settings

## 📈 Future Enhancements

- [ ] Web interface for project management
- [ ] Integration with GitHub Issues/Projects
- [ ] Time tracking functionality
- [ ] Reporting and analytics
- [ ] Team communication features
- [ ] Mobile app support

## 🤝 Contributing

This template is designed for Daniel and Jason's collaboration, but contributions and suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Daniel** - [@danyan90](https://github.com/danyan90)
- **Jason** - Collaborator

## 🙏 Acknowledgments

- Python community for excellent tooling
- Open source contributors for inspiration
- All the developers who believe in collaborative coding

---

*Happy Coding! 🐍✨*
