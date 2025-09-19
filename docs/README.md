# D-J Project Documentation

Welcome to the D-J project documentation!

## Table of Contents

- [Getting Started](getting_started.md)
- [API Reference](api_reference.md)
- [Development Guide](development.md)
- [Collaboration Guidelines](collaboration.md)

## Quick Start

```python
from dj_project import DJProject, hello_world

# Create a new project
project = DJProject("My Project")

# Add a task
project.add_task("task1", {
    "title": "Setup environment",
    "description": "Install dependencies and configure development environment"
})

# Assign task to a collaborator
project.assign_task("task1", "Daniel")

# Get project summary
summary = project.get_project_summary()
print(f"Project: {summary['project_name']}")
print(f"Completion rate: {summary['completion_rate']:.1%}")
```

## CLI Usage

```bash
# Install the package
pip install -e .

# Say hello
dj-cli hello --name "Daniel"

# Create a project
dj-cli create-project "My New Project" --collaborators "Daniel,Jason"

# Run a demo
dj-cli demo
```