"""
Command-line interface for the D-J project.
"""

import click
import json
from typing import Optional

from .core import DJProject
from .utils import hello_world, get_current_timestamp


@click.group()
@click.version_option(version="0.1.0")
def main():
    """D-J Project CLI - Collaborative project management tool."""
    pass


@main.command()
@click.option("--name", default="World", help="Name to greet")
def hello(name: str):
    """Say hello to someone."""
    click.echo(hello_world(name))


@main.command()
@click.argument("project_name")
@click.option("--collaborators", default="Daniel,Jason", help="Comma-separated list of collaborators")
def create_project(project_name: str, collaborators: str):
    """Create a new D-J project."""
    collab_list = [name.strip() for name in collaborators.split(",")]
    project = DJProject(project_name, collab_list)
    
    click.echo(f"Created project: {project_name}")
    click.echo(f"Collaborators: {', '.join(collab_list)}")
    click.echo(f"Timestamp: {get_current_timestamp()}")


@main.command()
def demo():
    """Run a demonstration of the D-J project functionality."""
    click.echo("D-J Project Demo")
    click.echo("=" * 40)
    
    # Create a demo project
    project = DJProject("Demo Project", ["Daniel", "Jason"])
    
    # Add some demo tasks
    project.add_task("task1", {
        "title": "Setup development environment",
        "description": "Install Python, dependencies, and configure IDE"
    })
    
    project.add_task("task2", {
        "title": "Write core functionality",
        "description": "Implement the main project features"
    })
    
    project.add_task("task3", {
        "title": "Write tests",
        "description": "Create comprehensive test suite"
    })
    
    # Assign tasks
    project.assign_task("task1", "Daniel")
    project.assign_task("task2", "Jason")
    
    # Complete a task
    project.complete_task("task1")
    
    # Show project summary
    summary = project.get_project_summary()
    click.echo("\nProject Summary:")
    click.echo(f"Name: {summary['project_name']}")
    click.echo(f"Collaborators: {', '.join(summary['collaborators'])}")
    click.echo(f"Total tasks: {summary['total_tasks']}")
    click.echo(f"Completed: {summary['completed_tasks']}")
    click.echo(f"Assigned: {summary['assigned_tasks']}")
    click.echo(f"Pending: {summary['pending_tasks']}")
    click.echo(f"Completion rate: {summary['completion_rate']:.1%}")
    
    # Show Daniel's tasks
    daniel_tasks = project.get_tasks_by_collaborator("Daniel")
    click.echo(f"\nDaniel's tasks: {len(daniel_tasks)}")
    for task_id, task_info in daniel_tasks.items():
        click.echo(f"  {task_id}: {task_info['data']['title']} [{task_info['status']}]")
    
    # Show Jason's tasks
    jason_tasks = project.get_tasks_by_collaborator("Jason")
    click.echo(f"\nJason's tasks: {len(jason_tasks)}")
    for task_id, task_info in jason_tasks.items():
        click.echo(f"  {task_id}: {task_info['data']['title']} [{task_info['status']}]")


if __name__ == "__main__":
    main()