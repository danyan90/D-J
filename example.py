#!/usr/bin/env python3
"""
Example usage of the D-J project management system.

This script demonstrates how to use the DJProject class for
collaborative project management between two developers.
"""

from dj_project import DJProject, hello_world
from dj_project.utils import format_collaborator_list, get_current_timestamp


def main():
    """Run the example demonstration."""
    print("=" * 60)
    print("D-J Project Management System - Example Usage")
    print("=" * 60)
    
    # Say hello
    print(f"\n{hello_world('Everyone')}")
    print(f"Current time: {get_current_timestamp()}")
    
    # Create a new project
    print("\n🚀 Creating a new project...")
    project = DJProject("Web Application Development")
    
    # Add some realistic tasks
    tasks = [
        {
            "id": "setup-env",
            "data": {
                "title": "Setup Development Environment",
                "description": "Install Python, Node.js, set up virtual environment, install dependencies"
            }
        },
        {
            "id": "design-api",
            "data": {
                "title": "Design REST API",
                "description": "Define API endpoints, data models, and authentication strategy"
            }
        },
        {
            "id": "implement-backend",
            "data": {
                "title": "Implement Backend",
                "description": "Create Flask/Django application with database integration"
            }
        },
        {
            "id": "create-frontend",
            "data": {
                "title": "Create Frontend",
                "description": "Build React/Vue frontend application with responsive design"
            }
        },
        {
            "id": "write-tests",
            "data": {
                "title": "Write Tests",
                "description": "Create unit tests, integration tests, and end-to-end tests"
            }
        },
        {
            "id": "deploy-app",
            "data": {
                "title": "Deploy Application",
                "description": "Set up CI/CD pipeline and deploy to production"
            }
        }
    ]
    
    # Add all tasks
    print("\n📋 Adding tasks to the project...")
    for task in tasks:
        project.add_task(task["id"], task["data"])
        print(f"  ✓ Added: {task['data']['title']}")
    
    # Assign tasks to collaborators
    print("\n👥 Assigning tasks to collaborators...")
    assignments = [
        ("setup-env", "Daniel"),
        ("design-api", "Jason"),
        ("implement-backend", "Daniel"),
        ("create-frontend", "Jason"),
        ("write-tests", "Daniel"),
        ("deploy-app", "Jason")
    ]
    
    for task_id, collaborator in assignments:
        if project.assign_task(task_id, collaborator):
            task_title = project.tasks[task_id]["data"]["title"]
            print(f"  ✓ Assigned '{task_title}' to {collaborator}")
    
    # Simulate some work being completed
    print("\n✅ Completing some tasks...")
    completed_tasks = ["setup-env", "design-api", "implement-backend"]
    for task_id in completed_tasks:
        if project.complete_task(task_id):
            task_title = project.tasks[task_id]["data"]["title"]
            print(f"  ✓ Completed: {task_title}")
    
    # Show project summary
    print("\n📊 Project Summary:")
    summary = project.get_project_summary()
    print(f"  Project Name: {summary['project_name']}")
    print(f"  Collaborators: {format_collaborator_list(summary['collaborators'])}")
    print(f"  Total Tasks: {summary['total_tasks']}")
    print(f"  Completed: {summary['completed_tasks']}")
    print(f"  In Progress: {summary['assigned_tasks']}")
    print(f"  Pending: {summary['pending_tasks']}")
    print(f"  Completion Rate: {summary['completion_rate']:.1%}")
    
    # Show individual workloads
    print("\n👤 Individual Workloads:")
    for collaborator in project.collaborators:
        tasks_assigned = project.get_tasks_by_collaborator(collaborator)
        print(f"\n  {collaborator}'s Tasks ({len(tasks_assigned)} total):")
        
        for task_id, task_info in tasks_assigned.items():
            status_emoji = {
                "completed": "✅",
                "assigned": "🔄",
                "pending": "⏳"
            }
            emoji = status_emoji.get(task_info["status"], "❓")
            print(f"    {emoji} {task_info['data']['title']} [{task_info['status']}]")
    
    # Export project data
    print("\n💾 Exporting project data...")
    json_data = project.export_to_json()
    print("  ✓ Project data exported to JSON format")
    print(f"  📏 JSON size: {len(json_data)} characters")
    
    print("\n" + "=" * 60)
    print("🎉 Example completed successfully!")
    print("💡 Try running 'dj-cli demo' for an interactive demonstration")
    print("=" * 60)


if __name__ == "__main__":
    main()