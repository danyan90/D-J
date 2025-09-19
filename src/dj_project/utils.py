"""
Utility functions for the D-J project.
"""

from datetime import datetime
from typing import Any, Dict, List


def hello_world(name: str = "World") -> str:
    """
    Return a greeting message.
    
    Args:
        name: Name to greet
        
    Returns:
        Greeting message
    """
    return f"Hello, {name}! Welcome to the D-J collaborative project."


def get_current_timestamp() -> str:
    """
    Get the current timestamp in ISO format.
    
    Returns:
        Current timestamp as ISO formatted string
    """
    return datetime.now().isoformat()


def format_collaborator_list(collaborators: List[str]) -> str:
    """
    Format a list of collaborators into a readable string.
    
    Args:
        collaborators: List of collaborator names
        
    Returns:
        Formatted string of collaborators
    """
    if not collaborators:
        return "No collaborators"
    elif len(collaborators) == 1:
        return collaborators[0]
    elif len(collaborators) == 2:
        return f"{collaborators[0]} and {collaborators[1]}"
    else:
        return f"{', '.join(collaborators[:-1])}, and {collaborators[-1]}"


def validate_task_data(task_data: Dict[str, Any]) -> bool:
    """
    Validate task data structure.
    
    Args:
        task_data: Dictionary containing task information
        
    Returns:
        True if task data is valid, False otherwise
    """
    required_fields = ["title", "description"]
    
    if not isinstance(task_data, dict):
        return False
        
    for field in required_fields:
        if field not in task_data:
            return False
        if not isinstance(task_data[field], str):
            return False
        if not task_data[field].strip():
            return False
            
    return True


def calculate_project_stats(tasks: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate various statistics for project tasks.
    
    Args:
        tasks: Dictionary of tasks
        
    Returns:
        Dictionary containing calculated statistics
    """
    if not tasks:
        return {
            "completion_rate": 0.0,
            "assignment_rate": 0.0,
            "pending_rate": 0.0
        }
    
    total = len(tasks)
    completed = sum(1 for task in tasks.values() if task.get("status") == "completed")
    assigned = sum(1 for task in tasks.values() if task.get("status") == "assigned")
    pending = sum(1 for task in tasks.values() if task.get("status") == "pending")
    
    return {
        "completion_rate": completed / total,
        "assignment_rate": assigned / total,
        "pending_rate": pending / total
    }