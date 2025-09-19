"""
Core functionality for the D-J project.
"""

from typing import Any, Dict, Optional
import json


class DJProject:
    """
    Main class for the D-J collaborative project.
    
    This class provides the core functionality for managing
    a collaborative project between Daniel and Jason.
    """
    
    def __init__(self, project_name: str, collaborators: Optional[list] = None):
        """
        Initialize a new DJProject instance.
        
        Args:
            project_name: Name of the project
            collaborators: List of collaborator names
        """
        self.project_name = project_name
        self.collaborators = collaborators or ["Daniel", "Jason"]
        self.tasks: Dict[str, Any] = {}
        
    def add_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """
        Add a new task to the project.
        
        Args:
            task_id: Unique identifier for the task
            task_data: Dictionary containing task information
        """
        self.tasks[task_id] = {
            "data": task_data,
            "assigned_to": None,
            "status": "pending"
        }
        
    def assign_task(self, task_id: str, collaborator: str) -> bool:
        """
        Assign a task to a collaborator.
        
        Args:
            task_id: ID of the task to assign
            collaborator: Name of the collaborator
            
        Returns:
            True if assignment was successful, False otherwise
        """
        if task_id not in self.tasks:
            return False
            
        if collaborator not in self.collaborators:
            return False
            
        self.tasks[task_id]["assigned_to"] = collaborator
        self.tasks[task_id]["status"] = "assigned"
        return True
        
    def complete_task(self, task_id: str) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of the task to complete
            
        Returns:
            True if task was marked as completed, False otherwise
        """
        if task_id not in self.tasks:
            return False
            
        self.tasks[task_id]["status"] = "completed"
        return True
        
    def get_tasks_by_collaborator(self, collaborator: str) -> Dict[str, Any]:
        """
        Get all tasks assigned to a specific collaborator.
        
        Args:
            collaborator: Name of the collaborator
            
        Returns:
            Dictionary of tasks assigned to the collaborator
        """
        return {
            task_id: task_info
            for task_id, task_info in self.tasks.items()
            if task_info["assigned_to"] == collaborator
        }
        
    def get_project_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the project status.
        
        Returns:
            Dictionary containing project summary information
        """
        total_tasks = len(self.tasks)
        completed_tasks = len([
            task for task in self.tasks.values()
            if task["status"] == "completed"
        ])
        pending_tasks = len([
            task for task in self.tasks.values()
            if task["status"] == "pending"
        ])
        assigned_tasks = len([
            task for task in self.tasks.values()
            if task["status"] == "assigned"
        ])
        
        return {
            "project_name": self.project_name,
            "collaborators": self.collaborators,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "assigned_tasks": assigned_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0
        }
        
    def export_to_json(self) -> str:
        """
        Export project data to JSON format.
        
        Returns:
            JSON string representation of the project
        """
        return json.dumps({
            "project_name": self.project_name,
            "collaborators": self.collaborators,
            "tasks": self.tasks
        }, indent=2)