"""
Tests for the core functionality of the D-J project.
"""

import pytest
from dj_project.core import DJProject


class TestDJProject:
    """Test class for DJProject functionality."""
    
    def test_project_initialization(self):
        """Test project initialization."""
        project = DJProject("Test Project")
        assert project.project_name == "Test Project"
        assert project.collaborators == ["Daniel", "Jason"]
        assert project.tasks == {}
        
    def test_project_initialization_with_custom_collaborators(self):
        """Test project initialization with custom collaborators."""
        collaborators = ["Alice", "Bob", "Charlie"]
        project = DJProject("Test Project", collaborators)
        assert project.project_name == "Test Project"
        assert project.collaborators == collaborators
        
    def test_add_task(self):
        """Test adding a task to the project."""
        project = DJProject("Test Project")
        task_data = {"title": "Test Task", "description": "A test task"}
        
        project.add_task("task1", task_data)
        
        assert "task1" in project.tasks
        assert project.tasks["task1"]["data"] == task_data
        assert project.tasks["task1"]["assigned_to"] is None
        assert project.tasks["task1"]["status"] == "pending"
        
    def test_assign_task_success(self):
        """Test successful task assignment."""
        project = DJProject("Test Project")
        task_data = {"title": "Test Task", "description": "A test task"}
        project.add_task("task1", task_data)
        
        result = project.assign_task("task1", "Daniel")
        
        assert result is True
        assert project.tasks["task1"]["assigned_to"] == "Daniel"
        assert project.tasks["task1"]["status"] == "assigned"
        
    def test_assign_task_invalid_task(self):
        """Test task assignment with invalid task ID."""
        project = DJProject("Test Project")
        
        result = project.assign_task("nonexistent", "Daniel")
        
        assert result is False
        
    def test_assign_task_invalid_collaborator(self):
        """Test task assignment with invalid collaborator."""
        project = DJProject("Test Project")
        task_data = {"title": "Test Task", "description": "A test task"}
        project.add_task("task1", task_data)
        
        result = project.assign_task("task1", "Unknown")
        
        assert result is False
        assert project.tasks["task1"]["assigned_to"] is None
        assert project.tasks["task1"]["status"] == "pending"
        
    def test_complete_task_success(self):
        """Test successful task completion."""
        project = DJProject("Test Project")
        task_data = {"title": "Test Task", "description": "A test task"}
        project.add_task("task1", task_data)
        
        result = project.complete_task("task1")
        
        assert result is True
        assert project.tasks["task1"]["status"] == "completed"
        
    def test_complete_task_invalid_task(self):
        """Test task completion with invalid task ID."""
        project = DJProject("Test Project")
        
        result = project.complete_task("nonexistent")
        
        assert result is False
        
    def test_get_tasks_by_collaborator(self):
        """Test getting tasks by collaborator."""
        project = DJProject("Test Project")
        
        # Add and assign tasks
        project.add_task("task1", {"title": "Task 1", "description": "First task"})
        project.add_task("task2", {"title": "Task 2", "description": "Second task"})
        project.add_task("task3", {"title": "Task 3", "description": "Third task"})
        
        project.assign_task("task1", "Daniel")
        project.assign_task("task3", "Daniel")
        project.assign_task("task2", "Jason")
        
        daniel_tasks = project.get_tasks_by_collaborator("Daniel")
        jason_tasks = project.get_tasks_by_collaborator("Jason")
        
        assert len(daniel_tasks) == 2
        assert "task1" in daniel_tasks
        assert "task3" in daniel_tasks
        
        assert len(jason_tasks) == 1
        assert "task2" in jason_tasks
        
    def test_get_project_summary(self):
        """Test getting project summary."""
        project = DJProject("Test Project")
        
        # Add tasks with different statuses
        project.add_task("task1", {"title": "Task 1", "description": "First task"})
        project.add_task("task2", {"title": "Task 2", "description": "Second task"})
        project.add_task("task3", {"title": "Task 3", "description": "Third task"})
        
        project.assign_task("task1", "Daniel")
        project.complete_task("task1")
        project.assign_task("task2", "Jason")
        # task3 remains pending
        
        summary = project.get_project_summary()
        
        assert summary["project_name"] == "Test Project"
        assert summary["collaborators"] == ["Daniel", "Jason"]
        assert summary["total_tasks"] == 3
        assert summary["completed_tasks"] == 1
        assert summary["assigned_tasks"] == 1
        assert summary["pending_tasks"] == 1
        assert summary["completion_rate"] == 1/3
        
    def test_export_to_json(self):
        """Test exporting project to JSON."""
        project = DJProject("Test Project")
        task_data = {"title": "Test Task", "description": "A test task"}
        project.add_task("task1", task_data)
        
        json_str = project.export_to_json()
        
        assert isinstance(json_str, str)
        assert "Test Project" in json_str
        assert "Daniel" in json_str
        assert "Jason" in json_str
        assert "task1" in json_str