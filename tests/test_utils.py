"""
Tests for the utility functions of the D-J project.
"""

import pytest
from datetime import datetime
from dj_project.utils import (
    hello_world,
    get_current_timestamp,
    format_collaborator_list,
    validate_task_data,
    calculate_project_stats
)


class TestUtils:
    """Test class for utility functions."""
    
    def test_hello_world_default(self):
        """Test hello_world with default parameter."""
        result = hello_world()
        assert result == "Hello, World! Welcome to the D-J collaborative project."
        
    def test_hello_world_custom_name(self):
        """Test hello_world with custom name."""
        result = hello_world("Alice")
        assert result == "Hello, Alice! Welcome to the D-J collaborative project."
        
    def test_get_current_timestamp(self):
        """Test get_current_timestamp returns valid ISO format."""
        timestamp = get_current_timestamp()
        
        # Should be able to parse as ISO format
        parsed = datetime.fromisoformat(timestamp)
        assert isinstance(parsed, datetime)
        
    def test_format_collaborator_list_empty(self):
        """Test formatting empty collaborator list."""
        result = format_collaborator_list([])
        assert result == "No collaborators"
        
    def test_format_collaborator_list_single(self):
        """Test formatting single collaborator."""
        result = format_collaborator_list(["Alice"])
        assert result == "Alice"
        
    def test_format_collaborator_list_two(self):
        """Test formatting two collaborators."""
        result = format_collaborator_list(["Alice", "Bob"])
        assert result == "Alice and Bob"
        
    def test_format_collaborator_list_multiple(self):
        """Test formatting multiple collaborators."""
        result = format_collaborator_list(["Alice", "Bob", "Charlie", "David"])
        assert result == "Alice, Bob, Charlie, and David"
        
    def test_validate_task_data_valid(self):
        """Test validating valid task data."""
        task_data = {
            "title": "Test Task",
            "description": "This is a test task"
        }
        
        result = validate_task_data(task_data)
        assert result is True
        
    def test_validate_task_data_invalid_type(self):
        """Test validating task data with invalid type."""
        result = validate_task_data("not a dictionary")
        assert result is False
        
    def test_validate_task_data_missing_title(self):
        """Test validating task data missing title."""
        task_data = {
            "description": "This is a test task"
        }
        
        result = validate_task_data(task_data)
        assert result is False
        
    def test_validate_task_data_missing_description(self):
        """Test validating task data missing description."""
        task_data = {
            "title": "Test Task"
        }
        
        result = validate_task_data(task_data)
        assert result is False
        
    def test_validate_task_data_empty_title(self):
        """Test validating task data with empty title."""
        task_data = {
            "title": "   ",
            "description": "This is a test task"
        }
        
        result = validate_task_data(task_data)
        assert result is False
        
    def test_validate_task_data_empty_description(self):
        """Test validating task data with empty description."""
        task_data = {
            "title": "Test Task",
            "description": ""
        }
        
        result = validate_task_data(task_data)
        assert result is False
        
    def test_validate_task_data_non_string_fields(self):
        """Test validating task data with non-string fields."""
        task_data = {
            "title": 123,
            "description": "This is a test task"
        }
        
        result = validate_task_data(task_data)
        assert result is False
        
    def test_calculate_project_stats_empty(self):
        """Test calculating stats for empty project."""
        stats = calculate_project_stats({})
        
        assert stats["completion_rate"] == 0.0
        assert stats["assignment_rate"] == 0.0
        assert stats["pending_rate"] == 0.0
        
    def test_calculate_project_stats_mixed(self):
        """Test calculating stats for project with mixed task statuses."""
        tasks = {
            "task1": {"status": "completed"},
            "task2": {"status": "assigned"},
            "task3": {"status": "pending"},
            "task4": {"status": "completed"},
        }
        
        stats = calculate_project_stats(tasks)
        
        assert stats["completion_rate"] == 0.5  # 2 out of 4
        assert stats["assignment_rate"] == 0.25  # 1 out of 4
        assert stats["pending_rate"] == 0.25  # 1 out of 4