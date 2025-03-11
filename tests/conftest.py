import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Optional: Import any fixtures that should be available to all tests
import pytest

@pytest.fixture
def recipe_config_path():
    """Return the path to the recipe test configuration file"""
    return os.path.join(project_root, "config", "text_generation_test.yaml")
