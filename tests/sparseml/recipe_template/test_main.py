import pytest
from sparseml import recipe_template
import yaml
from sparseml.pytorch.optim import ScheduledModifierManager

@pytest.mark.parametrize("pruning, quantization, kwargs", [
    (True, True, {}),
    ("true", "true", {}),
    ("false", "False", {}),
    ("acdc", "false", {})
])
def test_recipe_template_returns_a_loadable_recipe():
    actual = recipe_template()
    assert actual
    ScheduledModifierManager.from_yaml(file_path=actual)
