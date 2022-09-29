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
def test_recipe_template_returns_a_loadable_recipe(pruning, quantization, kwargs):
    actual = recipe_template(pruning=pruning, quantization=quantization, **kwargs)
    assert actual
    ScheduledModifierManager.from_yaml(file_path=actual)
