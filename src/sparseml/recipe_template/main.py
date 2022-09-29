# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, List, Optional, Union

from torch.nn import Module

__all__ = [
    "recipe_template",
]

from sparseml.recipe_template.utils import (
    ModifierBuildInfo,
    get_quantization_info, get_pruning_info, get_training_info,
    build_recipe_from_modifier_info,
)


def recipe_template(
    pruning: Union[str, bool] = False,
    quantization: Union[str, bool] = False,
    lr_func: str = "linear",
    mask_type: str = "unstructured",
    global_sparsity: bool = False,
    model: Optional[Module] = None,
    **kwargs,
):
    """
    Return a valid recipe given specified args and kwargs
    :return: A string representing a valid recipe
    """
    # This is where we setup all the modifiers
    modifier_groups: Dict[str, List[ModifierBuildInfo]] = {}

    modifier_groups["training_modifiers"] = get_training_info(lr_func=lr_func)

    pruning_modifier = get_pruning_info(pruning, mask_type, global_sparsity,
                                        model)
    modifier_groups["pruning_modifiers"] = pruning_modifier

    quantization_modifier = get_quantization_info(quantization)
    modifier_groups["quantization_modifiers"] = quantization_modifier

    yaml_recipe = build_recipe_from_modifier_info(
        modifier_info_groups=modifier_groups,
    )
    return yaml_recipe


def local_test():
    print(recipe_template(pruning="acdc", quantization=True))


if __name__ == '__main__':
    local_test()
