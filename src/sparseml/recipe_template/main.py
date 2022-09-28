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

__all__ = [
    "recipe_template",
]

from typing import Any, Dict, List, Optional, Union

from sparseml.pytorch.sparsification import (
    ACDCPruningModifier, EpochRangeModifier, GMPruningModifier,
    LearningRateFunctionModifier, MagnitudePruningModifier, QuantizationModifier,
)
from sparseml.sparsification import ModifierYAMLBuilder, RecipeYAMLBuilder
from torch.nn import Module


class ModifierInfo:

    def __init__(
        self,
        modifier,
        modifier_name,
        fields: Optional[Dict[str, Any]] = None,
    ):
        self.modifier = modifier
        self.modifier_name = modifier_name
        self.modifier_builder = ModifierYAMLBuilder(modifier)
        self.fields = fields or {}
        self.__modifier_recipe_variables = {}
        self._set_fields()

    def _set_fields(self):
        for key, value in self.fields.items():
            variable_name = f"{self.modifier_name}_{key}"
            setattr(self.modifier_builder, key, f"eval({variable_name})")
            self.__modifier_recipe_variables[variable_name] = self.fields[key]

    @property
    def modifier_recipe_variables(self):
        return self.__modifier_recipe_variables


def build_recipe_from_modifier_info(
    modifier_builders: List[ModifierInfo],
):
    recipe_variables = {}
    for builder in modifier_builders:
        recipe_variables.update(builder.modifier_recipe_variables)

    recipe_builder = RecipeYAMLBuilder(
        variables=recipe_variables,
        modifier_groups={
            "modifiers": [builder.modifier_builder for builder in modifier_builders],
        })

    return recipe_builder.build_yaml_str()


def _get_quantization_params():
    return {
        "start_epoch": 0.0,
        "submodules": ['blocks.0', 'blocks.2'],
        "model_fuse_fn_name": 'fuse_module',
        "disable_quantization_observer_epoch": 2.0,
        "freeze_bn_stats_epoch": 3.0,
        "reduce_range": False,
        "activation_bits": False,
    }


_PRUNING_MODIFIER_REGISTRY = {
    "false": None,
    "true": ModifierInfo(
        modifier=MagnitudePruningModifier,
        modifier_name="pruning",
        fields={
            "init_sparsity": 0.05,
            "final_sparsity": 0.8,
            "start_epoch": 0.0,
            "end_epoch": 10.0,
            "update_frequency": 1.0,
            "params": "__ALL_PRUNABLE__",
            "leave_enabled": True,
            "inter_func": "cubic",
            "mask_type": "unstructured",
        }),
    "acdc": ModifierInfo(
        modifier=ACDCPruningModifier,
        modifier_name="pruning",
        fields={
            "compression_sparsity": 0.9,
            "start_epoch": 0,
            "end_epoch": 100,
            "update_frequency": 5,
            "params": "__ALL_PRUNABLE__",
            "global_sparsity": True,
        }),
    "gmp": ModifierInfo(
        modifier=GMPruningModifier,
        modifier_name="pruning",
        fields={
            "init_sparsity": 0.05,
            "final_sparsity": 0.8,
            "start_epoch": 0.0,
            "end_epoch": 10.0,
            "update_frequency": 1.0,
            "params": ["re:.*weight"],
            "leave_enabled": True,
            "inter_func": "cubic",
            "mask_type": "unstructured",
        }),

}
_QUANTIZATION_MODIFIER_REGISTRY = {
    "false": None,
    "true": ModifierInfo(modifier=QuantizationModifier,
                         fields=_get_quantization_params(),
                         modifier_name="quantization",
                         ),
}


def recipe_template(
    pruning: Optional[Union[str, bool]] = None,
    quantization: Optional[Union[str, bool]] = None,
    lr: str = "linear",
    model: Optional[Module] = None,
    **kwargs,
):
    """
    Return a valid recipe given specified args and kwargs

    # TODO: fill params
    # TODO: What about adding a list of pruning Modifiers? Epoch range + Acdc ex..
    :return: A string representing a valid recipe
    """
    # This is where we setup all the modifiers
    all_modifiers_with_info: List[ModifierInfo] = []
    all_modifiers_with_info.extend(_get_default_modifiers())

    pruning_modifier = _add_pruning_modifiers(pruning)
    all_modifiers_with_info.append(pruning_modifier)

    quantization_modifier = _add_quantization_modifier(quantization)
    all_modifiers_with_info.append(quantization_modifier)

    all_modifiers_with_info = [elem for elem in all_modifiers_with_info if elem]
    yaml_recipe = build_recipe_from_modifier_info(
        modifier_builders=all_modifiers_with_info,
    )
    return yaml_recipe


def _add_quantization_modifier(quantization):
    if isinstance(quantization, bool):
        quantization = str(quantization)
    return _QUANTIZATION_MODIFIER_REGISTRY.get(quantization.lower())


def _add_pruning_modifiers(pruning):
    if isinstance(pruning, bool):
        pruning = str(pruning)
    return _PRUNING_MODIFIER_REGISTRY.get(pruning.lower())


def _get_default_modifiers():
    epoch_modifier_info = ModifierInfo(
        modifier=EpochRangeModifier,
        modifier_name="epoch_range_mod",
        fields={
            "start_epoch": 0.0,
            "end_epoch": 10,
        })
    lr_modifier_info = ModifierInfo(
        modifier=LearningRateFunctionModifier,
        modifier_name="lr_function_mod",
        fields={
            "start_epoch": 0.0,
            "end_epoch": 10,
            "lr_func": "linear",
            "init_lr": "1e-3",
            "final_lr": "1e-8",
        })

    return [epoch_modifier_info, lr_modifier_info]


def local_test():
    print(recipe_template(pruning="acdc", quantization=True))


if __name__ == '__main__':
    local_test()
