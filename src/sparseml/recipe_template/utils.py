from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from sparseml.pytorch.sparsification import (
    ACDCPruningModifier,
    EpochRangeModifier, GMPruningModifier, LearningRateFunctionModifier,
    MagnitudePruningModifier,
    QuantizationModifier,
)
from sparseml.pytorch.utils import get_prunable_layers, get_quantizable_layers
from sparseml.sparsification import ModifierYAMLBuilder, RecipeYAMLBuilder
from torch.nn import Module


class ModifierBuildInfo:
    """
    A class with state and helper methods for building a recipe from
    modifier(s)
    """

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
        self.update(updated_fields=self.fields)

    def update(self, updated_fields: Optional[Dict[str, Any]] = None):
        for key, value in updated_fields.items():
            variable_name = f"{self.modifier_name}_{key}"
            setattr(self.modifier_builder, key, f"eval({variable_name})")
            self.__modifier_recipe_variables[variable_name] = updated_fields[key]

    @property
    def modifier_recipe_variables(self):
        return self.__modifier_recipe_variables


def build_recipe_from_modifier_info(
    modifier_info_groups: Dict[str, List[ModifierBuildInfo]],
    convert_to_md: bool = False
) -> str:
    """
    # TODO
    """
    recipe_variables = {}
    modifier_groups = defaultdict(list)

    for group_name, modifier_group in modifier_info_groups.items():
        for modifier_info in modifier_group:
            if modifier_info is not None:
                recipe_variables.update(modifier_info.modifier_recipe_variables)
                modifier_groups[group_name].append(modifier_info.modifier_builder)

    recipe_builder = RecipeYAMLBuilder(
        variables=recipe_variables,
        modifier_groups=modifier_groups,
    )

    yaml_str = recipe_builder.build_yaml_str()

    return (
        f"---\n{yaml_str}\n---\n" if convert_to_md else yaml_str
    )


_PRUNING_MODIFIER_INFO_REGISTRY = {
    "false": None,
    "true": ModifierBuildInfo(
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
    "acdc": ModifierBuildInfo(
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
    "gmp": ModifierBuildInfo(
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
_QUANTIZATION_MODIFIER_INFO_REGISTRY = {
    "false": None,
    "true": ModifierBuildInfo(
        modifier=QuantizationModifier,
        modifier_name="quantization",
        fields={
            "start_epoch": 0.0,
            "submodules": "null",
            "model_fuse_fn_name": 'fuse_module',
            "disable_quantization_observer_epoch": 2.0,
            "freeze_bn_stats_epoch": 3.0,
            "reduce_range": False,
            "activation_bits": False,
            "tensorrt": False,
        },
    ),
}


def get_quantization_info(
    quantization: Union[str, bool] = False,
    target: str = "vnni",
    model: Optional[Module] = None,
) -> List[ModifierBuildInfo]:
    """
    # TODO
    """
    if isinstance(quantization, bool):
        quantization = str(quantization)

    quantization_info = _QUANTIZATION_MODIFIER_INFO_REGISTRY.get(quantization.lower())
    if quantization_info is None:
        return []

    quantization_info.update({"tensorrt": target == "tensorrt"})

    if model:
        quantizable_layers = [name for name, _ in get_quantizable_layers(module=model)]
        quantization_info.update({"submodules": quantizable_layers})

    return [quantization_info]


def get_pruning_info(
    pruning,
    mask_type,
    global_sparsity,
    model: Optional[Module] = None,
) -> List[ModifierBuildInfo]:
    """
    # TODO
    """
    if isinstance(pruning, bool):
        pruning = str(pruning)

    modifier_info = _PRUNING_MODIFIER_INFO_REGISTRY.get(pruning.lower())
    if modifier_info is None:
        return []

    if isinstance(
        modifier_info.modifier, (MagnitudePruningModifier, GMPruningModifier)
    ):
        modifier_info.update(
            updated_fields={
                "mask_type": mask_type,
                "global_sparsity": global_sparsity,
            }
        )
    elif isinstance(modifier_info.modifier, ACDCPruningModifier):
        modifier_info.update(
            updated_fields={
                "global_sparsity": global_sparsity,
            }
        )

    if model:
        prunable_layers = [name for name, module in get_prunable_layers(module=model)]
        modifier_info.update({"params": prunable_layers})

    return [modifier_info]


def get_training_info(lr_func: str = "linear") -> List[ModifierBuildInfo]:
    """
    # TODO
    """
    epoch_modifier_info = ModifierBuildInfo(
        modifier=EpochRangeModifier,
        modifier_name="epoch_range_mod",
        fields={
            "start_epoch": 0.0,
            "end_epoch": 10,
        })
    lr_modifier_info = ModifierBuildInfo(
        modifier=LearningRateFunctionModifier,
        modifier_name="lr_function_mod",
        fields={
            "start_epoch": 0.0,
            "end_epoch": 10,
            "lr_func": lr_func,
            "init_lr": "1e-3",
            "final_lr": "1e-8",
        })

    return [epoch_modifier_info, lr_modifier_info]
