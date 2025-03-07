---
# General Hyperparams
num_epochs: &num_epochs 2
init_lr: &init_lr 0.01
final_lr: &final_lr 0.002
weights_warmup_lr: &weights_warmup_lr 0
biases_warmup_lr: &biases_warmup_lr 0.1
quantization_lr: &quantization_lr 0.000002

# Pruning Hyperparams
init_sparsity: &init_sparsity 0.05
pruning_start_epoch: &pruning_start_epoch 0
pruning_end_epoch: &pruning_end_epoch 1
update_frequency: &pruning_update_frequency 0.2
mask_type: &mask_type [1, 4]
prune_none_target_sparsity: &prune_none_target_sparsity 0.4
prune_low_target_sparsity: &prune_low_target_sparsity 0.5
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.65
prune_high_target_sparsity: &prune_high_target_sparsity 0.75

# Quantization Params
quantization_start_epoch: &quantization_start_epoch 1

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: *num_epochs
    
pruning_modifiers:
  - !GMPruningModifier
    params: __ALL_PRUNABLE__
    init_sparsity: *init_sparsity
    final_sparsity: *prune_high_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
    mask_type: *mask_type
             
quantization_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch
    submodules: [ 'model.0', 'model.1', 'model.2', 'model.3', 'model.4', 'model.5', 'model.6', 'model.7', 'model.8', 'model.9', 'model.10', 'model.11', 'model.12', 'model.13', 'model.14', 'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 'model.20', 'model.21', 'model.22', 'model.23' ]
    
    
---