Fine-tuning
===========

This section describes the process of fine-tuning a pre-trained model to
adapt it to new tasks or datasets. Fine-tuning is a common technique used
in transfer learning, where a model is trained on a large dataset and then
fine-tuned on a smaller dataset to improve its performance on specific tasks.
So far the fine-tuning capabilities are only available for PET model.


# Fine-Tuning PET Model with LoRA

Fine-tuning a PET model using LoRA (Low-Rank Adaptation) can significantly
enhance the model's performance on specific tasks while reducing the
computational cost. Below are the steps to fine-tune a PET model from 
`metatrain.experimental.pet` using LoRA.

# What is LoRA?

LoRA (Low-Rank Adaptation) is a technique used to adapt pre-trained models
to new tasks by introducing low-rank matrices into the model's architecture.
This approach reduces the number of trainable parameters, making the
fine-tuning process more efficient and less resource-intensive. LoRA is
particularly useful in scenarios where computational resources are limited
or when quick adaptation to new tasks is required.

By following the steps outlined above, you can effectively fine-tune your
PET model using LoRA, leveraging its benefits for improved performance and
efficiency.

# Prerequisites

1. **Train the Base Model**: 
    - You can either train the base model using the command:
      ```
      mtt train options.yaml
      ```
    - Alternatively, you can use a pre-trained foundational model,
      if you have access to its state dict. 

2. **Define Paths in `options.yaml`**:
    Specify the paths to `model_state_dict`, `all_species.npy`, and
    `self_contributions.npy` in the `options.yaml` file.
    ```yaml
    training:
    MODEL_TO_START_WITH: <path_to_model_state_dict>
    ALL_SPECIES_PATH: <path_to_all_species.npy>
    SELF_CONTRIBUTIONS_PATH: <path_to_self_contributions.npy>
    ```

3. **Set LoRA Parameters**:
    - Set the following parameters in `options.yaml`:
      ```yaml
      LORA_RANK: <desired_rank>
      LORA_ALPHA: <desired_alpha>
      USE_LORA_PEFT: True
      ```

4. **Fine-Tune the Model**:
    - Run the following command to fine-tune the model:
      ```
      mtt train options.yaml
      ```

# Fine-Tuning Options

When `USE_LORA_PEFT` is set to `True`, the original model's weights will be frozen, and
only the adapter layers introduced by LoRA will be trained. This allows for efficient fine-tuning
with fewer parameters. If `USE_LORA_PEFT` is set to `False`, all the weights of the model will be
trained. This can lead to better performance on the specific task but may require more computational
resources, and the model may be prone to overfitting (i.e. loosing accuracy on the original training
set).

