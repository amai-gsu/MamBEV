import torch

# Input tensors
nums = torch.zeros((10, 8, 10, 2))
mask = torch.normal(nums[..., 0].clone() + 0.5, 0.25).round().bool()

# Vectorized operation
flattened_mask = mask.view(-1)  # Flatten the mask
flattened_nums = nums.view(
    -1, nums.size(-1)
)  # Flatten the nums array except for the last dimension

# Masked selection
masked_output = flattened_nums[flattened_mask]  # Select the masked elements

# Grouping indices for future access
batch_indices, feature_indices = torch.where(mask)[:2]
output_indices = torch.stack(
    (batch_indices, feature_indices), dim=1
)  # Optional grouping for later indexing
print(batch_indices, feature_indices)
