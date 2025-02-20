import torch
from einops import rearrange, repeat

from torch.utils.benchmark import Timer
import torch._dynamo

torch._dynamo.config.suppress_errors = True  # type:ignore

# print(torch._inductor.list_options())
# exit()


# Loop-based implementation
# @torch.compile()
def loop_based(
    vmasks: torch.Tensor,
    ex_maps: torch.Tensor,
    input_seq: torch.Tensor,
    vxBCdt: torch.Tensor,
    qxBCdt: torch.Tensor,
):
    for i, (vm, em) in enumerate(zip(vmasks, ex_maps)):
        input_seq[i, vm] = vxBCdt[i]
        input_seq[i, ~vm] = qxBCdt[i, em]
    return input_seq


#
# def _inner_loop(
#     vmask: torch.Tensor,
#     ex_map: torch.Tensor,
#     input_seq: torch.Tensor,
#     vxBCdt: torch.Tensor,
#     qxBCdt: torch.Tensor,
# ):
#     # print(vmask.size(), vxBCdt.size(), qxBCdt.size(), ex_map.size())
#     # input_seq = torch.masked_scatter(input_seq, vmask, vxBCdt)
#     # input_seq = torch.masked_scatter(input_seq, ~vmask, qxBCdt[ex_map])
#     # return input_seq
#
#     # input_seq.where(vmask > 0, vxBCdt)
#     # input_seq.where(vmask < 0, qxBCdt[ex_map])
#     # return input_seq
#     #
#     # return torch.where(vmask > 0, vxBCdt, qxBCdt[ex_map])
#     input_seq[vmask] = vxBCdt
#     input_seq[~vmask] = qxBCdt[ex_map]
#     return input_seq
#
#
# # @torch.compile(dynamic=True, mode="max-autotune")
# def compile_based(
#     vmasks: torch.Tensor,
#     ex_maps: torch.Tensor,
#     input_seq: torch.Tensor,
#     vxBCdt: torch.Tensor,
#     qxBCdt: torch.Tensor,
# ):
#     batched_masked_index_put = torch.vmap(_inner_loop)
#     return batched_masked_index_put(vmasks, ex_maps, input_seq, vxBCdt, qxBCdt)
#


# Vectorized implementation
# @torch.compile(dynamic=True)
def vectorized(
    vmasks: torch.Tensor,
    ex_maps: torch.Tensor,
    input_seq: torch.Tensor,
    vxBCdt: torch.Tensor,
    qxBCdt: torch.Tensor,
    batch_indices: torch.Tensor,
):
    # Mask-based assignment
    input_seq[vmasks] = rearrange(vxBCdt, "N V D -> (N V) D")

    # Prepare gathered indices
    # gathered_qxBCdt = qxBCdt[batch_indices, ex_maps].flatten(0, 1)  # Shape: (N, M, D)
    gathered_qxBCdt = rearrange(qxBCdt[batch_indices, ex_maps], "N M D -> (N M) D")
    # Assign for ~vmasks
    input_seq[~vmasks] = gathered_qxBCdt
    return input_seq


def test_parity():
    device = "cuda"
    # Define dimensions
    (N, T) = (2, 20_000)  # Minimal sizes

    torch.manual_seed(0)  # For reproducibility
    vmasks = torch.rand(T, device=device) > 0.3  # Boolean mask
    V = int(vmasks.sum().item())
    M = T - V
    vmasks = repeat(vmasks, "T -> N T", N=N).contiguous()
    Q, D = 2_500, 290
    # Generate tensors
    ex_maps = torch.randint(0, Q, (N, M), device=device)  # Indices in qxBCdt
    input_seq = torch.zeros(N, T, D, device=device)  # Output tensor
    vxBCdt = torch.rand(N, V, D, device=device)  # Values for vmasks
    qxBCdt = torch.rand(N, Q, D, device=device)  # Values to be gathered
    out_loop = loop_based(vmasks, ex_maps, input_seq.clone(), vxBCdt, qxBCdt)

    batch_indices = torch.arange(len(vmasks), device=device).unsqueeze(
        1
    )  # Shape: (N, 1)

    out_vector = vectorized(
        vmasks, ex_maps, input_seq, vxBCdt.clone(), qxBCdt, batch_indices
    )
    assert torch.equal(out_loop, out_vector)


for i in range(10000):
    test_parity()


setup = """
device = "cuda"
# Define dimensions
(N, T) = (2, 20_000)  # Minimal sizes

torch.manual_seed(0)  # For reproducibility
vmasks = torch.rand(T, device=device) > 0.5  # Boolean mask
V = int(vmasks.sum().item())
M = T - V
vmasks = repeat(vmasks, "T -> N T", N=N).contiguous()
Q, D = 2_500, 290
# Generate tensors
ex_maps = torch.randint(0, Q, (N, M), device=device)  # Indices in qxBCdt
input_seq = torch.zeros(N, T, D, device=device)  # Output tensor
vxBCdt = torch.rand(N, V, D, device=device)  # Values for vmasks
qxBCdt = torch.rand(N, Q, D, device=device)  # Values to be gathered

batch_indices = torch.arange(len(vmasks),device=device).unsqueeze(1)  # Shape: (N, 1)
"""
# Time loop-based implementation
loop_timer = Timer(
    stmt="loop_based(vmasks, ex_maps, input_seq, vxBCdt, qxBCdt)",
    setup=setup,
    globals=globals(),
    label="Loop Time",
    num_threads=20,
)
vec_timer = Timer(
    stmt="vectorized(vmasks, ex_maps, input_seq, vxBCdt, qxBCdt, batch_indices)",
    setup=setup,
    globals=globals(),
    label="Vectorized Time",
    num_threads=20,
)

loop_res = loop_timer.blocked_autorange(min_run_time=5)
vec_res = vec_timer.blocked_autorange(min_run_time=5)

print(loop_res)
print(vec_res)
# Check if the results are identical
# are_equal = torch.allclose(input_seq_loop, input_seq_vectorized)
#
# print(loop_time, vectorized_time, are_equal)
