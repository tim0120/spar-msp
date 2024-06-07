# %%

import torch

import quant.boolean_fourier as bf

# %%


def n_tribes(num_tribes: int, tribe_size: int) -> bf.Function:
    # outputs a function which returns -1 if any of the n tribes contain only -1s,
    # and +1 otherwise
    def func(boolean_strings: torch.Tensor) -> torch.Tensor:
        assert boolean_strings.shape[-1] == tribe_size * num_tribes
        if len(boolean_strings.shape) == 1:
            boolean_strings = boolean_strings.unsqueeze(0)
        tribe_ands = torch.zeros(boolean_strings.shape[0], num_tribes)
        for tribe in range(num_tribes):
            tribe_bits = boolean_strings[:, tribe * tribe_size : (tribe + 1) * tribe_size]
            tribe_ands[:, tribe] = torch.all(tribe_bits == -1, dim=1)
        return 1 - torch.any(tribe_ands, dim=1).int() * 2

    return func


fun = n_tribes(3, 3)
fun(
    torch.tensor(
        [
            [1, 1, 1, -1, 1, -1, 1, -1, 1],
            [1, 1, 1, -1, -1, -1, 1, -1, 1],
            [-1, 1, -1, 1, -1, -1, -1, -1, 1],
        ],
    )
)
