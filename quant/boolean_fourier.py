# %%
from collections.abc import Callable

import numpy as np
import torch

Function = Callable[[torch.Tensor], torch.Tensor]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numbers_to_bit_strings(numbers: torch.Tensor, input_size: int, zero_one=False):
    assert torch.all(numbers < 2**input_size), f"Numbers are too large for input size {input_size}"
    zero_one_bits = torch.tensor(
        [[int(bit) for bit in format(number, f"0{input_size}b")] for number in numbers],
        dtype=torch.float,
    )
    if zero_one:
        return zero_one_bits
    return (-1) ** zero_one_bits


def zero_one_bits_to_minus_one_one_bits(bits: torch.Tensor):
    assert torch.all((bits == 0) | (bits == 1))
    return (-1) ** bits


def minus_one_one_bits_to_zero_one_bits(bits: torch.Tensor):
    assert torch.all((bits == -1) | (bits == 1))
    return (-bits + 1) // 2


def majority_function(bit_strings: torch.Tensor) -> torch.Tensor:
    assert torch.all(torch.abs(bit_strings) == 1)
    # for each bit_string in bit_strings, return the sign of the sum of the bits
    assert bit_strings.shape[1] % 2 == 1, "majority function only works for odd input size"
    return torch.sign(torch.sum(bit_strings, dim=1))


def parity_function(bit_strings: torch.Tensor):
    assert torch.all(torch.abs(bit_strings) == 1)
    return torch.prod(bit_strings, dim=1)


def dictator_function(dictator_string: torch.Tensor) -> Function:
    dictator_string = dictator_string.float()

    def func(bit_strings: torch.Tensor) -> torch.Tensor:
        assert torch.all(torch.abs(bit_strings) == 1)
        assert bit_strings.shape[1] == len(dictator_string)
        out = (bit_strings == dictator_string).all(dim=1).float()
        return zero_one_bits_to_minus_one_one_bits(out)

    return func


def sparse_parity_function(bit_strings: torch.Tensor, indices: torch.Tensor):
    """
    Given a tensor of bit_strings and a tensor of indices, return a matrix of the parities of the
    bitstrings at the given indices.

    Args:
        bit_strings: A tensor of shape (m, n) where m is the number of bit_strings and n is the
        length of each bit string. Each bit string should contain only 1s and -1s
        indices: A tensor of shape (k, n). Each row contains only 0s and 1s. The 1s indicate the
        indices that we want to take the parity of.
    Returns:
        A tensor of shape (m, k) where each row contains the parity of the bits at the indices
        specified by the corresponding row in indices.
    """
    assert torch.all(torch.abs(bit_strings) == 1)
    assert torch.all((indices == 0) | (indices == 1))
    assert bit_strings.shape[1] == indices.shape[1]
    zero_one_bit_strings = minus_one_one_bits_to_zero_one_bits(bit_strings).to(device)
    indices = indices.to(device)
    zero_one_outputs = torch.einsum("ij, kj -> ik", zero_one_bit_strings, indices) % 2
    return zero_one_bits_to_minus_one_one_bits(zero_one_outputs).to("cpu")


# bit_strings = torch.tensor([[1, -1, 1], [-1, 1, 1], [1, 1, -1]])
# indices = torch.tensor([[1, 0, 1], [0, 1, 1]])
# print(sparse_parity_function(bit_strings, indices))


def all_parity_vectors(input_size: int):
    bit_strings = numbers_to_bit_strings(torch.tensor(range(2**input_size)), input_size)
    indices = subsets(input_size)
    return sparse_parity_function(bit_strings, indices)


def function_to_vector(func: Function, input_size: int):
    vec = func(numbers_to_bit_strings(torch.tensor(range(2**input_size)), input_size))
    all_bit_strings = numbers_to_bit_strings(
        torch.tensor(range(2**input_size)), input_size, zero_one=True
    ).int()
    all_bit_strings = all_bit_strings.tolist()
    all_bit_strings = [",".join(map(str, bit_string)) for bit_string in all_bit_strings]
    vec_dict = {all_bit_strings[i]: vec[i].item() for i in range(2**input_size)}
    return vec, vec_dict


def inner_product(func1: Function, func2: Function, input_size: int):
    return (
        torch.dot(
            function_to_vector(func1, input_size)[0], function_to_vector(func2, input_size)[0]
        )
        / 2**input_size
    )


# a function which takes in a number n and returns a tensor of all subsets of [n]
def subsets(input_size: int):
    return numbers_to_bit_strings(torch.tensor(range(2**input_size)), input_size, zero_one=True)


def fourier_transform(func: Function, input_size: int, boolean_outputs: bool = True):
    vec, _ = function_to_vector(func, input_size)
    vec = vec.to(device)
    basis = all_parity_vectors(input_size).to(device)
    coeffs = basis @ vec / 2**input_size
    all_bit_strings = numbers_to_bit_strings(
        torch.tensor(range(2**input_size)), input_size, zero_one=True
    ).int()
    # convert all_bit_strings to list of strings
    all_bit_strings = all_bit_strings.tolist()
    all_bit_strings = [",".join(map(str, bit_string)) for bit_string in all_bit_strings]

    coeffs_dict = {all_bit_strings[i]: coeffs[i].item() for i in range(2**input_size)}
    if boolean_outputs:
        assert (
            sum(coeffs**2 for coeffs in coeffs) == 1
        ), f"The coefficients squared add to {sum(coeffs ** 2)} instead of 1."
    return coeffs, coeffs_dict


def vec_fourier_transform(vec: torch.Tensor, boolean_outputs: bool = True):
    if 0 in vec:
        vec = zero_one_bits_to_minus_one_one_bits(vec)
    vec = vec.to(device)
    assert vec.shape[0] == 2 ** int(np.log2(float(vec.shape[0])))
    input_size = int(np.log2(vec.shape[0]))
    basis = all_parity_vectors(input_size).to(device)
    coeffs = basis @ vec / 2**input_size
    all_bit_strings = numbers_to_bit_strings(
        torch.tensor(range(2**input_size)), input_size, zero_one=True
    ).int()
    # convert all_bit_strings to list of strings
    all_bit_strings = all_bit_strings.tolist()
    all_bit_strings = [",".join(map(str, bit_string)) for bit_string in all_bit_strings]

    coeffs_dict = {all_bit_strings[i]: coeffs[i].item() for i in range(2**input_size)}
    if boolean_outputs:
        assert (
            sum(coeffs**2 for coeffs in coeffs) == 1
        ), f"The coefficients squared add to {sum(coeffs ** 2)} instead of 1."
    return coeffs, coeffs_dict


a = torch.tensor([1, 0, 0, 0, 0, 1.0, 0, 0])
print(torch.fft.fft(a))
