# %%
from collections.abc import Callable

import numpy as np
import torch

Function = Callable[[torch.Tensor], torch.Tensor]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%


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
    print(input_size)
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


def walsh_hadamard_transform(vec: torch.Tensor):
    """
    Compute the Walsh-Hadamard transform of a Boolean function f using PyTorch.

    Parameters:
    f (torch.Tensor): Tensor representing the Boolean function values.

    Returns:
    torch.Tensor: Tensor representing the Fourier coefficients.
    """
    n = vec.shape[0]
    assert (n & (n - 1)) == 0, "Size of input must be a power of 2"

    # Initialize H as the input tensor f
    H = vec.clone()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                x = H[i + j]
                y = H[i + j + h]
                H[i + j] = x + y
                H[i + j + h] = x - y
        h *= 2

    return H / torch.sqrt(torch.tensor(n, dtype=vec.dtype))


# def boolean_fourier_transform(vec: torch.Tensor):
#     # n = vec.dim()  # Get the number of input variables
#     # truth_table = torch.tensor(
#     #     [vec[tuple(int(b) for b in bin(i)[2:].zfill(n))] for i in range(2**n)], dtype=torch.float32
#     # )
#     truth_table = vec
#     input_size = len(vec)
#     truth_table = truth_table * 2 - 1  # Convert from {0, 1} to {-1, 1}

#     # Compute the Walsh-Hadamard transform
#     wht_coefficients = torch.fft.fft(truth_table)

#     # Scale the coefficients to obtain the Boolean Fourier transform coefficients
#     bft_coefficients = wht_coefficients / (2 ** (input_size / 2))

#     return bft_coefficients


# # Example Boolean function as a list of 0s and 1s
# boolean_function = [1, 0, 1, 0, 1, 1.0, 0, 0]

# # Convert Boolean function to a PyTorch tensor
# boolean_tensor = torch.tensor(boolean_function, dtype=torch.float32)
# # boolean_tensor = zero_one_bits_to_minus_one_one_bits(boolean_tensor)

# # Compute the Boolean Fourier transform
# fourier_coefficients = walsh_hadamard_transform(boolean_tensor)

# print(fourier_coefficients, (fourier_coefficients**2).sum())
# print(vec_fourier_transform(boolean_tensor)[0])
# # print(
# #     torch.fft.fft(boolean_tensor) / 2**3,
# # )
# print(boolean_fourier_transform(boolean_tensor))

# %%
# tensor = torch.zeros(64)
# tensor[-1] = 1.0
# vec_fourier_transform(tensor)[1]


# %%
def boolean_fourier_transform(f: torch.Tensor):
    n = f.size(0).bit_length() - 1
    assert f.size(0) == 2**n, "Input length must be a power of 2"

    # Define the Hadamard matrix recursively
    def hadamard_matrix(order: int):
        if order == 0:
            return torch.tensor([[1]], dtype=torch.float32)
        else:
            H = hadamard_matrix(order - 1)
            return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)

    H = hadamard_matrix(n)

    # Compute the Fourier coefficients
    F = torch.matmul(H, f.float()) / (2**n)

    return torch.sign(F)  # Output is either -1 or 1


# boolean_fourier_transform(tensor)
# tensor
