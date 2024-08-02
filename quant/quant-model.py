# %%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset
import itertools

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleTaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    # create a dataset of sequences of length `n_control_bits` + `n_task_bits`
    # the sequences are bit strings. The first `n_control_bits` bits describe the task:
    # they are zero everywhere except for the `task_num`-th bit, which is 1.
    # The next `n_task_bits` bits are random.
    # The target is the parity of the relevant variables, which are at the indices of the task bits
    # given in `relevant_vars`
    def __init__(
        self,
        n_task_bits: int,
        n_control_bits: int,
        task_num: int,
        relevant_vars: torch.Tensor,
        dataset_length: int,
    ):
        assert len(relevant_vars.shape) == 1
        assert relevant_vars.dtype == torch.int64
        assert relevant_vars.shape[0] <= n_task_bits
        assert all([0 <= i < n_task_bits for i in relevant_vars])
        self.data = torch.zeros(dataset_length, n_control_bits + n_task_bits, device=device)
        self.task_bits = torch.randint(
            0, 2, (dataset_length, n_task_bits), dtype=torch.float32, device=device
        )
        self.data[:, n_control_bits:] = self.task_bits
        self.data[:, task_num] = 1.0
        self.dataset_length = dataset_length
        self.targets = self.task_bits[:, relevant_vars].sum(dim=1) % 2

        self.relevant_vars = relevant_vars

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx: int | slice):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y


class MultiTaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        n_task_bits: int,
        n_control_bits: int,
        relevant_vars: torch.Tensor,
        dataset_length_per_task: int | torch.Tensor,
    ):
        if isinstance(dataset_length_per_task, torch.Tensor):
            assert len(dataset_length_per_task) == n_control_bits
            data_set_lengths = dataset_length_per_task
        else:
            data_set_lengths = torch.Tensor([dataset_length_per_task] * n_control_bits)
        assert len(relevant_vars) == n_control_bits
        self.datasets = {
            i: SingleTaskDataset(
                n_task_bits, n_control_bits, i, relevant_vars[i], int(data_set_lengths[i].item())
            )
            for i in range(n_control_bits)
        }
        self.dataset_length = sum(data_set_lengths)

        # mix all the datasets into a single dataset, including labels
        self.data = torch.cat([self.datasets[i].data for i in range(len(relevant_vars))])
        self.targets = torch.cat([self.datasets[i].targets for i in range(len(relevant_vars))])
        self.relevant_variables = {
            i: self.datasets[i].relevant_vars for i in range(len(relevant_vars))
        }

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx: int | slice):
        return self.data[idx], self.targets[idx]


class MultiTaskTestDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        n_task_bits: int,
        n_control_bits: int,
        relevant_vars: torch.Tensor,
    ):
        self.n_task_bits = n_task_bits
        self.n_control_bits = n_control_bits
        self.relevant_vars = relevant_vars
        
        # Generate all possible task bit combinations
        self.task_bits = self.generate_all_bit_vectors(n_task_bits)
        
        # Create control bits for each task
        self.control_bits = torch.eye(n_control_bits)
        
        # Combine control bits and task bits
        self.data = torch.cat([
            self.control_bits.repeat_interleave(len(self.task_bits), dim=0),
            self.task_bits.repeat(n_control_bits, 1)
        ], dim=1)
        
        # Calculate targets
        self.targets = torch.stack([
            self.task_bits[:, task].sum(dim=1) % 2
            for task in relevant_vars
        ]).t().flatten()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int | slice):
        return self.data[idx], self.targets[idx]

    def generate_all_bit_vectors(self, n: int):
        # Generate all possible combinations of 0 and 1 of length n
        all_combinations = list(itertools.product([0, 1], repeat=n))
        return torch.tensor(all_combinations, dtype=torch.float32)

def generate_random_relevant_vars(
    n_task_bits: int, n_control_bits: int, parity_size: int
) -> torch.Tensor:
    vars = torch.zeros(n_control_bits, parity_size, dtype=torch.int64)
    for i in range(n_control_bits):
        vars[i] = torch.randperm(n_task_bits)[:parity_size]
    return vars


def generate_dataset(
    n_task_bits: int,
    n_control_bits: int,
    relevant_vars: torch.Tensor,
    subtask_probs: torch.Tensor,
    dataset_length: int,
):
    # relevant_vars = generate_random_relevant_vars(n_task_bits, n_control_bits)
    assert len(subtask_probs) == n_control_bits
    assert sum(subtask_probs) == 1.0
    dataset_lengths = torch.tensor(
        np.random.multinomial(dataset_length, subtask_probs, 1), device=device
    ).squeeze()
    return MultiTaskDataset(n_task_bits, n_control_bits, relevant_vars, dataset_lengths)

def generate_test_set(
    n_task_bits: int,
    n_control_bits: int,
    tasks: torch.Tensor, # aka relevant_vars
):
    return MultiTaskTestDataset(n_task_bits, n_control_bits, tasks)


# %%
# # test the multitask dataset
# n_task_bits = 4
# n_control_bits = 3
# relevant_vars = torch.tensor([[0, 1], [0, 2], [1, 2]])
# dataset_length_per_task = 4

# dataset = MultiTaskDataset(n_task_bits, n_control_bits, relevant_vars, dataset_length_per_task)
# print(dataset.data, dataset.targets)

# %%
class MicroTransformer(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        vocab_size: int = 2,
        d_model: int = 4,
        num_layers: int = 4,
        num_heads: int = 2,
        d_head: int | None = None,
        d_mlp: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_head = d_head if d_head is not None else d_model // num_heads
        self.d_mlp = d_mlp if d_mlp is not None else 4 * d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, num_heads, self.d_head, self.d_mlp)
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)
        x = self.fc(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_head: int, d_mlp: int, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, d_head, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        attn_output = self.attention(x)
        x = x + attn_output
        x = self.layer_norm1(x)

        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.layer_norm2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_head: int, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head

        self.query = nn.Linear(d_model, num_heads * d_head)
        self.key = nn.Linear(d_model, num_heads * d_head)
        self.value = nn.Linear(d_model, num_heads * d_head)

        self.out = nn.Linear(num_heads * d_head, d_model)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        attn_weights = nn.functional.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.d_head)
        )
        attn_output = self.out(attn_output)

        return attn_output


class MLP(nn.Module):
    def __init__(self, d_input: int, d_mlp: int, n_hidden_layers: int, d_output: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(d_input, d_mlp)] + [nn.Linear(d_mlp, d_mlp) for _ in range(n_hidden_layers)]
        )
        self.fc = nn.Linear(d_mlp, d_output)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = self.fc(x)
        return F.sigmoid(x)


# %%

# training loop with logging to wandb. Each step, generate a new dataset to train with of size
# batch_size

n_control_bits = 2
n_task_bits = 10
n_active_bits = 2 # k in the MSP spec
subtask_probs = torch.tensor([2**i for i in range(n_control_bits)], dtype=torch.float32)
# subtask_probs = torch.ones(n_control_bits, dtype=torch.float32)
loss_fn = nn.BCELoss()
n_steps = 25000
batch_size = 100
log_interval = 100

tasks = generate_random_relevant_vars(n_task_bits, n_control_bits, n_active_bits)
# tasks = torch.tensor([[0, 1], [0, 2]])
subtask_probs = subtask_probs / subtask_probs.sum()
d_mlp = 32
n_hidden_layers = 5
model = MLP(n_control_bits + n_task_bits, d_mlp, n_hidden_layers, 1).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

testset = generate_test_set(n_task_bits, n_control_bits, tasks)

wandb.init(project="quant")
wandb.config.update(
    {
        "n_control_bits": n_control_bits,
        "n_task_bits": n_task_bits,
        "subtask_probs": subtask_probs.tolist(),
        "tasks": {i: tasks[i].tolist() for i in range(len(tasks))},
    }
)
wandb.watch(model)
run_id = wandb.run.id
run_name = wandb.run.name

model.train()
for i, step in enumerate(range(n_steps)):
    dataset = generate_dataset(n_task_bits, n_control_bits, tasks, subtask_probs, batch_size)
    x = dataset.data
    y = dataset.targets
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred.squeeze(), y)
    loss.backward()
    optimizer.step()
    if i % log_interval == 0:
        wandb.log({"loss": loss.item()})
        accuracy = (y_pred.round() == y).float().mean().item()
        wandb.log({"accuracy": accuracy})
        print(f"Step {step}, Loss: {loss.item()}, Accuracy: {accuracy}")

        # Calculate and log per-task accuracy
        x = testset.data
        y = testset.targets
        y_pred = model(x)
        for task_id in range(n_control_bits):
            task_mask = x[:, task_id] == 1
            task_pred = y_pred[task_mask]
            task_true = y[task_mask]
            task_accuracy = (task_pred.round() == task_true).float().mean().item()
            wandb.log({f"full_accuracies/task_{task_id}": task_accuracy})

        # log the loss on each subtask
        for j in range(n_control_bits):
            # generate a dataset of size batch_size for each subtask
            dataset = SingleTaskDataset(n_task_bits, n_control_bits, j, tasks[j], batch_size)
            x = dataset.data
            y = dataset.targets
            y_pred = model(x)
            loss = loss_fn(y_pred.squeeze(), y)
            wandb.log({f"losses/task_{j}": loss.item()})
            accuracy = (y_pred.round() == y).float().mean().item()
            wandb.log({f"accuracies/task_{j}": accuracy})

with torch.no_grad():
    testset = generate_test_set(n_task_bits, n_control_bits, tasks)
    
    model.eval()
    
    x = testset.data
    y = testset.targets
    y_pred = model(x).round().squeeze()
    
    test_accuracy = (y == y_pred).float().mean().item()
    print(f"Test Accuracy: {test_accuracy:.4f}")
    wandb.log({"test_accuracy": test_accuracy})
    
    # Evaluate per-task accuracy
    for task_id in range(n_control_bits):
        task_mask = x[:, task_id] == 1
        task_accuracy = (y_pred[task_mask] == y[task_mask]).float().mean().item()
        print(f"Task {task_id} Accuracy: {task_accuracy:.4f}")
        wandb.log({f"test_accuracies/task_{task_id}": task_accuracy})

wandb.finish()
# save model
torch.save(model.state_dict(), f"runs/model_{run_name}.pth")
# %%
