from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb

# from learning_circuit import BooleanCircuit
# from quant_model import MLP
from quant.learning_circuit import BooleanCircuit
from quant.quant_model import MLP

class HookedMLP(nn.Module):
    def __init__(self, mlp: MLP):
        super().__init__()
        self.mlp = mlp
        self.activations = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.activations.clear()
        for i, layer in enumerate(self.mlp.layers):
            x = layer(x)
            x = torch.relu(x)
            self.activations[f'layer_{i}'] = x
        x = self.mlp.fc(x)
        self.activations['output'] = x
        return torch.sigmoid(x)

def generate_dataset(circuit: BooleanCircuit, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    inputs = []
    outputs = []
    intermediates = []
    for _ in range(num_samples):
        input_vector = torch.randint(0, 2, (circuit.width,))
        output_vector, intermediate_vectors = circuit(input_vector.tolist())
        inputs.append(input_vector)
        intermediates.append(torch.tensor(intermediate_vectors[:-1]))
        outputs.append(torch.tensor(output_vector))

    return torch.stack(inputs).float(), torch.stack(intermediates).float(), torch.stack(outputs).float()

def train_mlp(mlp: MLP, circuit: BooleanCircuit, num_samples: int, num_epochs: int, batch_size: int, device: torch.device) -> MLP:
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        inputs, _, outputs = generate_dataset(circuit, num_samples)
        dataset = TensorDataset(inputs, outputs)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        mlp.train()
        for batch_inputs, batch_outputs in train_dataloader:
            batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
            
            optimizer.zero_grad()
            predictions = mlp(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            mlp.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for test_inputs, test_outputs in test_dataloader:
                    test_inputs, test_outputs = test_inputs.to(device), test_outputs.to(device)
                    test_predictions = mlp(test_inputs)
                    predicted = (test_predictions > 0.5).float()
                    total += test_outputs.size(0)
                    correct += (predicted == test_outputs).sum().item()
                
                accuracy = 100 * correct / total
                print(f"MLP Training - Epoch {epoch}/{num_epochs}, Train Loss: {loss.item()}, Test Accuracy: {accuracy:.2f}%")

    print("MLP training completed")
    return mlp

def train_linear_probes(hooked_mlp: HookedMLP, circuit: BooleanCircuit, num_samples: int, num_epochs: int, batch_size: int, device: torch.device) -> Dict[str, nn.Module]:
    inputs, intermediates, _ = generate_dataset(circuit, num_samples)
    dataset = TensorDataset(inputs, intermediates)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    inputs = inputs.to(device)
    hooked_mlp.to(device)
    hooked_mlp.eval()
    hooked_mlp(inputs)

    probes = {}
    optimizers = {}
    for layer_name in hooked_mlp.activations.keys():
        probes[layer_name] = nn.Linear(hooked_mlp.mlp.layers[0].out_features, circuit.width * circuit.depth).to(device)
        optimizers[layer_name] = optim.Adam(probes[layer_name].parameters())

    criterion = nn.BCEWithLogitsLoss()

    wandb.init(project="linear_probes_boolean_circuit")
    
    for epoch in range(num_epochs):
        for batch_inputs, batch_intermediates in dataloader:
            batch_inputs, batch_intermediates = batch_inputs.to(device), batch_intermediates.to(device)
            batch_intermediates = batch_intermediates.view(-1, circuit.width * circuit.depth)
            
            # Forward pass through hooked MLP
            hooked_mlp(batch_inputs)
            
            # Train probes for each layer
            for layer_idx, (layer_name, activation) in enumerate(hooked_mlp.activations.items()):
                if layer_name == 'output':
                    continue
                optimizers[layer_name].zero_grad()
                activation = activation.detach()
                probe_output = probes[layer_name](activation)
                loss = criterion(probe_output, batch_intermediates)
                loss.backward()
                optimizers[layer_name].step()
                
                wandb.log({f"{layer_name}_loss": loss.item()})
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
    
    wandb.finish()
    return probes

def check_probe_accuracies(hooked_mlp: HookedMLP, probes: Dict[str, nn.Module], circuit: BooleanCircuit, num_samples: int, device: str = "cpu") -> Dict[str, List[float]]:
    inputs, intermediates, _ = generate_dataset(circuit, num_samples)
    inputs = inputs.to(device)
    intermediates = intermediates.view(-1, circuit.width * circuit.depth).to(device)

    hooked_mlp.eval()
    hooked_mlp(inputs)
    hooked_mlp.to(device)

    accuracies = {}
    for layer_name, probe in probes.items():
        if layer_name == 'output':
            continue
        probe.eval()
        with torch.no_grad():
            probe_output = probe(hooked_mlp.activations[layer_name])
            predicted = (probe_output > 0.5).float()
            # Calculate accuracy for each gate
            gate_accuracies = [(predicted[:, i] == intermediates[:, i]).sum().item() / num_samples * 100 for i in range(intermediates.size(1))]
            accuracies[layer_name] = gate_accuracies
    return accuracies

if __name__ == "__main__":
    width = 3
    depth = 7
    circuit = BooleanCircuit(width, depth)
    d_mlp = 10
    n_hidden_layers = 5
    model = HookedMLP(MLP(width, d_mlp, n_hidden_layers))
    inps, ints, outs = generate_dataset(circuit, 16)
    device = torch.device("mps")
    train_linear_probes(model, circuit, num_samples=16, num_epochs=100, batch_size=4, device=device)
    breakpoint()