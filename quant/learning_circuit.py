# %%
import random

import matplotlib.pyplot as plt
import networkx as nx


class Gate:
    def __init__(
        self,
        gate_type: str,
        operation: str | None = None,
        input_gates: list["Gate"] | None = None,
    ):
        self.gate_type = gate_type
        self.operation = operation
        self.input_gates = input_gates if input_gates else []

    def __str__(self) -> str:
        input_str = ", ".join([str(input_gate) for input_gate in self.input_gates])
        return f"{self.gate_type}({input_str})"

    def evaluate(self, input_values: list[int]) -> int:
        if self.operation == "INPUT":
            return input_values[0]
        elif self.operation == "AND":
            return int(all(input_values))
        elif self.operation == "OR":
            return int(any(input_values))
        elif self.operation == "NOT":
            return int(not input_values[0])
        elif self.operation == "NAND":
            return int(not all(input_values))
        elif self.operation == "NOR":
            return int(not any(input_values))
        elif self.operation == "XOR":
            return int(sum(input_values) % 2 == 1)
        return 0


class BooleanCircuit:
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.gates: list[list[Gate]] = []
        self.generate_random_circuit()

    def generate_random_circuit(self) -> None:
        gate_types = ["AND", "OR", "NOT", "NAND", "NOR", "XOR"]

        # Create input layer
        input_layer = [Gate(f"INPUT {i+1}", "INPUT") for i in range(self.width)]
        self.gates.append(input_layer)

        # Create hidden layers
        gate_id = 1
        for layer_index in range(self.depth):
            layer: list[Gate] = []
            if layer_index == self.depth - 1:
                # Final layer should have a single gate
                gate_type = random.choice(["AND", "OR", "NAND", "NOR", "XOR"])
                input_gates = self.gates[-1]
                gate = Gate(f"GATE {gate_id}", gate_type, input_gates)
                gate_id += 1
                layer.append(gate)
            else:
                for _ in range(self.width):
                    gate_type = random.choice(gate_types)
                    num_inputs = 1 if gate_type == "NOT" else random.randint(2, 3)
                    input_gates = random.sample(self.gates[-1], min(len(self.gates[-1]), num_inputs))
                    gate = Gate(f"GATE {gate_id}", gate_type, input_gates)
                    gate_id += 1
                    layer.append(gate)
            self.gates.append(layer)

    def __str__(self) -> str:
        circuit_str = ""
        for layer_idx, layer in enumerate(self.gates):
            circuit_str += f"Layer {layer_idx}:\n"
            for gate in layer:
                input_str = ", ".join([input_gate.gate_type for input_gate in gate.input_gates])
                circuit_str += f"{gate.gate_type}: {gate.operation}({input_str})\n"
        return circuit_str

    def __call__(self, inputs: list[int]) -> tuple[list[int], list[list[int]]]:
        intermediate_values: list[list[int]] = [inputs]
        
        for layer_index, layer in enumerate(self.gates[1:], start=1):  # Skip the input layer
            layer_values: list[int] = []
            for gate in layer:
                gate_inputs = [
                    intermediate_values[layer_index - 1][self.gates[layer_index - 1].index(input_gate)]
                    for input_gate in gate.input_gates
                ]
                layer_values.append(gate.evaluate(gate_inputs))
            intermediate_values.append(layer_values)

        output_values = intermediate_values[-1]
        return output_values, intermediate_values

    def plot_circuit(self) -> None:
        G = nx.DiGraph()
        pos = {}
        labels = {}
        node_ids = {}

        # Assign unique IDs to all gates
        node_id = 0
        for layer_idx, layer in enumerate(self.gates):
            for gate_idx, gate in enumerate(layer):
                node_ids[gate] = node_id
                G.add_node(node_id)
                pos[node_id] = (layer_idx, -gate_idx)
                assert isinstance(gate.operation, str)
                label = (gate.gate_type + ":" + gate.operation).split(" ")[1]
                labels[node_id] = label
                node_id += 1

        # Add edges based on gate inputs
        for layer in self.gates:
            for gate in layer:
                for input_gate in gate.input_gates:
                    G.add_edge(node_ids[input_gate], node_ids[gate])

        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            labels=labels,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            ax=ax,
        )
        plt.title("Boolean Circuit Computational Graph")
        plt.show()

if __name__ == "__main__":
    width = 3
    depth = 2
    circuit = BooleanCircuit(width, depth)
    circuit.plot_circuit()
    # test the circuit with all possible boolean inputs
    for i in range(2**width):
        input_vector = [int(x) for x in list(bin(i)[2:].zfill(width))]
        output_vector, intermediate_values = circuit(input_vector)
        print("Input Vector:", input_vector)
        print("Intermediate Values:", intermediate_values)