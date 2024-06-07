# %%
import random

import matplotlib.pyplot as plt
import networkx as nx


class Gate:
    def __init__(
        self,
        gate_type: str,
        operation: str | None = None,
        inputs: list["Gate"] | None = None,
    ):
        self.gate_type = gate_type
        self.operation = operation
        self.inputs = inputs if inputs else []

    def __str__(self) -> str:
        input_str = ", ".join([str(input_gate) for input_gate in self.inputs])
        return f"{self.gate_type}({input_str})"

    def evaluate(self, input_values: list[int]) -> int:
        if self.operation == "INPUT":
            return int(input_values.pop(0))
        elif self.operation == "AND":
            return int(all(input_gate.evaluate(input_values.copy()) for input_gate in self.inputs))
        elif self.operation == "OR":
            return int(any(input_gate.evaluate(input_values.copy()) for input_gate in self.inputs))
        elif self.operation == "NOT":
            return int(not self.inputs[0].evaluate(input_values.copy()))
        elif self.operation == "NAND":
            return int(
                not all(input_gate.evaluate(input_values.copy()) for input_gate in self.inputs)
            )
        elif self.operation == "NOR":
            return int(
                not any(input_gate.evaluate(input_values.copy()) for input_gate in self.inputs)
            )
        elif self.operation == "XOR":
            return int(
                sum(input_gate.evaluate(input_values.copy()) for input_gate in self.inputs) % 2 == 1
            )
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
                inputs = self.gates[-1]
                gate = Gate(f"GATE {gate_id}", gate_type, inputs)
                gate_id += 1
                layer.append(gate)
            else:
                for _ in range(self.width):
                    gate_type = random.choice(gate_types)
                    num_inputs = 1 if gate_type == "NOT" else random.randint(2, 3)
                    inputs = random.sample(self.gates[-1], min(len(self.gates[-1]), num_inputs))
                    gate = Gate(f"GATE {gate_id}", gate_type, inputs)
                    gate_id += 1
                    layer.append(gate)
            self.gates.append(layer)

    def __str__(self) -> str:
        circuit_str = ""
        for layer_idx, layer in enumerate(self.gates):
            circuit_str += f"Layer {layer_idx}:\n"
            for gate in layer:
                input_str = ", ".join([input_gate.gate_type for input_gate in gate.inputs])
                circuit_str += f"{gate.gate_type}: {gate.operation}({input_str})\n"
        return circuit_str

    def __call__(self, inputs: list[int]) -> tuple[list[int], list[list[int]]]:
        input_values = inputs.copy()  # Make a copy of inputs to avoid modifying the original list
        intermediate_values: list[list[int]] = []
        current_values = input_values.copy()

        for layer in self.gates:
            next_values: list[int] = []
            for gate in layer:
                if gate.operation == "INPUT":
                    next_values.append(current_values.pop(0))
                else:
                    next_values.append(gate.evaluate(current_values.copy()))
            intermediate_values.append(next_values)
            current_values = next_values.copy()

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
                for input_gate in gate.inputs:
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


width = 5
circuit = BooleanCircuit(width, 2)
circuit.plot_circuit()
# test the circuit with all possible boolean inputs
for i in range(2**width):
    input_vector = [int(x) for x in list(bin(i)[2:].zfill(width))]
    output_vector, intermediate_values = circuit(input_vector)
    print("Input Vector:", input_vector)
    print("Intermediate Values:", intermediate_values)
