# %%
import random

import matplotlib.pyplot as plt
import networkx as nx


class Gate:
    def __init__(
        self,
        gate_id: int,
        operation: str,
        layer_index: int,
        intralayer_index: int,
        input_gates: list["Gate"] | None = None,
    ):
        self.gate_id = gate_id
        self.operation = operation
        self.layer_index = layer_index
        self.intralayer_index = intralayer_index
        self.input_gates = input_gates if input_gates else []

    def __str__(self) -> str:
        return str(self.gate_id)
        return f"{self.gate_id} ({self.layer_index}-{self.intralayer_index}): {self.operation}"
        input_str = ", ".join([str(input_gate) for input_gate in self.input_gates])
        return f"{self.gate_id}({input_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return NotImplemented
        return self.gate_id == other.gate_id

    def __hash__(self) -> int:
        return hash(self.gate_id)

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
        self.unused_gates: list[list[Gate]] = []
        self.generate_random_circuit()

    def generate_random_circuit(self) -> None:
        operations = ["AND", "OR", "NOT", "NAND", "NOR", "XOR"]

        # Create input layer
        input_layer = [Gate(i, "INPUT", layer_index=0, intralayer_index=i) for i in range(self.width)]
        self.gates.append(input_layer)

        # Create hidden layers
        gate_id = len(input_layer)
        for layer_index in range(1, self.depth + 1):
            layer: list[Gate] = []
            unused_gates: set[Gate] = set(self.gates[layer_index-1])
            if layer_index == self.depth:
                # Final layer should have a single gate
                # operation = random.choice(["AND", "OR", "NAND", "NOR", "XOR"])
                # using XOR for balance of zero and one outputs -- other operations lead to collapse to all zeros or all ones
                operation = "XOR"
                input_gates = self.gates[-1]
                gate = Gate(gate_id, operation, layer_index=layer_index, intralayer_index=0, input_gates=input_gates)
                gate_id += 1
                layer.append(gate)
                unused_gates = set()
            else:
                for i in range(self.width):
                    operation = random.choice(operations)
                    num_inputs = 1 if operation == "NOT" else random.randint(2, 3)
                    input_gates = random.sample(self.gates[-1], min(len(self.gates[-1]), num_inputs))
                    unused_gates -= set(input_gates)
                    gate = Gate(gate_id, operation, layer_index=layer_index, intralayer_index=i, input_gates=input_gates)
                    gate_id += 1
                    layer.append(gate)
            self.gates.append(layer)
            self.unused_gates.append(sorted(list(unused_gates), key=lambda gate: gate.gate_id))

    def __str__(self) -> str:
        circuit_str = ""
        for layer_idx, layer in enumerate(self.gates):
            circuit_str += f"Layer {layer_idx}:\n"
            for gate in layer:
                input_str = ", ".join([input_gate.gate_id for input_gate in gate.input_gates])
                circuit_str += f"{gate.gate_id}: {gate.operation}({input_str})\n"
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
                # label = (gate.gate_name + ":" + gate.operation).split(" ")[1] # old
                label = f"{gate.gate_id} ({gate.layer_index}.{gate.intralayer_index}):\n{gate.operation}" # new
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
            node_size=1000,
            node_color="lightblue",
            font_size=8,
            font_weight="normal",
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