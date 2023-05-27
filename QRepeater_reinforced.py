import gym
import qiskit.quantum_info as qi
from stable_baselines3 import PPO
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import random_unitary
import numpy as np
import uuid
import os


class QuantumRepeaterEnv(gym.Env):
    def __init__(self, paths):
        super(QuantumRepeaterEnv, self).__init__()

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)  # Choices for error correction (0) and entanglement purification (1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Measurements of quantum channel conditions

        self.qc = self.initialize_quantum_circuit()
        self.paths = paths

    def step(self, action):
        if action == 0:
            self.apply_error_correction()
        else:
            self.apply_entanglement_purification()

        observation = self.measure_channel_conditions()
        reward = self.measure_transmission_fidelity()  # Reward is based on successful quantum transmission
        done = reward >= 0.95  # Set the done condition when the fidelity reaches 0.95 or above

        return observation, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.qc = self.initialize_quantum_circuit()
        self.reset_quantum_channel()

        observation = self.measure_channel_conditions()
        return observation

    def initialize_quantum_circuit(self):
        # Initialize the quantum circuit
        qreg = QuantumRegister(6, name="q")
        cregz = ClassicalRegister(1, name="cregz")
        cregx = ClassicalRegister(1, name="cregx")
        creg_same = ClassicalRegister(1, name="cr_same")
        qc = QuantumCircuit(qreg, cregz, cregx, creg_same)
        return qc

    def apply_error_correction(self):
        # Apply error correction logic here
        error_correction_crs = [ClassicalRegister(1, name=f"cr_{i}") for i in range(1, 4)]
        self.qc.add_register(*error_correction_crs)
        syndrome_meas = ClassicalRegister(2, name="syndrome_meas")
        self.qc.add_register(syndrome_meas)

        qubit_trios = self.qc.qubits[:3]
        error_corrected_reg = error_correction(self.qc, qubit_trios, [self.qc.cregs[0]] * len(qubit_trios))
        
    def apply_entanglement_purification(self):
        # Apply entanglement purification logic here
        qubit_pair = [self.qc.qubits[0], self.qc.qubits[1]]
        entanglement_purification(self.qc, qubit_pair[0], qubit_pair[1], self.qc.cregs[-1])

    def measure_channel_conditions(self):
        # Measure the channel conditions and normalize
        self.qc.measure(self.qc.qubits[0], self.qc.cregs[0])
        self.qc.measure(self.qc.qubits[1], self.qc.cregs[1])

        backend = Aer.get_backend('qasm_simulator')
        result = execute(self.qc, backend, shots=1).result()
        counts = result.get_counts()

        channel_conditions = np.array([counts.get(f'0 {key}', 0) for key in ['00', '01', '10', '11']])
        channel_conditions = channel_conditions / np.sum(channel_conditions)
        return channel_conditions

    def measure_transmission_fidelity(self):
        # Measure the transmission fidelity, could be represented by error rate
        self.qc.measure(self.qc.qubits[2], self.qc.cregs[2])

        backend = Aer.get_backend('qasm_simulator')
        result = execute(self.qc, backend, shots=1000).result()
        counts = result.get_counts()

        fidelity = counts.get('000', 0) / sum(counts.values())
        return fidelity

    def reset_quantum_channel(self):
        # Reset the Quantum Channel
        self.qc.reset(self.qc.qubits)

    def multiplexing(self, channels):
        num_qubits = self.qc.num_qubits // len(channels)
        for i in range(0, self.qc.num_qubits, num_qubits):
            create_Bell_pair(self.qc, channels[i], channels[i + 1])

    def dijkstra_shortest_path(self, source, destination):
        paths = self.paths
        dist = {vertex: float('infinity') for vertex in paths}
        previous_vertices = {vertex: None for vertex in paths}
        dist[source] = 0
        vertices = paths.copy()
        
        while vertices:
            current_vertex = min(vertices, key=lambda vertex: dist[vertex])
            vertices.remove(current_vertex)
            if dist[current_vertex] == float('infinity'):
                break

            for neighbor, cost in paths[current_vertex].items():
                alternative_route = dist[current_vertex] + cost
                if alternative_route < dist[neighbor]:
                    dist[neighbor] = alternative_route
                    previous_vertices[neighbor] = current_vertex

        path, current_vertex = [], destination
        while previous_vertices[current_vertex] is not None:
            path.append((previous_vertices[current_vertex], current_vertex))
            current_vertex = previous_vertices[current_vertex]

        return path[::-1]

# Graph with nodes as qubit positions and edges as path cost (lower is better)
paths = {
    0: {1: 5, 2: 10},
    1: {3: 10},
    2: {4: 10},
    3: {5: 5},
    4: {5: 5}
}

source = 0
destination = 5
num_repeater_stations = 3

# Initialize the environment and the agent
env = QuantumRepeaterEnv(paths)

# Check if pre-existing model exists, if so, load it
if os.path.isfile("current_quantum_repeater_model.zip"):
    model = PPO.load("current_quantum_repeater_model")
else:
    model = PPO("MlpPolicy", env, verbose=1)

# Define best reward and the evaluation interval
best_reward = -np.inf
evaluation_interval = 1000  # Evaluate the model every 1000 steps
total_timesteps = 10000  # Total number of steps for training

for timestep in range(0, total_timesteps, evaluation_interval):
    # Train the agent
    model.learn(total_timesteps=evaluation_interval)

    # Save the current model
    model.save("current_quantum_repeater_model")

    # Evaluate the model and update the best reward
    avg_reward = np.mean([model.evaluate_policy(env, deterministic=True)[0] for _ in range(10)])
    if avg_reward > best_reward:
        best_reward = avg_reward
        model.save("best_quantum_repeater_model")

# Load the best model
model = PPO.load("best_quantum_repeater_model")

# Use the trained agent to operate the quantum repeater
observation = env.reset()
while True:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)
    if done:
        break

def create_Bell_pair(qc, a, b): 
    qc.h(a)
    qc.cx(a, b)
    qc.name = f"BellPair-{uuid.uuid4()}"

def Bell_pair_teleportation(qc, frodo, gandalf, iluvatar, crz, crx): 
    qc.cx(frodo, gandalf)
    qc.h(frodo)
    qc.barrier()
    qc.measure(frodo, crz)
    qc.measure(gandalf, crx)
    qc.barrier()
    qc.z(iluvatar).c_if(crz, 1)
    qc.x(iluvatar).c_if(crx, 1)

def entanglement_swapping(qc, a, b, crz, crx):
    qc.cx(a, b)
    qc.h(a)
    qc.barrier()
    qc.measure(a, crz)
    qc.measure(b, crx)

def entanglement_purification(qc, a, b, cr_same):
    qc.z(a).c_if(cr_same, 1)
    qc.z(b).c_if(cr_same, 1)

def quantum_memory(qc, a, duration):
    qc.append_gate("DoubleGate", [a], params=[duration])

def error_correction(qc, a, crs):  # Implement error correction codes
    # Three-qubit error correction code, implement more complex codes as needed
    num_qubits = len(a)
    error_corrected_reg = QuantumRegister(num_qubits, name="error_corrected")
    qc.add_register(error_corrected_reg)

    syndrome_meas = ClassicalRegister(1, "syndrome_meas")
    qc.add_register(syndrome_meas)
    for i in range(0, num_qubits, 3):
        qc.cx(a[i], a[i+1])
        qc.cx(a[i], a[i+2])
        qc.cx(a[i+1], a[i+2])
        qc.measure(a[i+2], syndrome_meas)
        qc.x(a[i+1]).c_if(syndrome_meas, 1)
        qc.cx(a[i], a[i+1])
        qc.x(a[i]).c_if(syndrome_meas, 1)
        qc.cx(a[i], error_corrected_reg[i // 3])

    return error_corrected_reg

def synchronization(qc, ops, relative_clock):
    current_timestamp = 0
    
    for clock_event in relative_clock:
        op, qubits, event_timestamp = clock_event
        if current_timestamp < event_timestamp:
            delay_duration = event_timestamp - current_timestamp
            qc.delay(delay_duration, qubits, unit='dt')  # Add delay to maintain synchronization
            current_timestamp = event_timestamp

        # If the operation is in the list of qubit operations, apply it to the qubits
        if op in ops:
            qc.append(op(), qubits)

def multiplexing(qc, channels, names=None):
    num_qubits = qc.num_qubits // len(channels)
    for i in range(0, qc.num_qubits, num_qubits):
        create_Bell_pair(qc, channels[i], channels[i + 1])

def error_correction_circuit(n_repeater_stations):
    circuits = []
    qubit_trios = [QuantumRegister(3, name=f"q_{i}") for i in range(n_repeater_stations)]
    for idx, q_regs in enumerate(zip(*qubit_trios)):
        qc = QuantumCircuit(sum(q_regs, QuantumRegister(0)))
        qc.add_register(ClassicalRegister(1, name=f"cr_{idx}"))
        qc.add_register(ClassicalRegister(2, name=f"syndrome_meas"))
        error_corrected_reg = error_correction(qc, q_regs, [qc.cregs[idx]] * len(q_regs))
        circuits.append(qc)
    return circuits

source = 0
destination = 5
num_repeater_stations = 3

# Initializations
rep_circuits = error_correction_circuit(num_repeater_stations)
shortest_path = env.dijkstra_shortest_path(source, destination)
print(f"Shortest path: {shortest_path}")

for idx, rep_circuit in enumerate(rep_circuits):
    qc = QuantumCircuit(rep_circuit.qregs[0], rep_circuit.qregs[1], ClassicalRegister(1, name=f"cregz_{idx}"), ClassicalRegister(1, name=f"cregx_{idx}"), ClassicalRegister(1, name=f"cr_same_{idx}"))
    qreg = rep_circuit.qregs[0]

    # Initialize the state
    initial_state = qi.random_statevector(2)
    qc.initialize(initial_state.data, qreg[0])

    # Quantum Repeater Workflow
    create_Bell_pair(qc, qreg[0], qreg[1])
    quantum_memory(qc, qreg[0], 100)  # Store qubit 0 for 100 timesteps
    Bell_pair_teleportation(qc, qreg[0], qreg[1], qreg[2], qc.cregs[idx+1], qc.cregs[idx+2])
    entanglement_swapping(qc, qreg[2], qreg[3], qc.cregs[idx+1], qc.cregs[idx+2])
    qubit_trios_qregs = [qubit_trios[i].qubits for i in range(len(qubit_trios))]
    error_corrected_qubits = error_correction(qc, qubit_trios_qregs[idx], [qc.cregs[idx]] * len(qubit_trios))
    qc.measure(qubit_trios_qregs[idx], qc.cregs[idx])
    entanglement_purification(qc, qreg[0], qreg[1], qc.cregs[-1])
    create_Bell_pair(qc, qreg[4], qreg[5])

    # Relative clock events in the form (operation, qubits, timestamp)
    relative_clock = [
        (HGate, [qreg[0]], 10),
        (XGate, [qreg[1]], 20),
        (ZGate, [qreg[2]], 30),
    ]

    # Apply operations with synchronization
    synchronization(qc, [HGate, XGate, ZGate], relative_clock)

    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1).result()
    print(result.get_counts())
