import gym
import qiskit.quantum_info as qi
import numpy as np
import os
import uuid
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import HGate, XGate, ZGate
from stable_baselines3 import PPO

class QuantumRepeaterEnv(gym.Env):
    def __init__(self, paths):
        super(QuantumRepeaterEnv, self).__init__()

        if not self.valid_graph(paths):
            raise ValueError("The input paths are not a valid graph with appropriate weights")
        
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.qc = self.initialize_quantum_circuit()
        self.paths = paths

    @staticmethod
    def valid_graph(paths):
        if not paths:
            return False
        for p in paths.values():
            if not p:
                return False
            for weight in p.values():
                if weight <= 0:
                    return False
        return True

    def step(self, action):
        if action not in [0,1]:
            raise ValueError(f"Illegal action: {action}. Action must be either 0 (error correction) or 1 (entanglement purification).")

        if action == 0:
            self.apply_error_correction()
        else:
            self.apply_entanglement_purification()

        observation = self.measure_channel_conditions()
        reward = self.measure_transmission_fidelity()
        done = reward >= 0.95

        return observation, reward, done, {}

    def reset(self):
        self.qc = self.initialize_quantum_circuit()
        self.reset_quantum_channel()

        observation = self.measure_channel_conditions()
        return observation

    def initialize_quantum_circuit(self):
        qreg = QuantumRegister(6, name="q")
        cregz = ClassicalRegister(1, name="cregz")
        cregx = ClassicalRegister(1, name="cregx")
        creg_same = ClassicalRegister(1, name="cr_same")
        qc = QuantumCircuit(qreg, cregz, cregx, creg_same)
        return qc

    def apply_error_correction(self):
        error_correction_crs = [ClassicalRegister(1, name=f"cr_{i}") for i in range(1, 4)]
        self.qc.add_register(*error_correction_crs)
        syndrome_meas = ClassicalRegister(2, name="syndrome_meas")
        self.qc.add_register(syndrome_meas)

        qubit_trios = self.qc.qubits[:3]
        error_corrected_reg = error_correction(self.qc, qubit_trios, [self.qc.cregs[0]] * len(qubit_trios))

    def apply_entanglement_purification(self):
        qubit_pair = [self.qc.qubits[0], self.qc.qubits[1]]
        entanglement_purification(self.qc, qubit_pair[0], qubit_pair[1], self.qc.cregs[-1])

    def measure_channel_conditions(self):
        self.qc.measure(self.qc.qubits[0], self.qc.cregs[0])
        self.qc.measure(self.qc.qubits[1], self.qc.cregs[1])

        backend = Aer.get_backend('qasm_simulator')
        result = execute(self.qc, backend, shots=1).result()
        counts = result.get_counts()

        channel_conditions = np.array([counts.get(f'0 {key}', 0) for key in ['00', '01', '10', '11']])
        channel_conditions = channel_conditions / np.sum(channel_conditions)
        return channel_conditions

    def measure_transmission_fidelity(self):
        self.qc.measure(self.qc.qubits[2], self.qc.cregs[2])

        backend = Aer.get_backend('qasm_simulator')
        result = execute(self.qc, backend, shots=1000).result()
        counts = result.get_counts()

        fidelity = sum(counts.values()) - counts.get('000', 0)
        fidelity /= sum(counts.values())
        return 1 - fidelity

    def reset_quantum_channel(self):
        self.qc.reset(self.qc.qubits)
        for creg in self.qc.cregs:
            creg.reset()

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

def error_correction_circuit(num_repeater_stations):
    circuits = []
    for i in range(num_repeater_stations):
        qreg = QuantumRegister(6, name=f"q_{i}")
        cregz = ClassicalRegister(1, name=f"cregz_{i}")
        cregx = ClassicalRegister(1, name=f"cregx_{i}")
        creg_same = ClassicalRegister(1, name=f"cr_same_{i}")
        qc = QuantumCircuit(qreg, cregz, cregx, creg_same)
        circuits.append(qc)
    return circuits

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

def error_correction(qc, a, crs):
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
            qc.delay(delay_duration, qubits, unit='dt')
            current_timestamp = event_timestamp

        if op in ops:
            qc.append(op(), qubits)

def multiplexing(qc, channels, names=None):
    num_qubits = qc.num_qubits // len(channels)
    for i in range(0, qc.num_qubits, num_qubits):
        create_Bell_pair(qc, channels[i], channels[i + 1])

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

env = QuantumRepeaterEnv(paths)

if os.path.isfile("current_quantum_repeater_model.zip"):
    model = PPO.load("current_quantum_repeater_model")
else:
    model = PPO("MlpPolicy", env, verbose=1)

best_reward = -np.inf
evaluation_interval = 1000
total_timesteps = 10000

for timestep in range(0, total_timesteps, evaluation_interval):
    model.learn(total_timesteps=evaluation_interval)

    model.save("current_quantum_repeater_model")

    avg_reward = np.mean([model.evaluate_policy(env, deterministic=True)[0] for _ in range(10)])
    if avg_reward > best_reward:
        best_reward = avg_reward
        model.save("best_quantum_repeater_model")

model = PPO.load("best_quantum_repeater_model")

observation = env.reset()
while True:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)
    if done:
        break

source = 0
destination = 5
num_repeater_stations = 3

rep_circuits = error_correction_circuit(num_repeater_stations)
shortest_path = env.dijkstra_shortest_path(source, destination)

print(f"Shortest path: {shortest_path}")

for idx, rep_circuit in enumerate(rep_circuits):
    qc = QuantumCircuit(rep_circuit.qregs[0], rep_circuit.qregs[1], ClassicalRegister(1, name=f"cregz_{idx}"), ClassicalRegister(1, name=f"cregx_{idx}"), ClassicalRegister(1, name=f"cr_same_{idx}"))
    qreg = rep_circuit.qregs[0]

    initial_state = qi.random_statevector(2)
    qc.initialize(initial_state.data, qreg[0])

    create_Bell_pair(qc, qreg[0], qreg[1])
    quantum_memory(qc, qreg[0], 100)
    Bell_pair_teleportation(qc, qreg[0], qreg[1], qreg[2], qc.cregs[idx+1], qc.cregs[idx+2])
    entanglement_swapping(qc, qreg[2], qreg[3], qc.cregs[idx+1], qc.cregs[idx+2])
    qubit_trios_qregs = [qubit_trios[i].qubits for i in range(len(qubit_trios))]
    error_corrected_qubits = error_correction(qc, qubit_trios_qregs[idx], [qc.cregs[idx]] * len(qubit_trios))
    qc.measure(qubit_trios_qregs[idx], qc.cregs[idx])
    entanglement_purification(qc, qreg[0], qreg[1], qc.cregs[-1])
    create_Bell_pair(qc, qreg[4], qreg[5])

    relative_clock = [
        (HGate, [qreg[0]], 10),
        (XGate, [qreg[1]], 20),
        (ZGate, [qreg[2]], 30),
    ]

    synchronization(qc, [HGate, XGate, ZGate], relative_clock)

    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1).result()
    print(result.get_counts())       
