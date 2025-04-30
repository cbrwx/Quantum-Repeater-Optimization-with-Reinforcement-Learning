import gym
import numpy as np
import os
import uuid
import logging
import time
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import pickle
import json
from pathlib import Path

# Qiskit imports
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, transpile
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error, thermal_relaxation_error, pauli_error
from qiskit.providers.ibmq import least_busy
from qiskit_ibm_provider import IBMProvider
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller, Optimize1qGates, CXCancellation
from qiskit.quantum_info import random_statevector, partial_trace, state_fidelity, DensityMatrix
from qiskit.circuit.library import HGate, XGate, ZGate, PhaseGate, U3Gate, RXGate, RYGate, RZGate
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.ignis.verification.randomized_benchmarking import randomized_benchmarking_seq
from qiskit.visualization import plot_histogram, plot_state_city, plot_bloch_multivector
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QFT, ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms.classifiers import VQC

# Stable Baselines RL imports
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

# Set up advanced logging with rotation
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
Path("./logs").mkdir(parents=True, exist_ok=True)

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = './logs/quantum_repeater.log'
log_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger('quantum_repeater')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Constants
DECOHERENCE_RATE = 0.001
MAX_FIDELITY = 0.99
MIN_ACCEPTABLE_FIDELITY = 0.75
ENTANGLEMENT_ATTEMPTS = 5
TIME_SLOT_DURATION = 1000  # in dt units

@dataclass
class QuantumChannel:
    """Represents a quantum channel between nodes."""
    distance: float
    loss_db_per_km: float = 0.2
    decoherence_rate: float = DECOHERENCE_RATE
    success_probability: float = 0.5
    
    def calculate_loss(self) -> float:
        """Calculate the total loss in the channel."""
        return 10**(-self.loss_db_per_km * self.distance / 10)
    
    def calculate_success_probability(self, time_elapsed: float = 0) -> float:
        """Calculate success probability with decoherence."""
        base_prob = self.success_probability * self.calculate_loss()
        return base_prob * np.exp(-self.decoherence_rate * time_elapsed)

class CustomNetworkFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for quantum network data."""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Linear(n_input_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)

class AdvancedQuantumRepeaterEnv(gym.Env):
    """
    Advanced Quantum Repeater Network Environment for reinforcement learning.
    Simulates a quantum repeater network with quantum error correction, 
    entanglement purification, and quantum memory management.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 network_topology: Dict[int, Dict[int, Union[float, QuantumChannel]]],
                 noise_model: Optional[NoiseModel] = None,
                 max_steps: int = 100,
                 reward_scaling: float = 1.0,
                 use_continuous_actions: bool = False,
                 num_qubits_per_node: int = 8,
                 advanced_features: bool = True):
        super(AdvancedQuantumRepeaterEnv, self).__init__()
        
        # Validate input network topology
        if not self.validate_network_topology(network_topology):
            raise ValueError("Invalid network topology provided")
        
        self.network_topology = self.convert_to_quantum_channels(network_topology)
        self.nodes = list(network_topology.keys())
        self.num_nodes = len(self.nodes)
        self.max_steps = max_steps
        self.reward_scaling = reward_scaling
        self.num_qubits_per_node = num_qubits_per_node
        self.advanced_features = advanced_features
        self.steps_taken = 0
        
        # Set up action and observation spaces
        if use_continuous_actions:
            # Continuous actions: [error_correction_strength, purification_threshold, memory_preservation]
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0]), 
                high=np.array([1.0, 1.0, 1.0]), 
                dtype=np.float32
            )
        else:
            # Discrete actions: 0=error correction, 1=entanglement purification, 2=quantum memory preservation
            # With advanced features: 3=entanglement swapping, 4=measurement, 5=reset
            if advanced_features:
                self.action_space = spaces.Discrete(6)
            else:
                self.action_space = spaces.Discrete(3)
        
        # Observation space includes:
        # - Channel fidelities matrix (N x N nodes)
        # - Qubit coherence times for each node
        # - Success probabilities for entanglement generation
        # - Node congestion metrics
        obs_dimension = (self.num_nodes * self.num_nodes) + (self.num_nodes * 3)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(obs_dimension,), 
            dtype=np.float32
        )
        
        # Initialize the quantum simulation
        self.backend = QasmSimulator()
        self.noise_model = noise_model
        
        if self.noise_model:
            self.backend.set_options(noise_model=self.noise_model)
            
        # Create quantum and classical registers for each node
        self.node_registers = {}
        self.initialize_quantum_registers()
        
        # Track entanglement pairs between nodes
        self.entanglement_pairs = {}
        
        # Performance metrics
        self.metrics = {
            'fidelity_history': [],
            'success_rate_history': [],
            'end_to_end_entanglement_time': [],
            'resource_consumption': [],
            'reward_history': []
        }
        
        # Network graph visualization
        self.network_graph = self.create_network_graph()
        
        # Time tracking for simulation
        self.current_time = 0
        self.time_slot = 0
        
        logger.info(f"Initialized AdvancedQuantumRepeaterEnv with {self.num_nodes} nodes and {'continuous' if use_continuous_actions else 'discrete'} action space")

    @staticmethod
    def validate_network_topology(network_topology):
        """Validate the network topology structure."""
        if not network_topology:
            return False
        
        # Check if graph is connected using BFS
        visited = set()
        start_node = next(iter(network_topology.keys()))
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            node = queue.pop(0)
            for neighbor in network_topology[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # If not all nodes are visited, graph is not connected
        if len(visited) != len(network_topology):
            return False
            
        # Check all edge weights
        for node, edges in network_topology.items():
            for target, weight in edges.items():
                if not isinstance(target, int) or target not in network_topology:
                    return False
                if isinstance(weight, (int, float)) and weight <= 0:
                    return False
                
        return True
        
    def convert_to_quantum_channels(self, topology):
        """Convert simple distance weights to QuantumChannel objects."""
        quantum_network = {}
        
        for source, targets in topology.items():
            quantum_network[source] = {}
            for target, weight in targets.items():
                if isinstance(weight, QuantumChannel):
                    quantum_network[source][target] = weight
                else:
                    # Create QuantumChannel with distance=weight and default parameters
                    quantum_network[source][target] = QuantumChannel(
                        distance=float(weight),
                        loss_db_per_km=0.2,
                        decoherence_rate=DECOHERENCE_RATE,
                        success_probability=np.exp(-float(weight)/10)  # Success prob decreases with distance
                    )
        
        return quantum_network
    
    def create_network_graph(self):
        """Create a NetworkX graph representation of the quantum network."""
        G = nx.Graph()
        
        # Add nodes
        for node in self.nodes:
            G.add_node(node)
            
        # Add edges with attributes
        for source, targets in self.network_topology.items():
            for target, channel in targets.items():
                G.add_edge(
                    source, 
                    target, 
                    weight=channel.distance,
                    success_prob=channel.success_probability,
                    loss=channel.loss_db_per_km
                )
                
        return G
    
    def initialize_quantum_registers(self):
        """Initialize quantum registers for each node in the network."""
        for node in self.nodes:
            qreg = QuantumRegister(self.num_qubits_per_node, f'q_{node}')
            creg = ClassicalRegister(self.num_qubits_per_node, f'c_{node}')
            
            # Create additional registers for syndrome measurements
            syndrome_reg = ClassicalRegister(2, f'synd_{node}')
            bell_result_reg = ClassicalRegister(2, f'bell_{node}')
            
            qc = QuantumCircuit(qreg, creg, syndrome_reg, bell_result_reg)
            
            # Initialize qubits to |0⟩ state
            qc.reset(qreg)
            
            self.node_registers[node] = {
                'qreg': qreg,
                'creg': creg,
                'syndrome_reg': syndrome_reg,
                'bell_result_reg': bell_result_reg,
                'circuit': qc,
                'qubit_status': ['idle'] * self.num_qubits_per_node,
                'qubit_fidelity': [1.0] * self.num_qubits_per_node,
                'entanglement_partners': [None] * self.num_qubits_per_node
            }
            
    def reset(self):
        """Reset the environment to initial state."""
        # Reset all metrics and counters
        self.steps_taken = 0
        self.current_time = 0
        self.time_slot = 0
        self.entanglement_pairs = {}
        
        # Reset quantum registers
        self.initialize_quantum_registers()
        
        # Reset performance metrics
        for key in self.metrics:
            self.metrics[key] = []
            
        # Get initial observation
        observation = self._get_observation()
        
        logger.info("Environment reset")
        return observation
    
    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: Either a discrete action index or continuous action vector
            
        Returns:
            observation: Current state observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        self.steps_taken += 1
        self.current_time += TIME_SLOT_DURATION
        self.time_slot += 1
        
        # Apply decoherence effects based on elapsed time
        self._apply_decoherence()
        
        # Parse and execute action
        if isinstance(self.action_space, spaces.Discrete):
            action_executed = self._execute_discrete_action(action)
        else:
            action_executed = self._execute_continuous_action(action)
            
        # Get state observation after action
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action_executed)
        self.metrics['reward_history'].append(reward)
        
        # Check termination conditions
        done = (self.steps_taken >= self.max_steps) or self._check_success_condition()
        
        # Gather additional info
        info = {
            'action_executed': action_executed,
            'average_fidelity': np.mean([f for node_data in self.node_registers.values() 
                                        for f in node_data['qubit_fidelity']]),
            'time_elapsed': self.current_time,
            'entanglement_pairs_count': len(self.entanglement_pairs)
        }
        
        return observation, reward * self.reward_scaling, done, info
    
    def _execute_discrete_action(self, action):
        """Execute discrete action in the environment."""
        if action == 0:  # Error correction
            node = np.random.choice(self.nodes)
            qubits_to_correct = self._select_error_correction_qubits(node)
            success = self._apply_error_correction(node, qubits_to_correct)
            return f"error_correction_node_{node}_success_{success}"
            
        elif action == 1:  # Entanglement purification
            # Select two nodes that have entanglement
            if not self.entanglement_pairs:
                # Create new entanglement if none exists
                source, target = self._select_nodes_for_entanglement()
                self._attempt_entanglement_generation(source, target)
                return f"entanglement_generation_{source}_{target}"
            else:
                # Purify existing entanglement
                pair_key = np.random.choice(list(self.entanglement_pairs.keys()))
                source, target = pair_key.split('_')
                source, target = int(source), int(target)
                success = self._apply_entanglement_purification(source, target)
                return f"entanglement_purification_{source}_{target}_success_{success}"
                
        elif action == 2:  # Quantum memory preservation
            node = np.random.choice(self.nodes)
            self._apply_quantum_memory_preservation(node)
            return f"quantum_memory_preservation_node_{node}"
            
        elif action == 3 and self.advanced_features:  # Entanglement swapping
            # Find a chain of three nodes where end nodes have entanglement with middle
            chains = self._find_entanglement_chains()
            if chains:
                chain = np.random.choice(chains)
                success = self._apply_entanglement_swapping(chain)
                return f"entanglement_swapping_chain_{chain}_success_{success}"
            else:
                # Fall back to entanglement generation
                source, target = self._select_nodes_for_entanglement()
                self._attempt_entanglement_generation(source, target)
                return f"fallback_entanglement_generation_{source}_{target}"
                
        elif action == 4 and self.advanced_features:  # Measurement
            # Select a node and measure its qubits
            node = np.random.choice(self.nodes)
            results = self._measure_node_qubits(node)
            return f"measurement_node_{node}_results_{results}"
            
        elif action == 5 and self.advanced_features:  # Reset qubits
            # Select a node and reset its qubits
            node = np.random.choice(self.nodes)
            self._reset_node_qubits(node)
            return f"reset_node_{node}"
            
        else:
            logger.warning(f"Unknown action {action}")
            return "unknown_action"
    
    def _execute_continuous_action(self, action):
        """Execute continuous action in the environment."""
        error_correction_strength = action[0]
        purification_threshold = action[1]
        memory_preservation = action[2]
        
        # Apply error correction with varying strength
        if error_correction_strength > 0.3:
            node = self._select_node_by_priority()
            qubits_to_correct = self._select_error_correction_qubits(node)
            self._apply_error_correction(node, qubits_to_correct, 
                                        strength=error_correction_strength)
        
        # Apply entanglement purification based on threshold
        if purification_threshold > 0:
            for pair_key in list(self.entanglement_pairs.keys()):
                source, target = pair_key.split('_')
                source, target = int(source), int(target)
                pair_data = self.entanglement_pairs[pair_key]
                
                # If fidelity is below threshold, try to purify
                if pair_data['fidelity'] < purification_threshold:
                    self._apply_entanglement_purification(source, target)
        
        # Apply quantum memory preservation based on action value
        if memory_preservation > 0:
            for node in self.nodes:
                self._apply_quantum_memory_preservation(node, 
                                                      strength=memory_preservation)
        
        return f"continuous_action_{action}"
    
    def _apply_decoherence(self):
        """Apply decoherence effects to all qubits."""
        for node, node_data in self.node_registers.items():
            # Apply decoherence to each qubit
            for i, fidelity in enumerate(node_data['qubit_fidelity']):
                # Decoherence model: exponential decay
                new_fidelity = fidelity * np.exp(-DECOHERENCE_RATE * TIME_SLOT_DURATION)
                node_data['qubit_fidelity'][i] = new_fidelity
                
                # If qubit is entangled, update the entanglement fidelity
                if node_data['entanglement_partners'][i] is not None:
                    partner_node, partner_qubit = node_data['entanglement_partners'][i]
                    pair_key = f"{min(node, partner_node)}_{max(node, partner_node)}"
                    
                    if pair_key in self.entanglement_pairs:
                        pair_data = self.entanglement_pairs[pair_key]
                        pair_data['fidelity'] *= np.exp(-DECOHERENCE_RATE * TIME_SLOT_DURATION)
    
    def _select_error_correction_qubits(self, node):
        """Select qubits for error correction based on their fidelity."""
        node_data = self.node_registers[node]
        # Select qubits with lowest fidelity that aren't currently in use
        qubits = []
        fidelities = []
        
        for i, status in enumerate(node_data['qubit_status']):
            if status != 'in_use':
                qubits.append(i)
                fidelities.append(node_data['qubit_fidelity'][i])
                
        if not qubits:
            return []
            
        # Sort by fidelity (ascending)
        sorted_indices = np.argsort(fidelities)
        # Choose up to 3 qubits with lowest fidelity
        return [qubits[i] for i in sorted_indices[:min(3, len(sorted_indices))]]
    
    def _apply_error_correction(self, node, qubits, strength=1.0):
        """Apply error correction to the specified qubits."""
        if not qubits:
            return False
            
        node_data = self.node_registers[node]
        qc = node_data['circuit']
        qreg = node_data['qreg']
        syndrome_reg = node_data['syndrome_reg']
        
        # Mark qubits as in use
        for q in qubits:
            node_data['qubit_status'][q] = 'in_use'
        
        # Apply error correction code (e.g., 3-qubit bit flip code)
        if len(qubits) >= 3:
            # Apply 3-qubit code
            q1, q2, q3 = qubits[:3]
            
            # Create the error correction circuit
            qc.cx(qreg[q1], qreg[q2])
            qc.cx(qreg[q1], qreg[q3])
            
            # Measure syndrome
            qc.barrier()
            qc.measure(qreg[q2], syndrome_reg[0])
            qc.measure(qreg[q3], syndrome_reg[1])
            
            # Apply corrections based on syndrome measurement
            qc.x(qreg[q1]).c_if(syndrome_reg, 3)  # Both syndrome bits are 1
            qc.x(qreg[q2]).c_if(syndrome_reg, 1)  # First syndrome bit is 1
            qc.x(qreg[q3]).c_if(syndrome_reg, 2)  # Second syndrome bit is 1
            
            # Execute circuit
            try:
                # Only run the relevant part of the circuit
                sub_qc = QuantumCircuit(3, 2)
                sub_qc.cx(0, 1)
                sub_qc.cx(0, 2)
                sub_qc.measure([1, 2], [0, 1])
                
                # Add noise scaled by (1-strength) to represent variable error correction quality
                if self.noise_model:
                    noise_factor = 1.0 - (0.5 * strength)  # Lower strength = more noise
                    custom_noise = NoiseModel()
                    custom_noise.add_all_qubit_quantum_error(
                        depolarizing_error(0.01 * noise_factor, 1), 
                        ['x', 'cx']
                    )
                    result = execute(sub_qc, 
                                    Aer.get_backend('qasm_simulator'),
                                    noise_model=custom_noise,
                                    shots=1).result()
                else:
                    result = execute(sub_qc, 
                                    Aer.get_backend('qasm_simulator'),
                                    shots=1).result()
                
                # Update fidelities based on error correction
                improvement_factor = 0.8 + (0.2 * strength)
                for q in qubits[:3]:
                    # Error correction improves fidelity but can't exceed 1.0
                    node_data['qubit_fidelity'][q] = min(
                        1.0, 
                        node_data['qubit_fidelity'][q] * improvement_factor
                    )
                    
                return True
                
            except Exception as e:
                logger.error(f"Error in error correction: {str(e)}")
                return False
                
        return False
    
    def _select_nodes_for_entanglement(self):
        """Select two nodes for entanglement generation."""
        # Find nodes with available qubits
        available_nodes = []
        for node, node_data in self.node_registers.items():
            if 'idle' in node_data['qubit_status']:
                available_nodes.append(node)
        
        if len(available_nodes) < 2:
            # Not enough available nodes, return a random pair
            return np.random.choice(self.nodes), np.random.choice(self.nodes)
            
        # Select pair of nodes that have a direct connection
        valid_pairs = []
        for i in range(len(available_nodes)):
            for j in range(i+1, len(available_nodes)):
                source = available_nodes[i]
                target = available_nodes[j]
                if target in self.network_topology[source]:
                    valid_pairs.append((source, target))
        
        if not valid_pairs:
            # No valid pairs with connections, return random pair
            idx = np.random.choice(len(available_nodes), size=2, replace=False)
            return available_nodes[idx[0]], available_nodes[idx[1]]
            
        # Return a random valid pair
        return valid_pairs[np.random.randint(len(valid_pairs))]
    
    def _attempt_entanglement_generation(self, source, target):
        """Attempt to generate entanglement between two nodes."""
        source_data = self.node_registers[source]
        target_data = self.node_registers[target]
        
        # Find available qubits
        source_qubit = -1
        target_qubit = -1
        
        for i, status in enumerate(source_data['qubit_status']):
            if status == 'idle':
                source_qubit = i
                break
                
        for i, status in enumerate(target_data['qubit_status']):
            if status == 'idle':
                target_qubit = i
                break
                
        if source_qubit == -1 or target_qubit == -1:
            logger.debug(f"No available qubits for entanglement between {source} and {target}")
            return False
            
        # Get the quantum channel
        channel = self.network_topology[source][target] if target in self.network_topology[source] else \
                 self.network_topology[target][source]
        
        # Calculate success probability based on distance and current time
        success_prob = channel.calculate_success_probability(time_elapsed=self.current_time)
        
        # Attempt entanglement
        if np.random.random() < success_prob:
            # Success! Create entanglement
            source_qc = source_data['circuit']
            target_qc = target_data['circuit']
            
            # Create Bell pair
            source_qc.h(source_data['qreg'][source_qubit])
            source_qc.barrier()
            
            # In a real system, this would be a non-local operation requiring classical communication
            # We simulate it as if it happened successfully
            
            # Record the entanglement
            pair_key = f"{min(source, target)}_{max(source, target)}"
            
            # Calculate initial fidelity based on channel properties
            initial_fidelity = success_prob * 0.9  # Some degradation from ideal
            
            self.entanglement_pairs[pair_key] = {
                'qubits': (source_qubit, target_qubit),
                'fidelity': initial_fidelity,
                'creation_time': self.current_time
            }
            
            # Update node registries
            source_data['qubit_status'][source_qubit] = 'entangled'
            target_data['qubit_status'][target_qubit] = 'entangled'
            
            source_data['entanglement_partners'][source_qubit] = (target, target_qubit)
            target_data['entanglement_partners'][target_qubit] = (source, source_qubit)
            
            # Update qubit fidelities
            source_data['qubit_fidelity'][source_qubit] = initial_fidelity
            target_data['qubit_fidelity'][target_qubit] = initial_fidelity
            
            logger.debug(f"Entanglement created between node {source} (qubit {source_qubit}) and node {target} (qubit {target_qubit}) with fidelity {initial_fidelity:.3f}")
            return True
        else:
            logger.debug(f"Entanglement attempt failed between {source} and {target}")
            return False
    
    def _apply_entanglement_purification(self, source, target):
        """Apply entanglement purification to improve fidelity."""
        pair_key = f"{min(source, target)}_{max(source, target)}"
        
        if pair_key not in self.entanglement_pairs:
            logger.debug(f"No entanglement pair exists between {source} and {target}")
            return False
            
        pair_data = self.entanglement_pairs[pair_key]
        source_qubit, target_qubit = pair_data['qubits']
        
        source_data = self.node_registers[source]
        target_data = self.node_registers[target]
        
        # We need additional qubits for purification
        source_aux = -1
        target_aux = -1
        
        for i, status in enumerate(source_data['qubit_status']):
            if status == 'idle' and i != source_qubit:
                source_aux = i
                break
                
        for i, status in enumerate(target_data['qubit_status']):
            if status == 'idle' and i != target_qubit:
                target_aux = i
                break
                
        if source_aux == -1 or target_aux == -1:
            logger.debug(f"No auxiliary qubits available for purification between {source} and {target}")
            return False
            
        # Create an auxiliary entangled pair
        success = self._attempt_entanglement_generation(source, target)
        if not success:
            return False
            
        # Now perform entanglement purification
        source_qc = source_data['circuit']
        target_qc = target_data['circuit']
        
        # Apply CNOT gates
        source_qc.cx(source_data['qreg'][source_qubit], source_data['qreg'][source_aux])
        target_qc.cx(target_data['qreg'][target_qubit], target_data['qreg'][target_aux])
        
        # Measure auxiliary qubits
        source_qc.measure(source_data['qreg'][source_aux], source_data['bell_result_reg'][0])
        target_qc.measure(target_data['qreg'][target_aux], target_data['bell_result_reg'][1])
        
        # In reality, this requires classical communication to compare results
        # We simulate the improvement probabilistically
        
        # Calculate purification success probability
        current_fidelity = pair_data['fidelity']
        purification_success_prob = current_fidelity * (1 - current_fidelity)
        
        if np.random.random() < purification_success_prob:
            # Purification succeeds - improve fidelity
            new_fidelity = min(
                (current_fidelity**2 + (1-current_fidelity)**2) / 
                (current_fidelity**2 + 2*current_fidelity*(1-current_fidelity) + (1-current_fidelity)**2),
                MAX_FIDELITY
            )
            
            # Update fidelities
            pair_data['fidelity'] = new_fidelity
            source_data['qubit_fidelity'][source_qubit] = new_fidelity
            target_data['qubit_fidelity'][target_qubit] = new_fidelity
            
            logger.debug(f"Purification succeeded: Fidelity improved from {current_fidelity:.3f} to {new_fidelity:.3f}")
            
            # Reset auxiliary qubits
            source_data['qubit_status'][source_aux] = 'idle'
            target_data['qubit_status'][target_aux] = 'idle'
            source_data['entanglement_partners'][source_aux] = None
            target_data['entanglement_partners'][target_aux] = None
            
            return True
        else:
            # Purification fails - slight degradation
            new_fidelity = current_fidelity * 0.9
            
            # Update fidelities
            pair_data['fidelity'] = new_fidelity
            source_data['qubit_fidelity'][source_qubit] = new_fidelity
            target_data['qubit_fidelity'][target_qubit] = new_fidelity
            
            logger.debug(f"Purification failed: Fidelity degraded from {current_fidelity:.3f} to {new_fidelity:.3f}")
            
            # Reset auxiliary qubits
            source_data['qubit_status'][source_aux] = 'idle'
            target_data['qubit_status'][target_aux] = 'idle'
            source_data['entanglement_partners'][source_aux] = None
            target_data['entanglement_partners'][target_aux] = None
            
            return False
    
    def _apply_quantum_memory_preservation(self, node, strength=1.0):
        """Apply quantum memory preservation techniques to qubits in a node."""
        node_data = self.node_registers[node]
        
        # Apply dynamical decoupling sequences to preserve quantum states
        for i, status in enumerate(node_data['qubit_status']):
            if status != 'idle':
                # Apply X-X decoupling sequence
                qc = node_data['circuit']
                qreg = node_data['qreg']
                
                # Simple Hahn echo sequence
                qc.barrier()
                qc.x(qreg[i])  # π pulse
                qc.barrier()
                
                # Calculate preservation effect
                # Higher strength means better preservation
                preservation_factor = 0.85 + (0.15 * strength)
                
                # Update fidelity - slower decay
                current_fidelity = node_data['qubit_fidelity'][i]
                new_fidelity = current_fidelity * preservation_factor
                node_data['qubit_fidelity'][i] = min(new_fidelity, MAX_FIDELITY)
                
                # If qubit is entangled, update entanglement fidelity
                if node_data['entanglement_partners'][i] is not None:
                    partner_node, partner_qubit = node_data['entanglement_partners'][i]
                    pair_key = f"{min(node, partner_node)}_{max(node, partner_node)}"
                    
                    if pair_key in self.entanglement_pairs:
                        pair_data = self.entanglement_pairs[pair_key]
                        pair_data['fidelity'] *= preservation_factor
                        pair_data['fidelity'] = min(pair_data['fidelity'], MAX_FIDELITY)
        
        logger.debug(f"Applied quantum memory preservation to node {node} with strength {strength:.2f}")
        return True
    
    def _find_entanglement_chains(self):
        """Find chains of three nodes where entanglement swapping can be applied."""
        chains = []
        
        # Get all entangled pairs
        entangled_pairs = {}
        for pair_key, pair_data in self.entanglement_pairs.items():
            node1, node2 = map(int, pair_key.split('_'))
            if node1 not in entangled_pairs:
                entangled_pairs[node1] = []
            if node2 not in entangled_pairs:
                entangled_pairs[node2] = []
                
            entangled_pairs[node1].append(node2)
            entangled_pairs[node2].append(node1)
        
        # Find nodes that have at least two entanglement connections
        for middle_node, connected_nodes in entangled_pairs.items():
            if len(connected_nodes) >= 2:
                # This node can be a middle node for swapping
                for i in range(len(connected_nodes)):
                    for j in range(i+1, len(connected_nodes)):
                        chain = (connected_nodes[i], middle_node, connected_nodes[j])
                        chains.append(chain)
        
        return chains
    
    def _apply_entanglement_swapping(self, chain):
        """Apply entanglement swapping on a chain of three nodes."""
        node1, middle_node, node3 = chain
        
        # Find the entanglement pair keys
        pair1_key = f"{min(node1, middle_node)}_{max(node1, middle_node)}"
        pair2_key = f"{min(middle_node, node3)}_{max(middle_node, node3)}"
        
        if pair1_key not in self.entanglement_pairs or pair2_key not in self.entanglement_pairs:
            logger.debug(f"Entanglement pairs not found for chain {chain}")
            return False
            
        # Get pair data
        pair1_data = self.entanglement_pairs[pair1_key]
        pair2_data = self.entanglement_pairs[pair2_key]
        
        # Get qubits involved
        if middle_node == int(pair1_key.split('_')[0]):
            middle_qubit1 = pair1_data['qubits'][0]
            node1_qubit = pair1_data['qubits'][1]
        else:
            middle_qubit1 = pair1_data['qubits'][1]
            node1_qubit = pair1_data['qubits'][0]
            
        if middle_node == int(pair2_key.split('_')[0]):
            middle_qubit2 = pair2_data['qubits'][0]
            node3_qubit = pair2_data['qubits'][1]
        else:
            middle_qubit2 = pair2_data['qubits'][1]
            node3_qubit = pair2_data['qubits'][0]
            
        # Get node data
        middle_data = self.node_registers[middle_node]
        node1_data = self.node_registers[node1]
        node3_data = self.node_registers[node3]
        
        # Apply Bell state measurement on middle node
        middle_qc = middle_data['circuit']
        middle_qc.cx(middle_data['qreg'][middle_qubit1], middle_data['qreg'][middle_qubit2])
        middle_qc.h(middle_data['qreg'][middle_qubit1])
        
        # Measure the qubits at the middle node
        middle_qc.measure(middle_data['qreg'][middle_qubit1], middle_data['bell_result_reg'][0])
        middle_qc.measure(middle_data['qreg'][middle_qubit2], middle_data['bell_result_reg'][1])
        
        # Reset middle node qubits
        middle_data['qubit_status'][middle_qubit1] = 'idle'
        middle_data['qubit_status'][middle_qubit2] = 'idle'
        middle_data['entanglement_partners'][middle_qubit1] = None
        middle_data['entanglement_partners'][middle_qubit2] = None
        
        # Create new entanglement pair between node1 and node3
        new_pair_key = f"{min(node1, node3)}_{max(node1, node3)}"
        
        # Calculate the fidelity of the new pair (reduced due to swapping)
        new_fidelity = pair1_data['fidelity'] * pair2_data['fidelity'] * 0.9  # additional loss factor
        
        # Create the new entanglement pair
        self.entanglement_pairs[new_pair_key] = {
            'qubits': (node1_qubit, node3_qubit),
            'fidelity': new_fidelity,
            'creation_time': self.current_time
        }
        
        # Update node data
        node1_data['entanglement_partners'][node1_qubit] = (node3, node3_qubit)
        node3_data['entanglement_partners'][node3_qubit] = (node1, node1_qubit)
        node1_data['qubit_fidelity'][node1_qubit] = new_fidelity
        node3_data['qubit_fidelity'][node3_qubit] = new_fidelity
        
        # Remove old entanglement pairs
        del self.entanglement_pairs[pair1_key]
        del self.entanglement_pairs[pair2_key]
        
        logger.debug(f"Entanglement swapping applied to chain {chain} with resulting fidelity {new_fidelity:.3f}")
        return True
    
    def _measure_node_qubits(self, node):
        """Measure qubits at a node and return results."""
        node_data = self.node_registers[node]
        qc = node_data['circuit']
        qreg = node_data['qreg']
        creg = node_data['creg']
        
        # Find qubits that are in use
        active_qubits = []
        for i, status in enumerate(node_data['qubit_status']):
            if status != 'idle':
                active_qubits.append(i)
                
        if not active_qubits:
            return {}
            
        # Measure the active qubits
        for i in active_qubits:
            qc.measure(qreg[i], creg[i])
            
            # If qubit was entangled, update the partner status
            if node_data['entanglement_partners'][i] is not None:
                partner_node, partner_qubit = node_data['entanglement_partners'][i]
                partner_data = self.node_registers[partner_node]
                partner_data['entanglement_partners'][partner_qubit] = None
                partner_data['qubit_status'][partner_qubit] = 'idle'
                
                # Remove from entanglement pairs
                pair_key = f"{min(node, partner_node)}_{max(node, partner_node)}"
                if pair_key in self.entanglement_pairs:
                    del self.entanglement_pairs[pair_key]
            
            # Update qubit status
            node_data['qubit_status'][i] = 'idle'
            node_data['entanglement_partners'][i] = None
        
        # Execute measurements (simulate with random outcomes weighted by fidelity)
        results = {}
        for i in active_qubits:
            fidelity = node_data['qubit_fidelity'][i]
            # Higher fidelity means higher chance of measuring correct state (0)
            result = 0 if np.random.random() < fidelity else 1
            results[i] = result
            
        return results
    
    def _reset_node_qubits(self, node):
        """Reset all qubits at a node to |0⟩ state."""
        node_data = self.node_registers[node]
        qc = node_data['circuit']
        qreg = node_data['qreg']
        
        # Reset all qubits
        for i in range(len(qreg)):
            qc.reset(qreg[i])
            
            # If qubit was entangled, update the partner status
            if node_data['entanglement_partners'][i] is not None:
                partner_node, partner_qubit = node_data['entanglement_partners'][i]
                partner_data = self.node_registers[partner_node]
                partner_data['entanglement_partners'][partner_qubit] = None
                partner_data['qubit_status'][partner_qubit] = 'idle'
                
                # Remove from entanglement pairs
                pair_key = f"{min(node, partner_node)}_{max(node, partner_node)}"
                if pair_key in self.entanglement_pairs:
                    del self.entanglement_pairs[pair_key]
            
            # Update qubit status
            node_data['qubit_status'][i] = 'idle'
            node_data['entanglement_partners'][i] = None
            node_data['qubit_fidelity'][i] = 1.0  # Reset to perfect fidelity
        
        logger.debug(f"Reset all qubits in node {node}")
        return True
    
    def _select_node_by_priority(self):
        """Select a node for operations based on priority metrics."""
        node_priorities = {}
        
        for node, node_data in self.node_registers.items():
            # Priority based on average fidelity (lower is higher priority)
            avg_fidelity = np.mean(node_data['qubit_fidelity'])
            
            # Priority based on connectivity (higher is higher priority)
            connectivity = len(self.network_topology[node])
            
            # Priority based on current entanglement (more is higher priority)
            entanglements = sum(1 for partner in node_data['entanglement_partners'] if partner is not None)
            
            # Calculate overall priority score
            priority_score = (1 - avg_fidelity) * 0.5 + (connectivity / self.num_nodes) * 0.3 + (entanglements / self.num_qubits_per_node) * 0.2
            node_priorities[node] = priority_score
            
        # Select node with highest priority
        selected_node = max(node_priorities, key=node_priorities.get)
        return selected_node
    
    def _get_observation(self):
        """Get the current observation of the environment state."""
        # Create channel fidelity matrix (N x N)
        channel_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        # Fill in entanglement fidelities
        for pair_key, pair_data in self.entanglement_pairs.items():
            node1, node2 = map(int, pair_key.split('_'))
            channel_matrix[node1, node2] = pair_data['fidelity']
            channel_matrix[node2, node1] = pair_data['fidelity']
            
        # Calculate average qubit coherence for each node
        node_coherence = np.zeros(self.num_nodes)
        for node, node_data in self.node_registers.items():
            node_coherence[node] = np.mean(node_data['qubit_fidelity'])
            
        # Calculate success probabilities for each node's channels
        success_probs = np.zeros(self.num_nodes)
        for node in self.nodes:
            channel_success_probs = []
            for target, channel in self.network_topology[node].items():
                channel_success_probs.append(channel.calculate_success_probability(self.current_time))
            success_probs[node] = np.mean(channel_success_probs) if channel_success_probs else 0.0
            
        # Calculate node congestion (proportion of qubits in use)
        node_congestion = np.zeros(self.num_nodes)
        for node, node_data in self.node_registers.items():
            in_use = sum(1 for status in node_data['qubit_status'] if status != 'idle')
            node_congestion[node] = in_use / self.num_qubits_per_node
            
        # Flatten channel matrix and combine with other metrics
        flat_channel = channel_matrix.flatten()
        combined_obs = np.concatenate([
            flat_channel,
            node_coherence,
            success_probs,
            node_congestion
        ])
        
        return combined_obs
    
    def _calculate_reward(self, action_executed):
        """Calculate reward based on the state after action execution."""
        # Base reward components
        fidelity_reward = 0
        entanglement_reward = 0
        action_penalty = 0
        
        # Calculate average entanglement fidelity
        if self.entanglement_pairs:
            avg_fidelity = np.mean([data['fidelity'] for data in self.entanglement_pairs.values()])
            fidelity_reward = avg_fidelity * 10  # Scale up the reward for high fidelity
            
            # Extra reward for high fidelity
            if avg_fidelity > 0.9:
                fidelity_reward += 5
                
            # Record fidelity history
            self.metrics['fidelity_history'].append(avg_fidelity)
        else:
            self.metrics['fidelity_history'].append(0)
            
        # Reward for number of entanglement pairs
        entanglement_count = len(self.entanglement_pairs)
        entanglement_reward = entanglement_count * 2
        
        # Penalty for ineffective actions
        if "failed" in action_executed or "unknown_action" in action_executed:
            action_penalty = -1
            
        # Extra reward for end-to-end entanglement (distant nodes)
        for pair_key in self.entanglement_pairs:
            node1, node2 = map(int, pair_key.split('_'))
            # Calculate network distance (minimum number of hops)
            path = nx.shortest_path(self.network_graph, node1, node2)
            path_length = len(path) - 1  # Number of edges
            
            if path_length > 1:  # Nodes aren't direct neighbors
                entanglement_reward += path_length * 3  # Higher reward for longer-distance entanglement
                
        # Calculate total reward
        total_reward = fidelity_reward + entanglement_reward - action_penalty
        
        return total_reward
        
    def _check_success_condition(self):
        """Check if the environment has reached a success condition."""
        # Success if we have high-fidelity entanglement between distant nodes
        for pair_key, pair_data in self.entanglement_pairs.items():
            node1, node2 = map(int, pair_key.split('_'))
            path = nx.shortest_path(self.network_graph, node1, node2)
            path_length = len(path) - 1  # Number of edges
            
            if path_length > 1 and pair_data['fidelity'] > 0.9:
                logger.info(f"Success condition met: High-fidelity entanglement between distant nodes {node1} and {node2}")
                return True
                
        return False
        
    def render(self, mode='human'):
        """Render the current state of the environment."""
        if mode == 'human':
            # Create a visualization of the network
            plt.figure(figsize=(12, 8))
            
            # Draw the network topology
            pos = nx.spring_layout(self.network_graph)
            
            # Draw edges with varying thickness based on channel quality
            edge_widths = []
            for u, v in self.network_graph.edges():
                channel = self.network_topology[u][v] if v in self.network_topology[u] else self.network_topology[v][u]
                success_prob = channel.calculate_success_probability(self.current_time)
                edge_widths.append(success_prob * 3)
                
            # Draw nodes with size reflecting qubit coherence
            node_sizes = []
            node_colors = []
            for node in self.network_graph.nodes():
                avg_coherence = np.mean(self.node_registers[node]['qubit_fidelity'])
                node_sizes.append(300 + 1000 * avg_coherence)
                node_colors.append(avg_coherence)
                
            # Draw the graph
            edges = nx.draw_networkx_edges(
                self.network_graph, pos,
                width=edge_widths,
                alpha=0.7,
                edge_color='grey'
            )
            
            nodes = nx.draw_networkx_nodes(
                self.network_graph, pos,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=plt.cm.viridis,
                alpha=0.8
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                self.network_graph, pos,
                font_size=12,
                font_color='black'
            )
            
            # Add entanglement pairs as green edges
            entanglement_edges = []
            for pair_key in self.entanglement_pairs:
                node1, node2 = map(int, pair_key.split('_'))
                entanglement_edges.append((node1, node2))
                
            nx.draw_networkx_edges(
                self.network_graph, pos,
                edgelist=entanglement_edges,
                width=2.5,
                alpha=0.8,
                edge_color='green'
            )
            
            plt.colorbar(nodes, label='Qubit Coherence')
            plt.title(f'Quantum Repeater Network (Time Slot: {self.time_slot})')
            plt.axis('off')
            
            # Plot metrics in second subplot
            if self.metrics['fidelity_history']:
                plt.figure(figsize=(12, 4))
                plt.plot(self.metrics['fidelity_history'], 'r-', label='Avg Fidelity')
                if self.metrics['reward_history']:
                    # Plot on secondary y-axis
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    ax2.plot(self.metrics['reward_history'], 'b--', label='Reward')
                    ax2.set_ylabel('Reward')
                    ax2.legend(loc='upper right')
                    
                plt.xlabel('Time Step')
                plt.ylabel('Fidelity')
                plt.title('Quantum Repeater Performance')
                plt.legend(loc='upper left')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        elif mode == 'rgb_array':
            # Return an RGB array for video rendering (not implemented)
            return np.zeros((400, 600, 3), dtype=np.uint8)
            
        return None

def create_advanced_noise_model(T1=50000, T2=70000, gate_error_prob=0.001, readout_error_prob=0.03):
    """
    Create a more realistic noise model based on current quantum hardware parameters.
    
    Args:
        T1: Amplitude damping time (in dt units)
        T2: Dephasing time (in dt units)
        gate_error_prob: Base gate error probability
        readout_error_prob: Measurement readout error probability
        
    Returns:
        NoiseModel object with thermal relaxation and gate errors
    """
    noise_model = NoiseModel()
    
    # Add thermal relaxation error to gates
    # Single-qubit gate error
    error_1q = thermal_relaxation_error(
        T1, T2, gate_error_prob, 
        time_unit='dt', 
        standard_gates=True
    )
    
    # Two-qubit gate error (CNOT takes longer, so higher error)
    error_2q = thermal_relaxation_error(
        T1, T2, gate_error_prob * 5, 
        time_unit='dt', 
        standard_gates=True
    )
    
    # Add errors to gates
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cy', 'cz', 'cp', 'cu1', 'cu2', 'cu3'])
    
    # Add readout error
    p_meas = readout_error_prob 
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    
    # Add reset error
    reset_error = pauli_error([('X', p_meas/2), ('I', 1 - p_meas/2)])
    noise_model.add_all_qubit_quantum_error(reset_error, "reset")
    
    return noise_model

def create_network_topology(num_nodes=6, topology_type='ring', random_seed=None):
    """
    Create a network topology with quantum channels.
    
    Args:
        num_nodes: Number of nodes in the network
        topology_type: 'ring', 'star', 'mesh', 'line', or 'random'
        random_seed: Seed for random topology generation
        
    Returns:
        Dictionary representing the network topology
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    topology = {i: {} for i in range(num_nodes)}
    
    if topology_type == 'ring':
        # Ring topology (each node connected to two neighbors)
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            distance = 5 + np.random.uniform(-1, 1)  # Random distance around 5 units
            topology[i][next_node] = QuantumChannel(distance=distance)
            
    elif topology_type == 'star':
        # Star topology (all nodes connected to central node)
        central_node = 0
        for i in range(1, num_nodes):
            distance = 5 + np.random.uniform(-2, 2)
            topology[central_node][i] = QuantumChannel(distance=distance)
            
    elif topology_type == 'mesh':
        # Mesh topology (partially connected)
        for i in range(num_nodes):
            # Each node has ~50% chance of connecting to another node
            for j in range(i+1, num_nodes):
                if np.random.random() < 0.5:
                    distance = 5 + np.random.uniform(-2, 2)
                    topology[i][j] = QuantumChannel(distance=distance)
                    
    elif topology_type == 'line':
        # Line topology (nodes in a line)
        for i in range(num_nodes - 1):
            distance = 5 + np.random.uniform(-1, 1)
            topology[i][i+1] = QuantumChannel(distance=distance)
            
    elif topology_type == 'random':
        # Random topology (Erdős–Rényi model)
        p_connection = 0.4  # Probability of edge between any two nodes
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if np.random.random() < p_connection:
                    distance = 5 + np.random.uniform(-2, 2)
                    topology[i][j] = QuantumChannel(distance=distance)
    
    # Check if graph is connected, if not add minimum connections to make it connected
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
        
    for i, neighbors in topology.items():
        for j in neighbors:
            G.add_edge(i, j)
            
    # If not connected, add edges to make it connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        while len(components) > 1:
            # Connect two components
            comp1 = list(components[0])[0]
            comp2 = list(components[1])[0]
            distance = 7.5  # Slightly longer distance for these added connections
            topology[comp1][comp2] = QuantumChannel(distance=distance)
            
            # Update graph and components
            G.add_edge(comp1, comp2)
            components = list(nx.connected_components(G))
    
    return topology

def initialize_environment(topology_type='mesh', num_nodes=6, advanced_features=True, use_continuous_actions=False):
    """Initialize the quantum repeater environment with specified parameters."""
    # Create network topology
    network_topology = create_network_topology(num_nodes=num_nodes, topology_type=topology_type)
    
    # Create noise model
    noise_model = create_advanced_noise_model()
    
    # Create environment
    env = AdvancedQuantumRepeaterEnv(
        network_topology=network_topology,
        noise_model=noise_model,
        max_steps=200,
        use_continuous_actions=use_continuous_actions,
        advanced_features=advanced_features
    )
    
    return env

def train_advanced_rl_agent(env, 
                           model_type='PPO', 
                           total_timesteps=500000, 
                           eval_freq=10000, 
                           save_path='./quantum_repeater_models'):
    """
    Train a reinforcement learning agent for the quantum repeater environment.
    
    Args:
        env: The environment to train on
        model_type: The type of RL algorithm to use ('PPO', 'A2C', 'SAC', 'TD3')
        total_timesteps: Total number of timesteps to train for
        eval_freq: How often to evaluate the agent
        save_path: Directory to save trained models
        
    Returns:
        Trained RL model
    """
    # Create directories
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(f"{save_path}/best_model").mkdir(parents=True, exist_ok=True)
    Path(f"{save_path}/logs").mkdir(parents=True, exist_ok=True)
    
    # Create vectorized environment
    env = Monitor(env, f"{save_path}/logs")
    env = DummyVecEnv([lambda: env])  
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=f"{save_path}/checkpoints",
        name_prefix="quantum_repeater_model"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{save_path}/best_model",
        log_path=f"{save_path}/logs",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Select RL algorithm based on model_type
    policy_kwargs = dict(
        features_extractor_class=CustomNetworkFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256, dict(pi=[128, 64], vf=[128, 64])]
    )
    
    if model_type == 'PPO':
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=f"{save_path}/tensorboard/"
        )
    elif model_type == 'A2C':
        model = A2C(
            "MlpPolicy", 
            env, 
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=7e-4,
            gamma=0.99,
            tensorboard_log=f"{save_path}/tensorboard/"
        )
    elif model_type == 'SAC':
        if isinstance(env.action_space, spaces.Discrete):
            logger.warning("SAC is designed for continuous action spaces, using PPO instead")
            model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                       tensorboard_log=f"{save_path}/tensorboard/")
        else:
            # Add noise to exploration
            action_noise = NormalActionNoise(
                mean=np.zeros(env.action_space.shape[0]),
                sigma=0.1 * np.ones(env.action_space.shape[0])
            )
            model = SAC(
                "MlpPolicy", 
                env, 
                verbose=1,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                learning_starts=1000,
                action_noise=action_noise,
                tensorboard_log=f"{save_path}/tensorboard/"
            )
    elif model_type == 'TD3':
        if isinstance(env.action_space, spaces.Discrete):
            logger.warning("TD3 is designed for continuous action spaces, using PPO instead")
            model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                       tensorboard_log=f"{save_path}/tensorboard/")
        else:
            action_noise = NormalActionNoise(
                mean=np.zeros(env.action_space.shape[0]),
                sigma=0.1 * np.ones(env.action_space.shape[0])
            )
            model = TD3(
                "MlpPolicy", 
                env, 
                verbose=1,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=100,
                tau=0.005,
                gamma=0.99,
                learning_starts=1000,
                action_noise=action_noise,
                tensorboard_log=f"{save_path}/tensorboard/"
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    logger.info(f"Starting training with {model_type} for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    # Save the final model
    model.save(f"{save_path}/final_{model_type}_quantum_repeater")
    
    # Save the normalized environment parameters
    env.save(f"{save_path}/vec_normalize.pkl")
    
    return model

def evaluate_model(model, env, n_eval_episodes=10):
    """Evaluate a trained model on the environment."""
    logger.info(f"Evaluating model on {n_eval_episodes} episodes")
    
    episode_rewards = []
    episode_lengths = []
    episode_fidelities = []
    
    for i in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        fidelities = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_step += 1
            
            # Record fidelity
            if 'average_fidelity' in info:
                fidelities.append(info['average_fidelity'])
                
            if episode_step >= 200:  # Safety limit
                break
                
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        episode_fidelities.append(np.mean(fidelities) if fidelities else 0)
        
        logger.info(f"Episode {i+1}: Reward = {episode_reward:.2f}, Length = {episode_step}, Avg Fidelity = {np.mean(fidelities) if fidelities else 0:.3f}")
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_fidelity = np.mean(episode_fidelities)
    
    logger.info(f"Evaluation results over {n_eval_episodes} episodes:")
    logger.info(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"Mean episode length: {mean_length:.1f}")
    logger.info(f"Mean fidelity: {mean_fidelity:.3f}")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.bar(['Mean Reward'], [mean_reward], yerr=[std_reward])
    plt.ylabel('Reward')
    
    plt.subplot(132)
    plt.bar(['Mean Episode Length'], [mean_length])
    plt.ylabel('Steps')
    
    plt.subplot(133)
    plt.bar(['Mean Fidelity'], [mean_fidelity])
    plt.ylabel('Fidelity')
    
    plt.tight_layout()
    plt.savefig('./evaluation_results.png', dpi=300)
    plt.show()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'mean_fidelity': mean_fidelity
    }

def visualize_quantum_network(env, include_metrics=True):
    """Create a comprehensive visualization of the quantum repeater network."""
    plt.figure(figsize=(15, 10))
    
    # Create layout
    if include_metrics:
        gs = plt.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])  # Network graph
        ax2 = plt.subplot(gs[0, 1])  # Node status
        ax3 = plt.subplot(gs[1, :])  # Performance metrics
    else:
        gs = plt.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])  # Network graph
        ax2 = plt.subplot(gs[0, 1])  # Node status
    
    # Draw network graph on ax1
    plt.sca(ax1)
    pos = nx.spring_layout(env.network_graph, seed=42)
    
    # Draw edges with width based on channel quality
    edge_widths = []
    edge_labels = {}
    for u, v in env.network_graph.edges():
        channel = env.network_topology[u][v] if v in env.network_topology[u] else env.network_topology[v][u]
        success_prob = channel.calculate_success_probability(env.current_time)
        edge_widths.append(success_prob * 3)
        edge_labels[(u, v)] = f"{channel.distance:.1f}km"
    
    # Draw nodes with size based on qubit coherence
    node_sizes = []
    node_colors = []
    for node in env.network_graph.nodes():
        avg_coherence = np.mean(env.node_registers[node]['qubit_fidelity'])
        node_sizes.append(300 + 700 * avg_coherence)
        node_colors.append(avg_coherence)
    
    # Draw base topology
    nx.draw_networkx_edges(
        env.network_graph, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='grey',
        ax=ax1
    )
    
    nodes = nx.draw_networkx_nodes(
        env.network_graph, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.8,
        ax=ax1
    )
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(
        env.network_graph, pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax1
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        env.network_graph, pos,
        font_size=10,
        font_color='black',
        ax=ax1
    )
    
    # Draw entanglement links as green edges
    entanglement_edges = []
    for pair_key in env.entanglement_pairs:
        node1, node2 = map(int, pair_key.split('_'))
        entanglement_edges.append((node1, node2))
    
    if entanglement_edges:
        nx.draw_networkx_edges(
            env.network_graph, pos,
            edgelist=entanglement_edges,
            width=2.5,
            alpha=0.8,
            edge_color='green',
            ax=ax1
        )
    
    plt.colorbar(nodes, ax=ax1, label='Qubit Coherence')
    ax1.set_title('Quantum Repeater Network Topology')
    ax1.axis('off')
    
    # Draw node status on ax2
    plt.sca(ax2)
    node_data = []
    for node in range(env.num_nodes):
        node_reg = env.node_registers[node]
        
        # Count qubit statuses
        idle_count = node_reg['qubit_status'].count('idle')
        entangled_count = node_reg['qubit_status'].count('entangled')
        in_use_count = len(node_reg['qubit_status']) - idle_count - entangled_count
        
        # Calculate average fidelity
        avg_fidelity = np.mean(node_reg['qubit_fidelity'])
        
        node_data.append({
            'Node': f"Node {node}",
            'Idle': idle_count,
            'Entangled': entangled_count,
            'In Use': in_use_count,
            'Avg Fidelity': avg_fidelity
        })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(node_data)
    
    # Create stacked bar chart for qubit statuses
    df_statuses = df[['Node', 'Idle', 'Entangled', 'In Use']]
    df_statuses.set_index('Node').plot(kind='bar', stacked=True, ax=ax2, 
                                      colormap='Set2', alpha=0.7)
    
    # Add fidelity as text
    for i, node in enumerate(df['Node']):
        ax2.text(i, env.num_qubits_per_node + 0.5, 
                f"Fid: {df.loc[i, 'Avg Fidelity']:.2f}", 
                ha='center', va='bottom', fontsize=9)
    
    ax2.set_title('Node Status')
    ax2.set_xlabel('')
    ax2.set_ylabel('Number of Qubits')
    ax2.legend(loc='upper right')
    
    # Add metrics plot if requested
    if include_metrics and env.metrics['fidelity_history']:
        plt.sca(ax3)
        
        # Plot fidelity history
        steps = np.arange(len(env.metrics['fidelity_history']))
        ax3.plot(steps, env.metrics['fidelity_history'], 'r-', label='Avg Fidelity')
        
        # Plot reward history on secondary y-axis if available
        if env.metrics['reward_history']:
            ax_reward = ax3.twinx()
            ax_reward.plot(steps, env.metrics['reward_history'], 'b--', label='Reward')
            ax_reward.set_ylabel('Reward')
            ax_reward.legend(loc='upper right')
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Fidelity')
        ax3.set_title('Performance Metrics')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./quantum_network_visualization.png', dpi=300)
    plt.show()
    
    return

def advanced_error_correction(qc, qubits, syndrome_regs, code_type='Steane'):
    """
    Apply advanced quantum error correction codes.
    
    Args:
        qc: Quantum circuit
        qubits: Qubit registers to apply error correction to
        syndrome_regs: Classical registers for syndrome measurements
        code_type: Type of error correction code ('Steane', '5-qubit', 'Surface')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if code_type == 'Steane':
            # Steane [[7,1,3]] code - can correct single qubit errors
            if len(qubits) < 7:
                logger.warning(f"Steane code requires 7 qubits, but only {len(qubits)} provided")
                return False
                
            # Create ancilla qubits for syndrome extraction
            q_main = qubits[:7]  # First 7 qubits for the code
            
            # Encode logical qubit
            # Apply Hadamard to create superposition
            qc.h(q_main[0])
            
            # Create GHZ-like state
            for i in range(1, 7):
                qc.cx(q_main[0], q_main[i])
            
            # Apply Hadamards for proper encoding
            for i in range(7):
                qc.h(q_main[i])
            
            # Add Steane stabilizers
            # X stabilizers
            for i, j, k in [(0, 2, 4), (0, 2, 6), (0, 4, 6), (2, 4, 6)]:
                qc.cx(q_main[i], syndrome_regs[0])
                qc.cx(q_main[j], syndrome_regs[0])
                qc.cx(q_main[k], syndrome_regs[0])
                
            # Z stabilizers
            for i, j, k in [(0, 1, 3), (0, 1, 5), (0, 3, 5), (1, 3, 5)]:
                qc.cz(q_main[i], syndrome_regs[1])
                qc.cz(q_main[j], syndrome_regs[1])
                qc.cz(q_main[k], syndrome_regs[1])
            
            return True
            
        elif code_type == '5-qubit':
            # Perfect 5-qubit code
            if len(qubits) < 5:
                logger.warning(f"5-qubit code requires 5 qubits, but only {len(qubits)} provided")
                return False
                
            q_main = qubits[:5]
            
            # Encode logical qubit
            qc.h(q_main[1])
            qc.h(q_main[2])
            qc.h(q_main[3])
            qc.h(q_main[4])
            
            qc.cx(q_main[0], q_main[1])
            qc.cx(q_main[0], q_main[2])
            qc.cx(q_main[0], q_main[3])
            qc.cx(q_main[0], q_main[4])
            
            qc.h(q_main[0])
            qc.h(q_main[2])
            qc.h(q_main[3])
            
            qc.cx(q_main[0], q_main[1])
            qc.cx(q_main[2], q_main[3])
            qc.cx(q_main[1], q_main[4])
            
            # Measure syndrome
            for i in range(4):
                qc.measure(q_main[i+1], syndrome_regs[i])
                
            return True
            
        elif code_type == 'Surface':
            # Simple surface code fragment
            if len(qubits) < 9:
                logger.warning(f"Surface code requires at least 9 qubits, but only {len(qubits)} provided")
                return False
                
            q_main = qubits[:9]
            
            # Initialize in superposition
            for i in range(9):
                qc.h(q_main[i])
            
            # Apply plaquette operators (simplified version)
            for i, j, k, l in [(0, 1, 3, 4), (1, 2, 4, 5), (3, 4, 6, 7), (4, 5, 7, 8)]:
                qc.cx(q_main[i], q_main[j])
                qc.cx(q_main[k], q_main[l])
                qc.cx(q_main[i], q_main[k])
                qc.cx(q_main[j], q_main[l])
                
            # Extract syndrome by measuring stabilizers
            for i in range(min(4, len(syndrome_regs))):
                qc.measure(q_main[i*2+1], syndrome_regs[i])
                
            return True
            
        else:
            logger.warning(f"Unknown error correction code type: {code_type}")
            return False
            
    except Exception as e:
        logger.error(f"Error in advanced error correction: {str(e)}")
        return False

def quantum_repeater_simulation():
    """Run a complete quantum repeater simulation with visualization and evaluation."""
    logger.info("Starting quantum repeater simulation")
    
    # Create the environment
    env = initialize_environment(
        topology_type='mesh',
        num_nodes=6,
        advanced_features=True,
        use_continuous_actions=True
    )
    
    # Visualize initial network
    visualize_quantum_network(env, include_metrics=False)
    
    # Train the agent
    model = train_advanced_rl_agent(
        env,
        model_type='SAC',  # Using SAC for continuous action space
        total_timesteps=200000,
        eval_freq=10000
    )
    
    # Evaluate the trained model
    evaluation_results = evaluate_model(model, env, n_eval_episodes=10)
    
    # Run a demonstration episode
    obs = env.reset()
    env.render()
    
    done = False
    for step in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if step % 10 == 0:  # Render every 10 steps
            env.render()
            
        if done:
            break
    
    # Final network visualization
    visualize_quantum_network(env, include_metrics=True)
    
    logger.info("Quantum repeater simulation completed")
    return model, env, evaluation_results

def main():
    """Main function to run the quantum repeater simulation."""
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Configure Qiskit settings
    qiskit.IBMQ.save_account('MY_API_TOKEN', overwrite=True)
    
    # Run simulation
    model, env, results = quantum_repeater_simulation()
    
    # Save results
    with open('./simulation_results.json', 'w') as f:
        json.dump({
            'mean_reward': float(results['mean_reward']),
            'std_reward': float(results['std_reward']),
            'mean_episode_length': float(results['mean_length']),
            'mean_fidelity': float(results['mean_fidelity'])
        }, f, indent=2)
    
    logger.info(f"Final results: Mean reward = {results['mean_reward']:.2f}, Mean fidelity = {results['mean_fidelity']:.3f}")
    
    # Create final plots
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(env.metrics['fidelity_history'], 'g-')
    plt.title('Entanglement Fidelity Over Time')
    plt.xlabel('Step')
    plt.ylabel('Average Fidelity')
    
    plt.subplot(132)
    plt.plot(env.metrics['reward_history'], 'b-')
    plt.title('Reward Over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    
    plt.subplot(133)
    end_to_end = len(env.metrics['end_to_end_entanglement_time']) > 0
    if end_to_end:
        plt.hist(env.metrics['end_to_end_entanglement_time'], bins=10)
        plt.title('End-to-End Entanglement Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, "No end-to-end entanglement achieved", 
                ha='center', va='center')
        plt.title('End-to-End Entanglement')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./final_results.png', dpi=300)
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()
