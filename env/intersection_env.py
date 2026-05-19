import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficIntersectionEnv(gym.Env):
    """
    Custom Environment for Autonomous Traffic Flow Optimizer.
    Simulates a 4-way intersection where an RL agent controls the traffic lights.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        # Action space: 4 discrete phases
        # 0: N-S green, 1: E-W green, 2: N-S left-turn, 3: E-W left-turn
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 9 floats
        # [queue_0, queue_1, queue_2, queue_3, phase_0, phase_1, phase_2, phase_3, time_elapsed]
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )
        
        # Simulation timing parameters
        self.timestep_duration = 5.0
        self.max_steps = 500
        self.current_step = 0
        
        # Traffic dynamics — Poisson arrival rates per lane per step
        self.arrival_rates = np.array([0.8, 0.8, 0.3, 0.3])
        
        # Max cars cleared per green step
        self.departure_rate = 3.0
        
        # Reward parameters
        self.penalty_threshold = 15.0
        self.penalty_value = 50.0
        self.clear_bonus = 10.0
        
        # State variables
        self.queue_lengths = np.zeros(4, dtype=np.float32)
        self.active_phase = 0
        self.time_since_change = 0.0

    def _get_obs(self):
        phase_one_hot = np.zeros(4, dtype=np.float32)
        phase_one_hot[self.active_phase] = 1.0
        return np.concatenate((
            self.queue_lengths,
            phase_one_hot,
            [self.time_since_change]
        )).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.queue_lengths = np.zeros(4, dtype=np.float32)
        self.active_phase = 0
        self.time_since_change = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        if action == self.active_phase:
            self.time_since_change += self.timestep_duration
        else:
            self.active_phase = action
            self.time_since_change = 0.0
            
        arrivals = self.np_random.poisson(self.arrival_rates)
        self.queue_lengths += arrivals
        
        cleared_cars = min(self.queue_lengths[self.active_phase], self.departure_rate)
        self.queue_lengths[self.active_phase] -= cleared_cars
        
        reward = -np.sum(self.queue_lengths)
        
        if np.max(self.queue_lengths) > self.penalty_threshold:
            reward -= self.penalty_value
            
        if self.queue_lengths[self.active_phase] == 0 and cleared_cars > 0:
            reward += self.clear_bonus
            
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            "total_queue": np.sum(self.queue_lengths),
            "max_queue": np.max(self.queue_lengths),
            "throughput": cleared_cars
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        print(f"--- Step: {self.current_step} ---")
        print(f"Queues [N-S, E-W, N-S L, E-W L]: {self.queue_lengths}")
        print(f"Active Phase: {self.active_phase} | Time since change: {self.time_since_change}s")