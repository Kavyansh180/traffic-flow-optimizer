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
        # We use spaces.Box to define continuous float values
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(9,), 
            dtype=np.float32
        )
        
        # Simulation timing parameters
        self.timestep_duration = 5.0  # 5 simulated seconds per step
        self.max_steps = 500          # Episode length
        self.current_step = 0
        
        # Traffic dynamics parameters
        # Arrival rates (lambda for Poisson dist) - average cars arriving per 5 seconds per lane
        self.arrival_rates = np.array([0.8, 0.8, 0.3, 0.3]) 
        
        # Departure rate - maximum cars that can clear the intersection in 5 seconds on a green light
        self.departure_rate = 3.0 
        
        # Reward function parameters
        self.penalty_threshold = 15.0  # Queue limit before applying starvation penalty
        self.penalty_value = 50.0      # Extra negative reward for starvation
        self.clear_bonus = 10.0        # Positive reward when a queue fully clears
        
        # State variables
        self.queue_lengths = np.zeros(4, dtype=np.float32)
        self.active_phase = 0
        self.time_since_change = 0.0

    def _get_obs(self):
        """Constructs the observation vector from the current state."""
        # Create one-hot encoding for the active phase (4 elements)
        phase_one_hot = np.zeros(4, dtype=np.float32)
        phase_one_hot[self.active_phase] = 1.0
        
        # Concatenate queues (4), one-hot phase (4), and time elapsed (1) -> 9 floats total
        return np.concatenate((
            self.queue_lengths,
            phase_one_hot,
            [self.time_since_change]
        )).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state and returns the initial observation."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.queue_lengths = np.zeros(4, dtype=np.float32)
        self.active_phase = 0
        self.time_since_change = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        """Advances the simulation by one timestep (5 seconds)."""
        self.current_step += 1
        
        # 1. Update phase and elapsed time
        if action == self.active_phase:
            # Signal stayed the same
            self.time_since_change += self.timestep_duration
        else:
            # Signal changed to a new phase
            self.active_phase = action
            self.time_since_change = 0.0
            
        # 2. Simulate traffic arrivals (Poisson distribution per lane)
        arrivals = self.np_random.poisson(self.arrival_rates)
        self.queue_lengths += arrivals
        
        # 3. Simulate traffic departures (only from the active phase lane)
        # We can only clear as many cars as there are in the queue (or max departure rate)
        cleared_cars = min(self.queue_lengths[self.active_phase], self.departure_rate)
        self.queue_lengths[self.active_phase] -= cleared_cars
        
        # 4. Calculate Reward based on defined logic
        # Primary: negative sum of all queues (minimize total waiting cars)
        reward = -np.sum(self.queue_lengths)
        
        # Penalty: check for lane starvation
        if np.max(self.queue_lengths) > self.penalty_threshold:
            reward -= self.penalty_value
            
        # Bonus: small positive reward if active queue was fully cleared this step
        if self.queue_lengths[self.active_phase] == 0 and cleared_cars > 0:
            reward += self.clear_bonus
            
        # 5. Check truncation condition (episode ends at 500 steps)
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Pass data into info dict for baseline comparison and metric tracking later
        info = {
            "total_queue": np.sum(self.queue_lengths),
            "max_queue": np.max(self.queue_lengths),
            "throughput": cleared_cars
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """Optional simple console render to visualize state during local testing."""
        print(f"--- Step: {self.current_step} ---")
        print(f"Queues [N-S, E-W, N-S L, E-W L]: {self.queue_lengths}")
        print(f"Active Phase: {self.active_phase} | Time since change: {self.time_since_change}s")