"""
RELight Configuration File
Contains all hyperparameters for the RELight DRL traffic control agent
"""

class Config:
    """
    Configuration class for RELight algorithm hyperparameters.
    
    RELight (Reweighting to mitigate cOnservative Light control) uses:
    - Ensemble of N independent Q-networks per agent
    - Random M-subset MIN aggregation for conservative target selection
    - Error-threshold based subset resampling to ensure stable Bellman updates
    - Training-time reward scaling (reward / REWARD_SCALE_DIVISOR) to stabilize Q-value magnitudes
    
    This base Config handles the ensemble + update algorithm.
    Subclass ABSTConfig adds spatial-temporal layers (ObservationEmbedding + GCN-MHA).
    """
    
    # ==================== Ensemble Configuration ====================
    # Number of Q-networks in the ensemble pool
    N_NETWORKS = 10                                 # Ensemble size: 10 independent Q-heads per agent
    
    # Number of networks randomly sampled for target Q-value calculation
    M_SUBSET_SIZE = 4                               # Sample M=4 heads; use MIN over this subset
    
    # Error threshold for subset resampling
    # If update error exceeds this, a new subset is sampled (RELight algorithm)
    ERROR_THRESHOLD = 150.0                          # Base MSE threshold before resampling
    
    # Training-time reward scaling divisor used in Bellman targets:
    # scaled_reward = reward / REWARD_SCALE_DIVISOR
    # Raw SUMO rewards can be 0-50 (queues), after scaling become 0-0.05 (numerically stable)
    REWARD_SCALE_DIVISOR = 1000.0

    # Dynamic error threshold: scale threshold based on mean target magnitude
    # Effective threshold = max(ERROR_THRESHOLD, RELATIVE_ERROR_THRESHOLD_RATIO * max(|target|, MIN_TARGET_SCALE))
    USE_DYNAMIC_ERROR_THRESHOLD = True
    RELATIVE_ERROR_THRESHOLD_RATIO = 0.05           # 5% of mean target magnitude
    MIN_TARGET_SCALE_FOR_THRESHOLD = 1.0            # Floor for relative scaling
    
    # Maximum number of resampling attempts per update
    MAX_RESAMPLING_ATTEMPTS = 5
    
    # ==================== Learning Parameters ====================
    # Learning rate for network optimization
    # Applied to all optimizers (front-end + per-head ensemble optimizers)
    LEARNING_RATE = 0.0001
    
    # Discount factor for future rewards (gamma in Bellman equation)
    # Higher gamma (closer to 1) → longer planning horizon
    GAMMA = 0.95
    
    # Batch size for experience replay sampling
    # Larger batch → more stable gradients; smaller batch → faster updates
    BATCH_SIZE = 64
    
    # ==================== Memory Configuration ====================
    # Maximum size of replay buffer
    MEMORY_SIZE = 20000
    
    # Minimum experiences before training starts
    MIN_MEMORY_SIZE = 4000
    
    # ==================== Exploration Parameters ====================
    # Epsilon-greedy exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.997                              # Applied once per episode via Trainer.train()
    
    # ==================== Training Configuration ====================
    # Number of training episodes
    NUM_EPISODES = 1000
    
    # Maximum steps per episode
    MAX_STEPS_PER_EPISODE = 3600  # 1 hours of simulation at 1s per step
    
    # Frequency of network updates (in steps)
    UPDATE_FREQUENCY = 10
    
    # Frequency of target network sync (if using target networks)
    TARGET_UPDATE_FREQUENCY = 100
    
    # ==================== Environment Configuration ====================
    # NOTE: These image-based parameters are LEGACY (from original RELight CNN).
    # ABSTLight uses FLAT multi-agent observations (see ABSTConfig.OBS_DIM).
    # These are kept for backward compatibility but NOT used by ABSTLight.
    STATE_HEIGHT = 84                               # Legacy: image height (unused in ABSTLight)
    STATE_WIDTH = 84                                # Legacy: image width (unused in ABSTLight)
    STATE_CHANNELS = 1                              # Legacy: grayscale channels (unused in ABSTLight)
    
    # Number of traffic signal phases / discrete actions per intersection
    # Modify if your SUMO network has different phase schemes (e.g., 8-phase intersections)
    NUM_ACTIONS = 4                                 # Standard 4-phase: N, E, S, W
    
    # ==================== SUMO Configuration ====================
    # Path to SUMO configuration file
    SUMO_CONFIG_PATH = "sumo_files/osm.sumocfg"
    
    # Use GUI or command-line SUMO
    USE_GUI = False
    
    # SUMO simulation step length (seconds)
    SUMO_STEP_LENGTH = 1.0
    
    # Yellow phase duration (seconds)
    YELLOW_PHASE_DURATION = 3
    
    # Minimum green phase duration (seconds)
    MIN_GREEN_DURATION = 10

    # Evaluation-only safety fallback: max seconds one phase can be held before
    # forcing a phase advance to guarantee minimum network connectivity.
    SAFETY_FALLBACK_MAX_PHASE_HOLD = 60
    
    # ==================== Reward Configuration ====================
    # Hybrid normalized reward weights
    # Formula: reward = (throughput_weight * throughput)
    #                 + (pressure_weight * pressure) 
    #                 + (insertion_penalty_weight * insertion)
    #                 + (phase_change_weight * phase_change_cost)
    REWARD_THROUGHPUT_WEIGHT = 1.0                  # Destination arrival events per step (positive signal)
    REWARD_PRESSURE_WEIGHT = -0.5                   # Queue buildup penalty (normalized & capped)
    REWARD_INSERTION_PENALTY_WEIGHT = -1.0          # Pending vehicles globally (normalized & clipped)
    REWARD_PHASE_CHANGE = -0.1                      # Phase transition cost (reduces flickering)

    # Reward normalization and clipping controls
    PRESSURE_NORMALIZATION_CAPACITY = 50.0          # Divider for pressure signal (vehicles)
    INSERTION_PENALTY_CLIP = -30.0                  # Max negative clip per step for insertion penalty

    # Observation normalization factors for flat obs encoding
    OBS_QUEUE_NORMALIZATION = 50.0                  # Divider for queue_length in get_flat_obs
    OBS_WAITING_TIME_NORMALIZATION = 300.0         # Divider for waiting_time in get_flat_obs

    # Legacy reward weights kept for backward compatibility
    REWARD_WAITING_TIME_WEIGHT = -0.25
    REWARD_QUEUE_LENGTH_WEIGHT = -0.25
    REWARD_DELAY_WEIGHT = -0.5
    
    # ==================== Logging Configuration ====================
    # Directory for saving models
    MODEL_SAVE_DIR = "model"
    
    # Directory for saving logs
    LOG_DIR = "logs"
    
    # Save model every N episodes
    SAVE_FREQUENCY = 10
    
    # Verbose logging
    VERBOSE = True
    
    # ==================== Device Configuration ====================
    # Use CUDA if available
    USE_CUDA = True
    
    @classmethod
    def to_dict(cls):
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if (
                not key.startswith('_')
                and not callable(value)
                and not isinstance(value, (classmethod, staticmethod))
            )
        }
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("=" * 60)
        print("RELight Configuration")
        print("=" * 60)
        for key, value in cls.to_dict().items():
            print(f"{key:.<40} {value}")
        print("=" * 60)


class ABSTConfig(Config):
    """
    Configuration for ABSTLight (Adaptive Behavioral Spatial-Temporal Light).

    Inherits all RELight parameters from Config and adds architectural
    hyper-parameters for the spatial-temporal front-end (Layers 1 & 2).

    Observation & Feature Processing
    ---------------------------------
    ABSTLight processes flat traffic observations (NOT images) via a 3-layer pipeline:
    
    Layer 1: ObservationEmbedding
      - Input: Stacked raw observations per intersection
      - Raw obs per intersection: ~20 features (8 queue counts + 8 waiting times + 4 phase one-hot)
      - Frame stacking (T=4): Concatenate last 4 raw obs → 80-dim vector (20 * 4)
      - Output: Embedded feature vectors (80 → 64 dim, via ReLU)
    
    Layer 2: GCN-MHA Stack
      - Input: (B, N_agents, 64) embedded features
      - Multi-head attention across graph topology (road network structure)
      - 2 stacked layers, 4 attention heads each
      - Output: Refined spatial-temporal features (B, N_agents, 64)
    
    Layer 3: Ensemble Q-Heads (RELight)
      - Input: 64-dim spatial features per agent
      - N=10 independent Q-networks (ensemble)
      - Each head: FC(64→512) + ReLU + FC(512→num_actions)
      - RELight training with M=4 subset MIN aggregation
    
    Configuration Parameters
    ------------------------
    """

    # ==================== Observation Embedding (Layer 1) ====================
    # Observation frame stacking window (Mnih et al., 2015): 
    # Temporally stack T consecutive raw observations to capture short-term dynamics
    OBS_STACK_SIZE = 4
    
    # Raw observation dimension (before stacking)
    # Breakdown for 4-approach × 2-lane intersection:
    #   8 lane vehicle queue counts
    #   8 lane mean waiting times (seconds)
    #   4 current signal phase (one-hot encoding)
    #   = 20 base features per intersection
    OBS_DIM_RAW = 20
    
    # Final observation dimension after frame stacking (passed to agent):
    # OBS_DIM = OBS_DIM_RAW * OBS_STACK_SIZE = 20 * 4 = 80
    OBS_DIM = OBS_DIM_RAW * OBS_STACK_SIZE        # Effective 80-dim input

    # Hidden state dimension after the embedding layer (Layer 1 output)
    # Must be divisible by NUM_HEADS for multi-head attention
    EMBED_DIM = 64

    # ==================== GCN-MHA Stack (Layer 2) ============================
    # Number of parallel attention heads in multi-head attention
    # Condition: EMBED_DIM % NUM_HEADS must equal 0
    NUM_HEADS = 4                                   # Each head gets 64/4 = 16 dims

    # Number of stacked GCNMHALayer instances (depth of the graph attention stack)
    # L layers means receptive field depth of ~L hops in the road network graph
    NUM_GCN_LAYERS = 2

    # Dropout probability applied inside each GCNMHALayer (after attention + residual)
    GCN_DROPOUT = 0.1

    # ==================== Q-Value Head (Layer 3 / RELight ensemble) ==========
    # Width of the hidden fully-connected layer inside each QValueHead
    # Mirrors original RELight BaseDQN's fc1 size (512) for architectural parity
    Q_HEAD_HIDDEN_DIM = 512

    # ==================== Multi-Agent Configuration ===========================
    # Number of traffic-light intersections (SUMO TLS entities) controlled by the agent.
    # The actual number of agents is determined by TrafficEnvironment at runtime
    # by reading TLS definitions from the SUMO network file, BUT this config value
    # determines the output layer dimensions. They MUST match for correct operation.
    # Verify: count TLS in your .net.xml and set NUM_AGENTS accordingly.
    NUM_AGENTS = 8

    @classmethod
    def print_config(cls):
        """Print all ABSTLight configuration parameters."""
        print("=" * 60)
        print("ABSTLight Configuration")
        print("=" * 60)
        for key, value in cls.to_dict().items():
            print(f"{key:.<40} {value}")
        print("=" * 60)
