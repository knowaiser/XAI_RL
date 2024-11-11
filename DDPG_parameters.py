# GPT Suggestions: 

# # Hyperparameters for DDPG Agent
# LR_ACTOR = 0.0001  # Learning rate for actor network
# LR_CRITIC = 0.001  # Learning rate for critic network
# GAMMA = 0.99       # Discount factor
# TAU = 0.001        # Soft update parameter
# BUFFER_SIZE = 100000  # Reduced buffer size
# BATCH_SIZE = 64    # Batch size for sampling from replay buffer
# LAYER1_SIZE = 400  # First hidden layer size
# LAYER2_SIZE = 300  # Second hidden layer size
# INPUT_DIMENSION = 108  # Input dimension (adjust as needed)

# BUFFER_SIZE = int(3e7)  # replay buffer size
BUFFER_SIZE = 10000000  # Reduced from 30000000 to 100000
BATCH_SIZE = 64         # minibatch size
# GAMMA = 1               # discount factor
GAMMA = 0.99               # discount factor
TAU = 1e-3              # for soft update of target parameters (not specified in the RL paper)
LR_ACTOR = 1e-4         # learning rate of the actor (alpha)
LR_CRITIC = 1e-4        # learning rate of the critic (beta)
WEIGHT_DECAY = 0        # L2 weight decay
LAYER1_SIZE = 1000      # Layer 1 of the critic\actor
LAYER2_SIZE = 1000      # Layer 2 of the critic\actor

# Environment related constants
# INPUT_DIMENSION = 27
# INPUT_DIMENSION = 36 # Including IMU data
INPUT_DIMENSION = 108  # For adaptive mode (st-2, st-1, st) = (36*3)


#DDPG Optimization (hyper)parameters
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 2e6
ACTION_STD_INIT = 0.2
TEST_TIMESTEPS = 5e4
DDPG_CHECKPOINT_DIR = 'preTrained_models/ddpg/'
POLICY_CLIP = 0.2

