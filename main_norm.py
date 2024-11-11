
from utils import *

import os
import torch

# from environment_waypoints_fskip import SimEnv
# from environment_waypoints_fskip_2 import SimEnv
# from environment_waypoints_fskip_PID import SimEnv
# from environment_waypoints_fskip_SHAP import SimEnv
# from environment_waypoints_fskip_rpm import SimEnv
# from environment_waypoints_fskip_collision import SimEnv
# from environment_waypoints_fskip_collision_mh import SimEnv
# from environment_waypoints_fskip_circles import SimEnv ## this one to avoid circles
# from environment_waypoints_circles_imu import SimEnv
# from environment_waypoints_circles_imu_mh import SimEnv
from environment_waypoints_circles_imu_mh_norm import SimEnv

from DDPG.ddpg_torch import DDPGAgent
from config import env_params, action_map
# from config_SHAP import env_params
from DDPG_parameters import *
from settings import *

def run_original():
    try:
        
        state_dim = INPUT_DIMENSION
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = "cpu"
        num_actions = 1 # a single continuous action
        episodes = 10000

        #replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        #model = DQN(num_actions, state_dim, in_channels, device)

        # this only works if you have a model in your weights folder. Replace this by that file
        #model.load('weights/model_ep_4400')

        # set to True if you want to run with pygame
        env = SimEnv(visuals=True, **env_params)

        # Initialize DDPG agent
        agent = DDPGAgent(alpha=LR_ACTOR, beta=LR_CRITIC, 
                          input_dims=INPUT_DIMENSION, 
                          tau=TAU, env=None, gamma=GAMMA,
                          n_actions=1, max_size=BUFFER_SIZE, 
                          layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE,
                          batch_size=BATCH_SIZE)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(ep, train_mode=False, shap_flag=False)
            env.reset()
    finally:
        env.reset()
        env.quit()

def run():
    try:
        episodes = 10000
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            try:
                #env.generate_episode(ep, mode='eval')
                env.generate_episode(ep, mode='resume', shap_flag = False)
            except ValueError as e:
                print(f"Encountered error during episode generation: {e}")
                print("Ending current episode and starting a new one...")
                env.end_episode()  # Ensure resources are cleaned up and the episode is properly closed
                continue  # Skip the rest of this loop iteration to start a new episode
            finally:
                env.reset()  # Reset the environment state for the next episode
    finally:
        env.reset()
        env.quit()


if __name__ == "__main__":
    run()
