import glob
import os
import sys
import csv
import numpy as np
import torch
import shap
import pandas as pd
shap.initjs()

from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

import logging
from torch.utils.tensorboard import SummaryWriter


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import pickle

from synch_mode import CarlaSyncMode
from controllers import PIDLongitudinalController
from utils import *
from settings import *
from DDPG.ddpg_torch import *
from DDPG_parameters import *
from config_SHAP import *

random.seed(78)
# np.random.seed
# torch.manual_seed

class SimEnv(object):
    def __init__(self, visuals=True, target_speed = 18, max_iter = 4000, start_buffer = 10, train_freq = 1,
        action_freq = 4, save_freq = 200, start_ep = 0, max_dist_from_waypoint = 20, mode = 'resume', shap_flag = True) -> None:
        

        self.visuals = visuals
        if self.visuals:
            self._initiate_visuals()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.load_world('Town02_Opt')
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        

        # self.spawn_points = self.world.get_map().get_spawn_points()
        # self.spawn_waypoints = self.world.get_map().generate_waypoints(5.0)
        # self.spawn_points = [waypoint.transform for waypoint in self.spawn_waypoints]
        self.spawn_points = self.generate_custom_spawn_points(distance = 10)
        # self.start_point = random.choice(self.spawn_points)

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.nissan.patrol')
        #self.vehicle_blueprint = self.blueprint_library.find('vehicle.tesla.model3')

        # input these later on as arguments
        self.global_t = 0 # global timestep
        self.target_speed = target_speed # km/h 
        self.max_iter = max_iter
        self.start_buffer = start_buffer
        self.train_freq = train_freq
        self.save_freq = save_freq

        
        self.total_rewards = 0
        self.average_rewards_list = []


        # Additional attributes for yaw calculation
        self.previous_yaw = None
        self.previous_yaw_rate = None
        self.delta_time = 1.0 / 30  # Assuming 30 FPS, adjust based on your simulation setup

        # Initiate states
        self.initial_observations = []
        self.navigation_obs = []

        # max_size: memory size for the replay buffer
        #

        # Initialize DDPG agent
        self.agent = DDPGAgent(alpha=LR_ACTOR, beta=LR_CRITIC, 
                          input_dims=INPUT_DIMENSION, 
                          tau=TAU, env=None, gamma=GAMMA,
                          n_actions=1, max_size=BUFFER_SIZE, 
                          layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE,
                          batch_size=BATCH_SIZE)
        
        # Initialize state vectors for normalization
        self.state_data = []  # List to collect state data for normalization
        # self.scaler = MinMaxScaler(feature_range=(0, 1))  # Min-max scaler for normalization
        # self.scaler_fitted = False  # Flag to check if the scaler is fitted

        # # Initialize logging and TensorBoard writer
        # self.logger = logging.getLogger(self.__class__.__name__)
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # self.writer = SummaryWriter(log_dir='./logs/simenv')
        log_dir = './logs/simenv'
        log_file = os.path.join(log_dir, 'simulation.log')

        csv_file = './csv/simenv.csv'
        csv_file_norm = './csv/simenv_norm.csv'
        csv_termination_log = './csv/termination_log.csv'
        csv_euc_waypoints_log = './csv/euc_waypoints_log.csv'
        csv_reward_log = './csv/reward_log.csv'
        csv_metrics_log = './csv/metrics_log.csv'
        csv_trajectory_log = './csv/trajectory_log.csv'
        csv_waypoints_log = './csv/waypoints_log.csv'
        csv_shap_log = './csv/shap_log.csv'

        self.logger = initialize_logger(log_file)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.csv_writer, self.csv_file_handle = initialize_csv_writer(csv_file)
        self.csv_writer_norm, self.csv_file_handle_norm = initialize_csv_writer(csv_file_norm)
        self.termination_log, self.csv_file_handle_termination = initialize_termination_log(csv_termination_log)
        self.euc_waypoints_log, self.csv_file_handle_euc_waypoints = initialize_euc_log(csv_euc_waypoints_log)
        self.reward_log, self.csv_file_handle_reward = initialize_reward_log(csv_reward_log)
        self.metrics_log, self.csv_file_handle_metrics = initialize_metrics_log(csv_metrics_log)
        self.trajectory_log, self.csv_file_handle_trajectory = initialize_trajectory_log(csv_trajectory_log)
        self.waypoints_log, self.csv_file_handle_waypoints = initialize_waypoints_log(csv_waypoints_log)
        self.shap_log, self.csv_file_handle_shap = initialize_shap_log(csv_shap_log)
        chkpt_dir = config['checkpoint_dir']
        # checkpoint_filename = config['checkpoint']  # name could be 'actor' or 'target_actor'
        # self.checkpoint_file = os.path.join(chkpt_dir, checkpoint_filename)

        self.mode = mode


        if self.mode == 'resume':
            checkpoint_file = get_latest_checkpoint(chkpt_dir)
            resume_episode = self.agent.resume_models(checkpoint_file)
            self.start_ep = resume_episode
        else:
            self.start_ep = start_ep
        

        if self.mode == 'eval':
            checkpoint_file = get_latest_checkpoint(chkpt_dir)
            self.agent.load_models(checkpoint_file)

        self.max_dist_from_waypoint = max_dist_from_waypoint
        self.start_train = self.start_ep + self.start_buffer
        

    def _initiate_visuals(self):
        pygame.init()

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()
    
    def create_actors(self):
        self.actor_list = []

        # Select a new random start point at the beginning of each episode
        self.start_point = random.choice(self.spawn_points)

        # spawn vehicle at random location
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.start_point)
        # vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

        self.camera_rgb = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        self.camera_rgb_vis = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_vis)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)

        self.speed_controller = PIDLongitudinalController(self.vehicle)

        # INITIATE STATE VALUES
        self.throttle = float(0.0)
        self.previous_steer = float(0.0)
        self.speed = float(0.0)
        self.velocity_x = float(0.0)
        self.velocity_y = float(0.0)
        self.velocity_z = float(0.0)
        self.distance_from_center = float(0.0)
        self.angle = float(0.0)
        self.center_lane_deviation = 0.0
        self.distance_covered = 0.0
        self.euclidean_dist_list = []
        self.dev_angle_array = []
        self.engine_rpm = float(0.0)
        self.closest_waypoint_index = None
        self.next_closest_waypoint_index = None
        self.route_waypoints = []
        self.steering_history = []
        self.previous_deviation_angles = None
        self.yaw_acceleration = 0.0
    
    def reset(self):
        for actor in self.actor_list:
            if actor.is_alive: # this line is newly added
                actor.destroy()
           
    def end_episode(self):
        # If you have any actors (vehicles, sensors, etc.) that need to be destroyed
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list.clear()  # Clear the list of actors for the next episode

        # Reset environment-specific variables
        self.route_waypoints = []  # Assuming this is where waypoints are stored
        self.current_waypoint_index = 0  # Reset index or similar variables
        # self.total_reward = 0  # If you're tracking rewards

        # Reset any other state or variables related to the episode
        # For example, if you're tracking episode length or time
        self.episode_length = 0

        # Add any additional cleanup or state resetting you require here
        # This is also where you might reset the simulation environment if needed
        if self.world is not None:
            self.reset()  # Assuming your simulation environment has a reset method

        # Log the episode's end if necessary
        print(f"Episode ended. Preparing for a new episode...")

   
    def generate_episode(self, ep, mode='resume', shap_flag = True):
        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.collision_sensor, fps=30) as sync_mode:
            
            # Set random weather for this episode
            # set_random_weather(self.world)
            # set_overcast_weather(self.world)

            counter = 0
            episode_reward = 0

            # Frame Skipping
            self.frame_skip = 4  # Number of frames to skip: Skip 4 frames, act on every 5th frame
            self.frame_count = 0  # Counter to keep track of skipped frames
            last_action = None  # Current action to apply

            # Steering Penalty
            last_steering = 0
            current_steering = 0

            # SHAP-specific setup
            reason_file = None
            shap_reason_counter = 0
            shap_reason_duration = 90
            current_reason_text = ""
            if shap_flag:
                
                # Load the background data from the CSV file
                background_data_file_path = 'C:/Users/g201901650/Desktop/ddpg/JabrahRL/self_driving_agent/Experiment-Compare Town02 to Town04/SHAP Experiment/background_data_sample.csv'
                background_data_df = pd.read_csv(background_data_file_path)

                # Convert the DataFrame to a numpy array
                background_data = background_data_df.values

                # Define the model prediction function
                def model_predict(data_as_numpy):
                    # Convert the numpy array to a torch tensor
                    data_as_tensor = torch.tensor(data_as_numpy, dtype=torch.float).to(self.agent.actor.device)
                    
                    # Set the model to evaluation mode
                    self.agent.actor.eval()

                    # Forward pass through the model
                    with torch.no_grad():
                        output = self.agent.actor(data_as_tensor)

                    # Convert the output tensor back to a numpy array
                    return output.cpu().numpy()
                
                # Initialize the SHAP KernelExplainer with the custom model prediction function and background data
                explainer = shap.KernelExplainer(model_predict, background_data)

                # SHAP calculation control
                shap_frame_skip = 30  # Calculate SHAP values every 30 frames (~ every 1 second)
                shap_frame_count = 0

                # Feature mapping for explanation
                feature_names = [
                    "Euclidean Distance 1", "Euclidean Distance 2", "Euclidean Distance 3",
                    "Euclidean Distance 4", "Euclidean Distance 5", "Euclidean Distance 6",
                    "Euclidean Distance 7", "Euclidean Distance 8", "Euclidean Distance 9",
                    "Euclidean Distance 10", 
                    "Deviation Angle 1", "Deviation Angle 2", "Deviation Angle 3",
                    "Deviation Angle 4", "Deviation Angle 5", "Deviation Angle 6",
                    "Deviation Angle 7", "Deviation Angle 8", "Deviation Angle 9",
                    "Deviation Angle 10",
                    "Velocity", "VelocityX", "VelocityY", "VelocityZ",
                    "Engine RPM", "Distance From Center", "Yaw Angle"
                ]

                # Initialize variables to keep SHAP reason on screen longer
                shap_reason_duration = 90  # Number of frames to keep the SHAP reason displayed (~3 seconds)
                shap_reason_counter = 0
                current_reason_text = ""

            # Generate route UPDATED 
            self.generate_route_3()


            # TO DRAW WAYPOINTS
            # for w in self.route_waypoints:
            #     self.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
            #                                 color=carla.Color(r=0, g=0, b=255), life_time=120.0,
            #                                 persistent_lines=True)
                
            for i, waypoint in enumerate(self.route_waypoints):  # Just as an example, adjust the range as needed
                print(f"Waypoint {i}: Location = {waypoint.transform.location}")
                location_x = waypoint.transform.location.x
                location_y = waypoint.transform.location.y
                location_z = waypoint.transform.location.z
                self.waypoints_log.writerow([ep, i, location_x, location_y, location_z])
            # for i, waypoint in enumerate(self.route_waypoints):
            #     next_waypoints = waypoint.next(5.0)  # This gets the next waypoints within 5 meters
            #     print(f"Waypoint {i} has {len(next_waypoints)} next waypoints.")

            # # Save graph of plotted points as route.png
            # x_graph = [p.transform.location.x for p in self.route_waypoints]
            # y_graph = [p.transform.location.y for p in self.route_waypoints]
            # plt.clf()  # Clear the figure to ensure no old data is plotted
            # plt.plot(x_graph, y_graph, marker = 'o')
            # # plt.savefig(f"route_ep_{ep}.png")
            # plt.savefig(f"route_ep_{ep}.jpg", quality = 45) # MAX qulaity = 95

            returned_data = sync_mode.tick(timeout=2.0)
            # Assuming the first item in returned_data is the snapshot
            snapshot = returned_data[0]
            # Assuming the last item is the collision data
            collision = returned_data[-1]
            # Retrieving image_rgb_vis, which should be the second-to-last item
            image_rgb_vis = returned_data[-2]

            # destroy if there is no data
            #if snapshot is None or image_rgb is None:
            if snapshot is None:
                print("No data, skipping episode")
                self.reset()
                return None


            try:
                # We will be tracking waypoints in the route and switch to next one wen we get close to current one
                self.curr_wp = 5

                # Capture the initial state
                next_state = self.capture_states(ep)

                if next_state is None:
                    print("Preparing for a new episode...")
                    self.reset()  # Clean up and log the end of the episode, if needed
                    return None  # Break out of the loop to finish the current episode

                
                while True: # simulation loop
                    if self.visuals: # Check the visuals and maintain a consistent frame rate
                        if should_quit(): # utils.py, checks pygame for quit events
                            return
                        self.clock.tick_busy_loop(30) # does not advance the simulation, only controls the fps rate

                    # Advance simulation
                    state = next_state
                    counter +=1
                    self.global_t +=1

                    # Apply model to get steering angle
                    if self.frame_count % self.frame_skip == 0:
                        action = self.agent.choose_action(state)
                        last_action = action
                    else:
                        action = last_action
                    
                    steer_nd = action
                    steer = steer_nd.item()
                    current_steering = steer

                    control = self.speed_controller.run_step(self.target_speed)
                    control.steer = steer
                    self.vehicle.apply_control(control)
                    self.steering_history.append(control.steer)

                    # Code to log evaluation metrics
                    # csv_writer.writerow(['Episode', 'YawAcceleration', 'SteeringAngle', 'AngleDeviation', 'DistanceDeviation'])
                    YawAcceleration = self.yaw_acceleration
                    SteeringAngle = control.steer
                    AngleDeviation = next_state[26]
                    DistanceDeviation = next_state[25]
                    self.metrics_log.writerow([ep, YawAcceleration, SteeringAngle, 
                                               AngleDeviation, DistanceDeviation])



                    fps = round(1.0 / snapshot.timestamp.delta_seconds)

                    # Increment frame_count for frame skip
                    self.frame_count += 1

                    # NEXT TICK
                    returned_data = sync_mode.tick(timeout=2.0)
                    # Assuming the first item in returned_data is the snapshot
                    snapshot = returned_data[0]
                    # Assuming the last item is the collision data
                    collision = returned_data[-1]
                    # Retrieving image_rgb_vis, which should be the second-to-last item
                    image_rgb_vis = returned_data[-2]

                    # Capture the next state
                    next_state = self.capture_states(ep)
                    
                    if next_state is None:
                        print("Preparing for a new episode...")
                        self.reset()  # Clean up and log the end of the episode, if needed
                        return None  # Break out of the loop to finish the current episode
                    
                                    # SHAP calculation every nth frame, only if shap_flag is True
                    if shap_flag:
                        shap_frame_count += 1
                        if shap_frame_count % shap_frame_skip == 0:
                            # Ensure state is a numpy array
                            if isinstance(state, torch.Tensor):
                                state_np = state.cpu().numpy()
                            else:
                                state_np = state

                            # Compute SHAP values
                            shap_values = explainer.shap_values(state_np)

                            # Combine SHAP values if necessary (in case of multi-output)
                            shap_values_combined = np.stack(shap_values, axis=0).mean(axis=0)

                            # Identify the dominant feature
                            max_shap_value = max(shap_values_combined, key=abs)
                            max_shap_index = shap_values_combined.tolist().index(max_shap_value)
                            dominant_feature = feature_names[max_shap_index]
                            current_time = time.strftime("%H:%M:%S", time.localtime())
                            current_reason_text = f"At [{current_time}] Main Reason for Steering: {dominant_feature} (SHAP Value: {max_shap_value:.3f})"

                            # Reset the duration counter
                            shap_reason_counter = 0

                            # Log the reason
                            self.shap_log.writerow([ep, self.closest_waypoint_index, self.global_t, dominant_feature, max_shap_value])

                    velocity = self.vehicle.get_velocity()

                    # Print the closest waypoint index
                    print("Episode waypoint index:", self.closest_waypoint_index)

                    # OLD REWARD CALL
                    cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, 
                                                    self.route_waypoints[self.closest_waypoint_index], collision)
                    # reward = reward_value(cos_yaw_diff, dist, collision)

                    # UPDATED REWARD
                    # reward = calculate_reward_1(self.angle, self.distance_from_center, self.velocity, yaw_acceleration)
                    
                    # reward = calculate_reward_A2(velocity.x, steer, collision, self.angle)
                    # reward = calculate_reward_A2(self.speed, steer, collision, self.angle)
                    reward = calculate_reward_3(ep, self.vehicle, self.route_waypoints[self.closest_waypoint_index],
                                                self.route_waypoints[self.next_closest_waypoint_index],
                                                current_steering, last_steering, self.steering_history, self.reward_log)
                    
                    # reward = calculate_reward_4(ep, self.vehicle, self.route_waypoints[self.closest_waypoint_index],
                    #                             self.route_waypoints[self.next_closest_waypoint_index],
                    #                             current_steering, last_steering, self.reward_log)


                    #if snapshot is None or image_rgb is None:
                    if snapshot is None:
                        print("Process ended here")
                        break


                    done = 1 if collision else 0

                    episode_reward += reward
                    self.total_rewards += reward

                    # Logging state before normalization
                    # self.logger.info(f"Pre-Norm State: {state}")
                    self.csv_writer.writerow([time.time(), ep, self.closest_waypoint_index,
                                            *state, steer, reward, episode_reward, self.total_rewards])
                    self.csv_writer_norm.writerow([time.time(), ep, self.closest_waypoint_index,
                                            *next_state, steer, reward, episode_reward, self.total_rewards])

                    ###################
                    # Normalization logic
                    ###################
                    # if ep <= 100:
                    #     # Collect state data for normalization
                    #     self.state_data.append(state)
                    # else: 
                    #     if not self.scaler_fitted:
                    #         # Fit the scaler to the collected state data
                    #         self.scaler.fit(self.state_data)
                    #         self.scaler_fitted = True # to ensure the one-time fitting of the scaler

                    # Check if the scaler is fitted, then normalize the state
                    # if self.scaler_fitted:
                    #     state = self.normalize_state(state)
                    #     self.logger.info(f"Post-Norm State: {state}")
                    #     # self.csv_writer_norm.writerow([time.time(), ep, *self.euclidean_dist_list, *self.dev_angle_array,
                    #     #   self.velocity, self.velocity_x, self.velocity_y, self.velocity_z,
                    #     #   self.engine_rpm, self.distance_from_center, self.angle])
                    #     self.csv_writer_norm.writerow([time.time(), ep, *state, steer, reward, self.total_rewards])



                    #CHECK THIS
                    #replay_buffer.add(state, action, next_state, reward, done)
                    self.agent.remember(state, action, reward, next_state, done)

                    if mode in ['train', 'resume']:
                        # Train after a number of episodes > start_train and do not train with every timestep
                        # if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                        if ep > 100 and (self.global_t % self.train_freq) == 0:
                            #model.train(replay_buffer)
                            self.agent.learn()

                    # Draw the display.
                    if self.visuals:
                        draw_image(self.display, image_rgb_vis)
                        self.display.blit(
                            self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                            (8, 10))
                        self.display.blit(
                            self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                            (8, 28))
                        self.display.blit(
                            self.font.render('% 5.4f Steering Angle' % steer, True, (255, 255, 255)), # CHECK THIS
                            (8, 46))
                        self.display.blit(
                            self.font.render('% 5.4f Speed' % self.speed, True, (255, 255, 255)), # CHECK THIS
                            (8, 64))
                        self.display.blit(
                            self.font.render('Ep: {} + {}'.format(ep, self.start_ep), True, (255, 255, 255)),
                            (8, 82))
                        # Increment the counter and display the reason text if within duration
                        if shap_reason_counter < shap_reason_duration:
                            shap_reason_counter += 1
                            if self.visuals:
                                self.display.blit(
                                    self.font.render(current_reason_text, True, (255, 255, 255)),
                                    (8, 100))
                                
                        # if shap_flag and shap_frame_count % shap_frame_skip == 0:
                        #     self.display.blit(
                        #         self.font.render(reason_text, True, (255, 255, 255)),
                        #         (8, 100))
                        pygame.display.flip()

                    # if collision == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                    #     print("Episode {} processed".format(ep), counter)
                    #     break
                    
                    if collision == 1:
                        print(f"Episode {ep} ended due to a collision after {counter} iterations.")
                        self.termination_reason = 'Collision'
                        self.termination_log.writerow([ep, self.termination_reason, episode_reward])
                        break
                    elif counter >= self.max_iter:
                        print(f"Episode {ep} reached the maximum iteration limit of {self.max_iter}.")
                        self.termination_reason = 'Maximum iterations'
                        self.termination_log.writerow([ep, self.termination_reason, episode_reward])
                        break
                    elif dist > self.max_dist_from_waypoint:
                        print(f"Episode {ep} ended because the vehicle was {dist} meters away from the closest waypoint.")
                        self.termination_reason = 'Far from waypoint'
                        self.termination_log.writerow([ep, self.termination_reason, episode_reward])
                        break
                    elif self.curr_wp >= len(self.route_waypoints) - 1:
                        print(f"Episode {ep} reached the end of the route.")
                        self.termination_reason = 'End of route'
                        self.termination_log.writerow([ep, self.termination_reason, episode_reward])
                        break

                if mode in ['train', 'resume']:
                    self.agent.noise.sigma = max(0.1, self.agent.noise.sigma * 0.995) # NEW
                    if ep % self.save_freq == 0 and ep > 0:
                        self.save(ep)
                    

                # Logging
                # self.termination_reason = 'Successfull termination'
                # self.termination_log.writerow([ep, self.termination_reason, episode_reward])
                print("Episode {} total rewards".format(ep), self.total_rewards)
                self.logger.info(f"Episode: {ep}, Total Reward: {self.total_rewards}")
                self.writer.add_scalar('Rewards/Total Reward', self.total_rewards, ep)

                # Steering Penalty
                last_steering = current_steering


            except KeyboardInterrupt:
                print("Simulation stopped by the user.")
                sys.exit()  # This will terminate the program

    def save(self, ep):
        # if ep % self.save_freq == 0 and ep > self.start_ep:
        if ep % self.save_freq == 0 and ep > 0:
            avg_reward = self.total_rewards/self.save_freq
            self.average_rewards_list.append(avg_reward)
            self.total_rewards = 0

            #model.save('weights/model_ep_{}'.format(ep))
            total_episodes = ep + self.start_ep
            self.agent.save_models(total_episodes)
            print("Saved model with average reward =", avg_reward)
    
    def quit(self):
        pygame.quit()
        # End logging
        self.writer.close()
        self.agent.writer.close()
        self.csv_file_handle.close()
        self.csv_file_handle_norm.close()
        self.csv_file_handle_termination.close()
        self.csv_file_handle_euc_waypoints.close()
        self.csv_file_handle_reward.close()
        self.csv_file_handle_metrics.close()
        self.csv_file_handle_trajectory.close()
        self.csv_file_handle_waypoints.close()
        self.csv_file_handle_shap.close()

    # -------------------------------------------------
    # Estimating Engine RPM                          |
    # -------------------------------------------------
    def estimate_engine_rpm(self):
        # Get the control input for the vehicle
        control = self.vehicle.get_control()
        physics = self.vehicle.get_physics_control()

        # Calculate engine RPM
        engine_rpm = physics.max_rpm * control.throttle

        # Throttle values: [0.0, 1.0]. Default is 0.0.
        # Max rpm values: For most passenger cars, a max RPM in the range of 6,000 to 8,000 RPM.

        # Check if the vehicle is in gear (not in neutral or reverse)
        if control.gear > 0:
            # Retrieve the gear information for the current gear
            gear = physics.forward_gears[control.gear]

            # Adjust engine RPM based on the gear ratio of the current gear
            engine_rpm *= gear.ratio

        return engine_rpm

    # -------------------------------------------------
    # Estimating Engine RPM                          |
    # -------------------------------------------------
    def estimate_wheel_rpm(self, velocity):
        # Get the control input for the vehicle
        physics = self.vehicle.get_physics_control()

        # Retrieve radius
        radius_cm = physics.wheels[0].radius
        radius_m = radius_cm/100

        # Calculate the wheel circumference in meters
        wheel_circumference_m = 2 * math.pi * radius_m

        # Convert the vehicle's velocity vector to speed in m/s
        speed_m_s = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
        # Calculate wheel rotation in revolutions per second (RPS)
        wheel_rps = speed_m_s / wheel_circumference_m

        # Convert RPS to revolutions per minute (RPM)
        wheel_rpm = wheel_rps * 60

        return wheel_rpm
        
    def distance_to_line(self, A, B, p):
        # Distance is calculated in meter
        # This method calculates the perpendicular distance from a point p 
        # to a line defined by two points A and B in a 3D space
        num   = np.linalg.norm(np.cross(B - A, A - p)) # calculate cross product 
        denom = np.linalg.norm(B - A) # Euclidean distance between points A and B
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom
    
    # -------------------------------------------------
    # Calculating Euclidean Distance List            |
    # -------------------------------------------------

    def calculate_euc_dist_no_pad(self, current_waypoint_index):
        ##########################################################
        # This version raises an error and stops the simulation when there are not enought waypoints
        ##########################################################
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) - current_waypoint_index < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")
        
        # Calculate the end index for slicing the route_waypoints list
        end_index = current_waypoint_index + num_closest_waypoints + 1

        # Retrieve the next 10 waypoints from the current waypoint
        closest_waypoints = self.route_waypoints[current_waypoint_index + 1:min(end_index, len(self.route_waypoints))]

        # Calculate the Euclidean distance to each waypoint
        euclidean_dist_list = []
        current_waypoint_vector = np.array([self.route_waypoints[current_waypoint_index].transform.location.x, 
                                        self.route_waypoints[current_waypoint_index].transform.location.y, 
                                        self.route_waypoints[current_waypoint_index].transform.location.z])
        
        for waypoint in closest_waypoints:
            # Extract the coordinates from the waypoint object and convert them into a NumPy array
            waypoint_location = np.array([waypoint.transform.location.x, waypoint.transform.location.y, 
                                          waypoint.transform.location.z])
            # Calculate the Euclidean distance from the current_waypoint to the waypoint
            distance = np.linalg.norm(current_waypoint_vector - waypoint_location)
            # Append the distance to the list
            euclidean_dist_list.append(distance)

        return euclidean_dist_list
    
    def calculate_euc_dist(self, current_waypoint_index):
        ##########################################################
        # Modified version to provide zero padding for insufficient waypoints
        ##########################################################
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # This is the fixed number of waypoints

        # Calculate the end index for slicing the route_waypoints list
        end_index = current_waypoint_index + num_closest_waypoints + 1

        # Retrieve the next waypoints from the current waypoint up to the number required
        closest_waypoints = self.route_waypoints[current_waypoint_index + 1:min(end_index, len(self.route_waypoints))]

        # Log the closest waypoints in a file
        self.euc_waypoints_log.writerow([current_waypoint_index, *closest_waypoints])

        # Calculate the Euclidean distance to each waypoint
        euclidean_dist_list = []
        current_waypoint_vector = np.array([self.route_waypoints[current_waypoint_index].transform.location.x, 
                                            self.route_waypoints[current_waypoint_index].transform.location.y, 
                                            self.route_waypoints[current_waypoint_index].transform.location.z])

        for waypoint in closest_waypoints:
            # Extract the coordinates from the waypoint object and convert them into a NumPy array
            waypoint_location = np.array([waypoint.transform.location.x, waypoint.transform.location.y, 
                                        waypoint.transform.location.z])
            # Calculate the Euclidean distance from the current_waypoint to the waypoint
            distance = np.linalg.norm(current_waypoint_vector - waypoint_location)
            # Append the distance to the list
            euclidean_dist_list.append(distance)

        # Ensure the list is always of length 10 by padding with zeros if necessary
        while len(euclidean_dist_list) < num_closest_waypoints:
            euclidean_dist_list.append(0.0)  # Append zero for missing waypoints

        return euclidean_dist_list

    
    def vector(self, v):
        # The vector method is a utility function that converts a Carla Location, Vector3D, 
        # or Rotation object to a NumPy array for easier manipulation and calculations.
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])
        
    def calculate_deviation_angle_tan_no_pad(self, current_waypoint_index):
        ##########################################################
        # This version raises an error and stops the simulation when there are not enought waypoints
        ##########################################################
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # You can adjust this as needed

        # Ensure there are at least 10 waypoints available
        if len(self.route_waypoints) - current_waypoint_index < num_closest_waypoints:
            raise ValueError("Not enough waypoints available for calculation.")
        
        # Calculate the end index for slicing the route_waypoints list
        end_index = current_waypoint_index + num_closest_waypoints + 1

        # Get the forward vector of the vehicle
        vehicle_forward_vector = self.vector(self.vehicle.get_transform().rotation.get_forward_vector())[:2]

        # Get the positions of the nearest 10 waypoints
        nearest_waypoints = self.route_waypoints[current_waypoint_index + 1:min(end_index, len(self.route_waypoints))]

        deviation_angles = []

        for waypoint in nearest_waypoints:
            # Calculate the direction vector from the vehicle to the waypoint
            #waypoint_vector = self.vector(waypoint.transform.location) - self.vector(self.vehicle.get_location())[:2]
            waypoint_vector = self.vector(waypoint.transform.location)[:2] - self.vector(self.vehicle.get_location())[:2]

            # Calculate the angle between the vehicle's forward vector and the direction to the waypoint
            deviation_angle = np.arctan2(waypoint_vector[1], waypoint_vector[0]) - np.arctan2(vehicle_forward_vector[1], vehicle_forward_vector[0])

            # Ensure the angle is within the range [-π, π]
            if deviation_angle > np.pi:
                deviation_angle -= 2 * np.pi
            elif deviation_angle <= -np.pi:
                deviation_angle += 2 * np.pi

            deviation_angles.append(deviation_angle)

        return deviation_angles
    
    def calculate_deviation_angle_tan(self, current_waypoint_index):
        ##########################################################
        # Adjusted version to handle insufficient waypoints gracefully
        ##########################################################
        # Define the number of closest waypoints to retrieve
        num_closest_waypoints = 10  # Desired number of waypoints

        available_waypoints = len(self.route_waypoints) - current_waypoint_index - 1
        waypoints_to_use = min(num_closest_waypoints, available_waypoints)

        # Calculate the end index for slicing the route_waypoints list
        end_index = current_waypoint_index + 1 + waypoints_to_use

        # Get the forward vector of the vehicle
        vehicle_forward_vector = self.vector(self.vehicle.get_transform().rotation.get_forward_vector())[:2]

        # Get the positions of the nearest available waypoints
        nearest_waypoints = self.route_waypoints[current_waypoint_index + 1:end_index]

        deviation_angles = []

        for waypoint in nearest_waypoints:
            waypoint_vector = self.vector(waypoint.transform.location)[:2] - self.vector(self.vehicle.get_location())[:2]

            # Calculate the angle between the vehicle's forward vector and the direction to the waypoint
            deviation_angle = np.arctan2(waypoint_vector[1], waypoint_vector[0]) - np.arctan2(vehicle_forward_vector[1], vehicle_forward_vector[0])

            # Ensure the angle is within the range [-π, π]
            if deviation_angle > np.pi:
                deviation_angle -= 2 * np.pi
            elif deviation_angle < -np.pi:
                deviation_angle += 2 * np.pi

            deviation_angles.append(deviation_angle)

        # Fill missing angles with a default value if there are not enough waypoints
        while len(deviation_angles) < num_closest_waypoints:
            deviation_angles.append(0)  # Append 0 or another neutral value

        return deviation_angles
    
    def calculate_deviation_angles_derivatives(self, current_waypoint_index, timestep=0.03):
        '''
        The updated calculate_deviation_angle_tan function now computes both 
        the absolute deviation angles and their rate of change (derivatives) 
        with respect to waypoints ahead of the vehicle in the CARLA simulation environment.

        Parameters:
        1- current_waypoint_index: Index of the current waypoint in the vehicle's route plan
        2- timestep: The time interval (in seconds) between consecutive calls to this function, 
                     necessary for calculating the rate of change of deviation angles accurately.
        '''
        num_closest_waypoints = 10
        available_waypoints = len(self.route_waypoints) - current_waypoint_index - 1
        waypoints_to_use = min(num_closest_waypoints, available_waypoints)
        end_index = current_waypoint_index + 1 + waypoints_to_use

        vehicle_forward_vector = self.vector(self.vehicle.get_transform().rotation.get_forward_vector())[:2]
        nearest_waypoints = self.route_waypoints[current_waypoint_index + 1:end_index]

        deviation_angles = []
        deviation_angle_derivatives = []

        for waypoint in nearest_waypoints:
            waypoint_vector = self.vector(waypoint.transform.location)[:2] - self.vector(self.vehicle.get_location())[:2]
            deviation_angle = np.arctan2(waypoint_vector[1], waypoint_vector[0]) - np.arctan2(vehicle_forward_vector[1], vehicle_forward_vector[0])

            if deviation_angle > np.pi:
                deviation_angle -= 2 * np.pi
            elif deviation_angle < -np.pi:
                deviation_angle += 2 * np.pi

            deviation_angles.append(deviation_angle)

        # Calculate derivatives if previous angles are available
        if self.previous_deviation_angles is not None:
            for current, previous in zip(deviation_angles, self.previous_deviation_angles):
                # Derivative is the change in angle over the timestep
                derivative = (current - previous) / timestep
                deviation_angle_derivatives.append(derivative)

        # Update previous angles
        self.previous_deviation_angles = deviation_angles.copy()

        # Fill in derivatives for the first calculation
        if not deviation_angle_derivatives:
            deviation_angle_derivatives = [0] * len(deviation_angles)

        return deviation_angle_derivatives

    
    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff_old(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        # If the angular difference is greater than π radians, 
        # it subtracts 2π to bring it within the range [-π, π]:
        if angle > np.pi: angle -= 2 * np.pi
        # If the angular difference is less than or equal to -π radians, 
        # it adds 2π to bring it within the range [-π, π]:
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle
    
    def angle_diff(self, wp):
        '''
        this function to find direction to selected waypoint fwd, wp_fwd

        Positive angles represent clockwise rotations from the vehicle's forward direction, 
        zero angle represents alignment with the forward direction, 
        and negative angles represent counterclockwise rotations from the forward direction.
        '''
        vehicle_pos = self.vehicle.get_transform()
        car_x = vehicle_pos.location.x
        car_y = vehicle_pos.location.y
        wp_x = wp.transform.location.x
        wp_y = wp.transform.location.y
        
        # vector to waypoint
        x = (wp_x - car_x)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
        y = (wp_y - car_y)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
        
        #car vector
        car_vector = vehicle_pos.get_forward_vector()
        angle = self.angle_between((x,y),(car_vector.x,car_vector.y))

        return angle
    
    def angle_between(self, v1, v2):
        return np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])

    
    #######################################################
    # For Yaw Acceleration Estimation
    #######################################################

    def calculate_yaw_rate(self, current_yaw):
        if self.previous_yaw is None:
            self.previous_yaw = current_yaw
            return 0  # Yaw rate is 0 for the first measurement

        yaw_rate = (current_yaw - self.previous_yaw) / self.delta_time
        self.previous_yaw = current_yaw  # Update for next iteration
        return yaw_rate

    def calculate_yaw_acceleration(self, current_yaw_rate):
        if self.previous_yaw_rate is None:
            self.previous_yaw_rate = current_yaw_rate
            return 0  # Yaw acceleration is 0 for the first measurement

        yaw_acceleration = (current_yaw_rate - self.previous_yaw_rate) / self.delta_time
        self.previous_yaw_rate = current_yaw_rate  # Update for next iteration
        return yaw_acceleration
    
    
    #######################################################
    # Capture States
    #######################################################
    def capture_states(self, episode):

        # Location of the car
        self.location = self.vehicle.get_location()
        #waypoint = self.world.get_map().get_waypoint(self.location, project_to_road=True, 
        #    lane_type=carla.LaneType.Driving)
        # Record vehicle location and write it to the csv file
        location_x = self.location.x
        location_y = self.location.y
        self.trajectory_log.writerow([episode, location_x, location_y])

        # Determine the current waypoint index
        # closest_waypoint_index = self.determine_current_waypoint_index()
        next_wp, next_next_wp, next_wp_index, next_next_wp_index = self.get_next_two_waypoints_and_indices()

        # CHECK if next_wp or next_next_wp is None, terminate episode
        if next_wp is None or next_next_wp is None:
            print("Next waypoint(s) not found. Ending episode.")
            self.reset()  # Gracefully end the episode and prepare for a new one
            return

        self.closest_waypoint_index = next_wp_index
        self.next_closest_waypoint_index = next_next_wp_index

        # Print waypoints and thier indices:
        if next_wp is not None:
            print(f"Next waypoint index: {next_wp_index}, Location: {next_wp.transform.location}")
        else:
            print("Next waypoint not found.")

        if next_next_wp is not None:
            print(f"Next next waypoint index: {next_next_wp_index}, Location: {next_next_wp.transform.location}")
        else:
            print("Next next waypoint not found.")

        # Draw next waypoint
        # self.world.debug.draw_point(next_wp.transform.location, life_time=5)

        
        # Retrieve vehicle's current location and velocity
        velocity = self.vehicle.get_velocity()
        self.velocity_x = velocity.x * 3.6
        self.velocity_y = velocity.y * 3.6
        self.velocity_z = velocity.z * 3.6
        self.speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

        # Estimate engine and wheel RPM
        self.engine_rpm = self.estimate_engine_rpm()
        self.wheel_rpm = self.estimate_wheel_rpm(velocity)

        # Retrieve current yaw and calculate yaw rate and acceleration
        current_yaw = self.vehicle.get_transform().rotation.yaw
        current_yaw_rate = self.calculate_yaw_rate(current_yaw)
        self.yaw_acceleration = self.calculate_yaw_acceleration(current_yaw_rate)


        # self.closest_waypoint_index = closest_waypoint_index
        # current_waypoint = self.route_waypoints[closest_waypoint_index]
        # next_waypoint = self.route_waypoints[closest_waypoint_index + 1]

        # CALCULATE d (distance_from_center) and Theta (angle)
        # The result is the distance of the vehicle from the center of the lane:
        self.distance_from_center = self.distance_to_line(self.vector(next_wp.transform.location),
                                                          self.vector(next_next_wp.transform.location),
                                                          self.vector(self.location))
        # Get angle difference between closest waypoint and vehicle forward vector
        fwd    = self.vector(self.vehicle.get_velocity())
        wp_fwd = self.vector(next_wp.transform.rotation.get_forward_vector()) # Return: carla.Vector3D
        # self.angle  = self.angle_diff(fwd, wp_fwd)
        self.angle  = self.angle_diff(next_wp)

        # Update Euclidean distances and deviation angles lists
        self.euclidean_dist_list = self.calculate_euc_dist(next_wp_index)
        self.dev_angle_array = self.calculate_deviation_angle_tan(next_wp_index)
        # self.dev_angle_array = self.calculate_deviation_angles_derivatives(next_wp_index, timestep = 0.03)

        state_dict = {
            "EuclideanDistances": self.euclidean_dist_list,  # Assuming this is a list of distances
            "DeviationAngles": self.dev_angle_array,        # Assuming this is a list of angles
            "Velocity": self.speed,
            "VelocityX": self.velocity_x,
            "VelocityY": self.velocity_y,
            "VelocityZ": self.velocity_z,
            "EngineRPM": self.engine_rpm,
            "DistanceFromCenter": self.distance_from_center,
            "Angle": self.angle,
            "FrameCount": self.frame_count
        }

        # Print each key-value pair in the dictionary
        for key, value in state_dict.items():
            print(f"{key}: {value}")
        # Package the state information
        state = np.array([*self.euclidean_dist_list, *self.dev_angle_array,
                        self.speed, self.velocity_x, self.velocity_y, self.velocity_z,
                        self.engine_rpm, self.distance_from_center, self.angle])

        return state
    
    
    #######################################################
    # Generate Route
    #######################################################
    
    def generate_route(self, total_distance=780):
        """
        Generates a route based on the vehicle's current location and a specified total distance.

        Args:
        - total_distance: The total distance of the route to generate.
        """
        # Initialize the route waypoints list and current waypoint index
        self.route_waypoints = []
        self.current_waypoint_index = 0
        self.total_distance = total_distance  # You can set this depending on the town

        # Get the initial waypoint based on the vehicle's current location
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(),
                                                             project_to_road=True,
                                                             lane_type=carla.LaneType.Driving)
        self.route_waypoints.append(current_waypoint)

        # Generate the rest of the waypoints for the route
        for x in range(self.total_distance):
            # Depending on the section of the route, select the appropriate next waypoint
            if x < 650:
                next_waypoint = current_waypoint.next(10.0)[0]
            else:
                next_waypoint = current_waypoint.next(10.0)[-1]

            self.route_waypoints.append(next_waypoint)
            current_waypoint = next_waypoint


    #######################################################
    # Generate Route - source: https://medium.com/@chardorn/creating-carla-waypoints-9d2cc5c6a656
    #######################################################
    
    def generate_route_2(self):
        """
        Generates waypoints based on the vehicle's lane 

        Positive lane_id values represent lanes to the right of the centerline of the road 
        (when facing in the direction of increasing waypoint s-values, 
        which usually corresponds to the driving direction).
        
        Negative lane_id values represent lanes to the left of the centerline of the road.
        
        The magnitude of the lane_id increases as you move further from the road's centerline,
        meaning lane_id = 1 or lane_id = -1 indicates the lane immediately adjacent to the centerline of the road.
        """
        # Initialize route_waypoints list
        self.route_waypoints = []

        waypoint_list = self.world.get_map().generate_waypoints(10)
        print("Length: " + str(len(waypoint_list)))

        # Determine the vehicle's lane
        vehicle_location = self.vehicle.get_location()

        # Get the waypoint corresponding to the vehicle's location
        waypoint = self.world.get_map().get_waypoint(vehicle_location)

        # Retrieve the lane ID
        target_lane = waypoint.lane_id

        # Retrieve the waypoints from that lane and make them the vehicle's route
        for i in range(len(waypoint_list) - 1):
            if waypoint_list[i].lane_id == target_lane:
                self.route_waypoints.append(waypoint_list[i])
    
    def generate_route_3(self):
        '''
        This method uses GlobalRoutePlanner to generate
        the longest route between the spawn point as a start point
        and a location loc
        '''

        sys.path.append('C:/Program Files/WindowsNoEditor/PythonAPI/carla') 
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        point_a = self.start_point.location #we start at where the car is

        sampling_resolution = 2.5 # the space between waypoints in m
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)

        # now let' pick the longest possible route
        distance = 0
        for loc in self.spawn_points: # we start trying all spawn points 
                                      # but we just exclude first at zero index
            cur_route = grp.trace_route(point_a, loc.location)
            if len(cur_route)>distance:
                distance = len(cur_route)
                route = cur_route
        
        # To save the first element of the route tuple, which is the waypoint object
        self.route_waypoints = [item[0] for item in route]

        # Draw waypoints
        for waypoint in self.route_waypoints:
            marker = self.world.debug.draw_string(waypoint.transform.location, '^', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=150.0, #og: 600.0, set to: 140 the length of completing route
                persistent_lines=True)
            
        # Draw waypoint numbers
        for index in range(25, len(route), 25):
            num_marker = self.world.debug.draw_string(route[index][0].transform.location, str(index), draw_shadow=False,
                                        color=carla.Color(r=0, g=0, b=255), life_time=150.0, #600.0
                                        persistent_lines=True)
            

        # waypoint1 = route[0][0]
        # waypoint2 = route[1][0]

        # # Calculate the Euclidean distance between the waypoints
        # distance = np.linalg.norm(np.array([waypoint1.transform.location.x, waypoint1.transform.location.y]) - 
        #                         np.array([waypoint2.transform.location.x, waypoint2.transform.location.y]))
        
        # print("Distance between waypoint 1 and waypoint 2:", distance)

        # BASED on the above code, DISTANCE = sampling_resolution (m probably)

    
    #######################################################
    # Determine the current waypoint index
    #######################################################
    
    def determine_current_waypoint_index(self):
        closest_waypoint_index = None
        max_dot_product = -float('inf')  # Initialize with very small number
        vehicle_location = self.vehicle.get_location()
        vehicle_forward_vector = self.vehicle.get_transform().get_forward_vector()
        
        for i, waypoint in enumerate(self.route_waypoints):
            waypoint_vector = waypoint.transform.location - vehicle_location
            dot_product = vehicle_forward_vector.dot(waypoint_vector)
            
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                closest_waypoint_index = i
                
        return closest_waypoint_index
    
    #######################################################
    # Get next waypoint
    #######################################################
    
    def get_next_waypoint(self):
        vehicle_location = self.vehicle.get_location()
        min_distance = 1000
        next_waypoint = None

        for waypoint in self.route_waypoints:
            waypoint_location = waypoint.transform.location

            # Only check waypoints that are in the front of the vehicle 
            # (if x is negative, then the waypoint is to the rear)
            #TODO: Check if this applies for all maps
            if (waypoint_location - vehicle_location).x > 0:

                # Find the waypoint closest to the vehicle, 
                # but once vehicle is close to upcoming waypoint, search for next one
                if vehicle_location.distance(waypoint_location) < min_distance and vehicle_location.distance(waypoint_location) > 5:
                    min_distance = vehicle_location.distance(waypoint_location)
                    next_waypoint = waypoint

        return next_waypoint
    
    #######################################################
    # Get next two waypoints and their indices
    #######################################################
    
    def get_next_two_waypoints_and_indices_old(self):
        vehicle_location = self.vehicle.get_location()
        min_distance = 1000
        next_waypoint = None
        next_next_waypoint = None
        index_of_next_waypoint = None
        index_of_next_next_waypoint = None

        if len(self.route_waypoints) > 0:
            print(f"First waypoint location: {self.route_waypoints[0].transform.location}")
            print(f"Last waypoint location: {self.route_waypoints[-1].transform.location}")

        print(f"Current vehicle location: {vehicle_location}")

        # Find the closest waypoint ahead of the vehicle
        for index, waypoint in enumerate(self.route_waypoints):
            waypoint_location = waypoint.transform.location

            # Assuming the vehicle's forward direction corresponds with increasing waypoint index
            distance = vehicle_location.distance(waypoint_location)
            if distance < min_distance and (waypoint_location - vehicle_location).x > 0:
                min_distance = distance
                next_waypoint = waypoint
                index_of_next_waypoint = index

        # If a closest waypoint is found, attempt to get the next waypoint in the list
        if next_waypoint is not None and index_of_next_waypoint is not None:
            try:
                next_next_waypoint = self.route_waypoints[index_of_next_waypoint + 1]
                index_of_next_next_waypoint = index_of_next_waypoint + 1
            except IndexError:
                next_next_waypoint = None  # This might happen if the next waypoint is the last one
                index_of_next_next_waypoint = None

        return next_waypoint, next_next_waypoint, index_of_next_waypoint, index_of_next_next_waypoint
    
    def get_next_two_waypoints_and_indices(self):
        '''
        This method retrieves the waypoint index
        it is a new version inspired by:
        https://github.com/vadim7s/SelfDrive/blob/master/Tutorials/tutorial_4_simple_navigation.ipynb
        works with generate_route_3
        '''
        next_waypoint = None
        next_next_waypoint = None
        index_of_next_waypoint = None
        index_of_next_next_waypoint = None

        if self.curr_wp < len(self.route_waypoints) - 1:
            next_waypoint = self.route_waypoints[self.curr_wp]
            index_of_next_waypoint = self.curr_wp

            # Check the distance to the next waypoint and move to the next one if too close
            # threshold = distance between waypoints/2
            while self.vehicle.get_transform().location.distance(next_waypoint.transform.location) < 5:
                self.curr_wp += 1
                if self.curr_wp >= len(self.route_waypoints) - 1:
                    return None, None, None, None  # End of route reached, no more waypoints

                next_waypoint = self.route_waypoints[self.curr_wp]
                index_of_next_waypoint = self.curr_wp

            if self.curr_wp < len(self.route_waypoints) - 2:
                next_next_waypoint = self.route_waypoints[self.curr_wp + 1]
                index_of_next_next_waypoint = self.curr_wp + 1

        return next_waypoint, next_next_waypoint, index_of_next_waypoint, index_of_next_next_waypoint    
    
    def generate_custom_spawn_points(self, distance=10):
        map = self.world.get_map()
        # Generate waypoints across the map at specified intervals
        waypoints = map.generate_waypoints(distance)
        custom_spawn_points = []

        for waypoint in waypoints:
            # Optionally filter waypoints; for example, ensure they are in driving lanes
            if waypoint.lane_type == carla.LaneType.Driving:
                # Create a Transform with the waypoint's location and rotation
                location = waypoint.transform.location
                # Adjust Z location to ensure the vehicle spawns above the ground
                location.z += 2
                rotation = waypoint.transform.rotation
                spawn_transform = carla.Transform(location, rotation)

                custom_spawn_points.append(spawn_transform)

        return custom_spawn_points


####################################################### End of SimEnv

#######################################################
# Original Reward Function (JabrahTutorials)
#######################################################
    
def get_reward_comp(vehicle, waypoint, collision):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y

    x_vh = vehicle_location.x
    y_vh = vehicle_location.y

    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])

    dist = np.linalg.norm(wp_array - vh_array)

    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

    collision = 0 if collision is None else 1
    
    return cos_yaw_diff, dist, collision

def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward

#######################################################
# Updated Reward Function as in RL Paper - V1
#######################################################

def calculate_reward_1(theta, d, v, theta_dot):
    """
    Calculate the reward based on the current state.

    :param theta: Current deviation angle from the trajectory
    :param d: Current lateral distance from the trajectory
    :param v: Current velocity of the car
    :param theta_dot: Current yaw acceleration (rate of change of theta)
    :return: The calculated reward
    """
    # Ensure theta and d are within the maximum bounds to avoid negative rewards
    angle_deviation_penalty = max(0, MAX_DEVIATION_ANGLE - abs(theta))
    lateral_distance_penalty = max(0, MAX_DEVIATION_DISTANCE - abs(d))
    # Assuming REWARD_CONSTANT_C is set such that this term is always positive
    comfort_penalty = REWARD_CONSTANT_C - abs(theta_dot)  
    velocity_reward = v / MAX_VELOCITY_THRESHOLD  # Scales with the velocity of the car

    # Compute the reward
    reward = angle_deviation_penalty * lateral_distance_penalty * comfort_penalty * velocity_reward
    return reward

#######################################################
# Updated Reward Function as in RL Paper - V2
#######################################################

def calculate_reward_2(theta, d, v, theta_dot):
    """
    Calculate the reward based on the current state.

    :param theta: Current deviation angle from the trajectory
    :param d: Current lateral distance from the trajectory
    :param v: Current velocity of the car
    :param theta_dot: Current yaw acceleration (rate of change of theta)
    :return: The calculated reward
    """
    # Ensure theta and d are within the maximum bounds to avoid negative rewards
    angle_deviation_penalty = max(0, MAX_DEVIATION_ANGLE - abs(theta))
    lateral_distance_penalty = max(0, MAX_DEVIATION_DISTANCE - abs(d))
    # Assuming REWARD_CONSTANT_C is set such that this term is always positive
    comfort_penalty = REWARD_CONSTANT_C - abs(theta_dot)  
    velocity_reward = v / MAX_VELOCITY_THRESHOLD  # Scales with the velocity of the car

    # Compute the reward
    reward = angle_deviation_penalty * lateral_distance_penalty * comfort_penalty * velocity_reward
    return reward

#######################################################
# Updated Reward Function as in RL Paper - V3
#######################################################

def correct_yaw(x):
    """
    Normalize yaw to the range [0, 360) degrees
    or to the range [0, 2 * pi) radians.
    """
    return ((x % (2 * np.pi)) + (2 * np.pi)) % (2 * np.pi)

def calculate_reward_3(ep, vehicle, next_waypoint, next_next_waypoint, current_steering, last_steering, steering_history,
                       reward_log, d_max=6.0, theta_max=1.5, v_max=5.0, C=1.0):
    """
    Calculate the reward as defined in the paper for the CARLA environment.

    Parameters:
    - vehicle: CARLA vehicle object
    - waypoint: Single CARLA waypoint object representing the immediate target
    - d_max: Maximum lateral deviation distance allowed (in meters)
    - theta_max: Maximum deviation angle allowed (in radians) (default: pi/12 ≈ 15 degrees)
    - v_max: Maximum velocity threshold (in m/s)
    - C: Scaling constant

    Returns:
    - reward: Computed reward value
    """

    """
    Weights in the weighted reward:
    w1: Weight for the lateral deviation normalization
    w2: Weight for the angular deviation normalization
    w3: Weight for the velocity normalization
    w4: Weight for the progress reward
    w5: Weight for the steering penalty
    """
    # # w1 = 1.0  # Weight for lateral deviation normalization
    # w1 = 1.1  # Weight for lateral deviation normalization #increment for shap_v4 after epsiode 500 (p3)
    # # w2 = 1.2  # Weight for angular deviation normalization
    # w2 = 1.3  # Weight for angular deviation normalization #increment for shap_v3
    # # w2 = 1.1  # Weight for angular deviation normalization #decrement for shap_v4
    # w3 = 1.0  # Weight for velocity normalization
    # # w4 = 1.2  # Weight for the progress reward
    # w4 = 1.5  # Weight for the progress reward in new version
    # w5 = 1.2  # Weight for the steering penalty

    w1 = 1.02  # Weight for lateral deviation normalization
    w2 = 1.22  # Weight for angular deviation normalization
    w3 = 1.0  # Weight for velocity normalization
    # w4 = 1.2  # Weight for the progress reward
    w4 = 1.5  # Weight for the progress reward in new version
    w5 = 1.2  # Weight for the steering penalty

    # Extract vehicle state
    vehicle_location = vehicle.get_location()
    vehicle_transform = vehicle.get_transform()
    vehicle_yaw = correct_yaw(np.radians(vehicle_transform.rotation.yaw))
    vehicle_velocity = vehicle.get_velocity()
    current_speed = np.linalg.norm([vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z])

    # Initialize deviation variables
    d = 0
    theta = 0

    if next_waypoint:
        # Extract waypoint position and yaw
        target_location = next_waypoint.transform.location
        target_yaw = correct_yaw(np.radians(next_waypoint.transform.rotation.yaw))

        # Calculate lateral deviation distance
        vec_wp = np.array([target_location.x - vehicle_location.x, target_location.y - vehicle_location.y])
        vehicle_forward = np.array([np.cos(vehicle_yaw), np.sin(vehicle_yaw)])
        d = np.cross(vehicle_forward, vec_wp) / np.linalg.norm(vehicle_forward)

        # Calculate angle deviation
        theta = abs(target_yaw - vehicle_yaw)
        theta = min(theta, 2 * np.pi - theta)  # Ensure within [0, pi)

    # Normalize the deviations
    d_norm = (d_max - abs(d)) / d_max
    theta_norm = (theta_max - abs(theta)) / theta_max
    v_norm = min(current_speed / v_max, 1.0)

    # Added progress reward
    progress_reward = calculate_progress_reward(vehicle, next_waypoint, next_next_waypoint)
    # steering_penalty = calculate_steering_penalty(last_steering, current_steering)
    steering_penalty = calculate_steering_penalty_avg(steering_history, current_steering)

    # Adjusted reward calculation
    reward = w1 * d_norm + w2 * theta_norm + w3 * v_norm + w4 * progress_reward - w5 * steering_penalty
    reward = max(reward, 0)  # Ensure non-negative

    # Reward Log
    reward_log.writerow([ep, next_waypoint, d_norm, theta_norm, v_norm, progress_reward, steering_penalty, reward])

    return reward


def calculate_reward_4(ep, vehicle, next_waypoint, next_next_waypoint, current_steering, last_steering, reward_log,
                       d_max=6.0, theta_max=1.5, v_max=5.0, C=1.0):
    
    # SAME as calculate_reward_3 but with the addition of oscillation penalty
    """
    Calculate the reward as defined in the paper for the CARLA environment.
    Parameters:
    - vehicle: CARLA vehicle object
    - waypoint: Single CARLA waypoint object representing the immediate target
    - d_max: Maximum lateral deviation distance allowed (in meters)
    - theta_max: Maximum deviation angle allowed (in radians) (default: pi/12 ≈ 15 degrees)
    - v_max: Maximum velocity threshold (in m/s)
    - C: Scaling constant
    Returns:
    - reward: Computed reward value
    """
    w1 = 0.8
    w2 = 1.2
    w3 = 1.0
    w4 = 1.2
    w5 = 1.2
    w6 = 0.5  # w6 is the new weight for oscillation damping

    # Extract vehicle state
    vehicle_location = vehicle.get_location()
    vehicle_transform = vehicle.get_transform()
    vehicle_yaw = correct_yaw(np.radians(vehicle_transform.rotation.yaw))
    vehicle_velocity = vehicle.get_velocity()
    current_speed = np.linalg.norm([vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z])

    # Initialize deviation variables
    d, theta = 0, 0

    if next_waypoint:
        # Extract waypoint position and yaw
        target_location = next_waypoint.transform.location
        target_yaw = correct_yaw(np.radians(next_waypoint.transform.rotation.yaw))

        # Calculate lateral deviation distance
        vec_wp = np.array([target_location.x - vehicle_location.x, target_location.y - vehicle_location.y])
        vehicle_forward = np.array([np.cos(vehicle_yaw), np.sin(vehicle_yaw)])
        d = np.cross(vehicle_forward, vec_wp) / np.linalg.norm(vehicle_forward)

        # Calculate angle deviation
        theta = abs(target_yaw - vehicle_yaw)
        theta = min(theta, 2 * np.pi - theta)  # Ensure within [0, pi)

    # Normalize the deviations
    d_norm = (d_max - abs(d)) / d_max
    theta_norm = (theta_max - abs(theta)) / theta_max
    v_norm = min(current_speed / v_max, 1.0)

    # Calculate rewards and penalties
    progress_reward = calculate_progress_reward(vehicle, next_waypoint, next_next_waypoint)
    steering_penalty = calculate_steering_penalty(last_steering, current_steering)
    oscillation_damping = calculate_oscillation_damping(last_steering, current_steering)

    # Adjusted reward calculation
    reward = (w1 * d_norm + w2 * theta_norm + w3 * v_norm +
              w4 * progress_reward - w5 * steering_penalty - w6 * oscillation_damping)
    reward = max(reward, 0)  # Ensure non-negative

    # Reward Log
    reward_log.writerow([ep, next_waypoint, d_norm, theta_norm, v_norm, progress_reward, steering_penalty, oscillation_damping, reward])

    return reward

def calculate_oscillation_damping(last_steering, current_steering, oscillation_threshold=0.05, oscillation_penalty=10.0):
    """
    Calculate a damping penalty for oscillations in steering commands.

    Parameters:
    - last_steering: The last normalized steering command (-1 to 1).
    - current_steering: The current normalized steering command (-1 to 1).
    - oscillation_threshold: Threshold for change considered as oscillation.
    - oscillation_penalty: Maximum penalty for oscillations.

    Returns:
    - penalty: Computed damping penalty.
    """
    # Calculate the change in steering command
    steering_change = abs(current_steering - last_steering)

    # Apply a damping penalty if the change is frequent and substantial
    if steering_change > oscillation_threshold:
        penalty = (steering_change / oscillation_threshold) * oscillation_penalty
        penalty = min(penalty, oscillation_penalty)  # Cap the penalty
    else:
        penalty = 0

    return penalty

    

def calculate_steering_penalty(last_steering, current_steering, steering_change_threshold=0.1, max_penalty=5.0):
    """
    Calculate a penalty for abrupt changes in steering, adjusted for normalized steering values.

    Parameters:
    - last_steering: The last normalized steering command (-1 to 1).
    - current_steering: The current normalized steering command (-1 to 1).
    - steering_change_threshold: The threshold for change in normalized steering command considered 'normal'.
    - max_penalty: The maximum penalty to apply for abrupt changes.

    Returns:
    - penalty: Computed steering change penalty.
    """
    # Calculate the absolute change in steering command
    steering_change = abs(current_steering - last_steering)

    # Apply a penalty if the steering change exceeds the threshold
    if steering_change > steering_change_threshold:
        # Normalize the penalty based on the amount of change beyond the threshold
        # Adjust the normalization range according to your vehicle's steering sensitivity
        penalty = (steering_change - steering_change_threshold) / (1 - steering_change_threshold) * max_penalty
        penalty = min(penalty, max_penalty)  # Cap the penalty to the maximum allowed
    else:
        penalty = 0

    return penalty

def calculate_steering_penalty_avg(steering_history, current_steering, window_size=150, steering_change_threshold=0.1, max_penalty=5.0):
    """
    Calculate a penalty for abrupt changes in steering, using a moving average for smoothing.

    Parameters:
    - steering_history: List of past normalized steering commands.
    - current_steering: The current normalized steering command (-1 to 1).
    - window_size: The number of past commands to consider for the moving average.
    - steering_change_threshold: The threshold for change in steering considered 'normal'.
    - max_penalty: The maximum penalty to apply for abrupt changes.

    Returns:
    - penalty: Computed steering change penalty.
    """
    # Calculate the moving average of past steering commands
    if len(steering_history) < window_size:
        avg_steering = sum(steering_history) / len(steering_history)
    else:
        avg_steering = sum(steering_history[-window_size:]) / window_size
    
    # Calculate the change relative to the moving average
    steering_change = abs(current_steering - avg_steering)

    # Apply a penalty if the steering change exceeds the threshold
    if steering_change > steering_change_threshold:
        penalty = (steering_change - steering_change_threshold) / (1 - steering_change_threshold) * max_penalty
        penalty = min(penalty, max_penalty)  # Cap the penalty to the maximum allowed
    else:
        penalty = 0

    return penalty


def calculate_progress_reward_old(vehicle, current_waypoint, next_waypoint, bonus_per_waypoint=10.0):
    """
    PROBLEM WITH THIS VERSION: yeilds large negative numbers, which yields to 0 overall reward


    Calculate the progress reward for moving towards and reaching waypoints.

    Parameters:
    - vehicle: CARLA vehicle object
    - current_waypoint: The waypoint the vehicle is currently targeting
    - next_waypoint: The next waypoint in the route after the current waypoint
    - bonus_per_waypoint: The reward bonus for reaching each new waypoint

    Returns:
    - progress_reward: Computed progress reward value
    """
    # Get the vehicle's current location
    vehicle_location = vehicle.get_location()
    vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])

    # Get the current waypoint's location
    current_wp_pos = np.array([current_waypoint.transform.location.x, current_waypoint.transform.location.y])

    # Get the next waypoint's location if it exists
    next_wp_pos = np.array([next_waypoint.transform.location.x, next_waypoint.transform.location.y])

    # Calculate the distance to the current waypoint
    distance_to_current_wp = np.linalg.norm(vehicle_pos - current_wp_pos)

    # Calculate the distance to the next waypoint if it exists
    distance_to_next_wp = np.linalg.norm(vehicle_pos - next_wp_pos)

    # Calculate progress reward based on the distance reduction to the next waypoint
    # Assume max_distance is the distance when the vehicle just reached the current waypoint
    max_distance = np.linalg.norm(current_wp_pos - next_wp_pos)
    distance_reduction = max_distance - distance_to_next_wp

    # Normalize the distance reduction to the range of 0 to 1
    distance_reduction_norm = distance_reduction / max_distance

    # Compute the reward as a scaled value of the normalized distance reduction
    progress_reward = distance_reduction_norm * bonus_per_waypoint

    # Check if the vehicle has reached the next waypoint
    # if distance_to_next_wp < 2.0:  # threshold to consider waypoint reached, adjust based on your scenario
    if distance_to_next_wp < 1.5:  # threshold to consider waypoint reached, adjust based on your scenario
        progress_reward += bonus_per_waypoint  # give a bonus for reaching the waypoint

    return progress_reward

def calculate_progress_reward(vehicle, current_waypoint, next_waypoint, bonus_per_waypoint=10.0):
    vehicle_location = vehicle.get_location()
    vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])

    current_wp_pos = np.array([current_waypoint.transform.location.x, current_waypoint.transform.location.y])
    next_wp_pos = np.array([next_waypoint.transform.location.x, next_waypoint.transform.location.y])

    # Calculate the current distance to the next waypoint
    current_distance_to_next_wp = np.linalg.norm(vehicle_pos - next_wp_pos)

    # We assume initial distance should be the distance at the start of moving towards next waypoint
    if 'initial_distance_to_next_wp' not in globals():
        global initial_distance_to_next_wp
        initial_distance_to_next_wp = current_distance_to_next_wp

    # Calculate the reduction in distance towards the next waypoint
    distance_reduction = initial_distance_to_next_wp - current_distance_to_next_wp
    initial_distance_to_next_wp = current_distance_to_next_wp  # update initial distance for the next cycle

    # Normalize the distance reduction (ensuring it's not negative)
    distance_reduction_norm = max(0, distance_reduction / initial_distance_to_next_wp)

    # Compute the reward as a scaled value of the normalized distance reduction
    progress_reward = distance_reduction_norm * bonus_per_waypoint

    # Check if the vehicle has reached the next waypoint
    if current_distance_to_next_wp < 1.5:  # threshold to consider waypoint reached. ranges between 1.5 and 2 depending on sampling_resolution
        progress_reward += bonus_per_waypoint  # give a bonus for reaching the waypoint

    return progress_reward



#######################################################
# Initialize Logger
#######################################################

def initialize_logger(log_file_path):
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create a file handler for the logger
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

#######################################################
# Initialize File Writer
#######################################################
def initialize_csv_writer(csv_file_path):
    # Set up CSV writer
    csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow([
    'Time', 'Episode', 'ClosestWaypointIndex',
    'EuclideanDist1', 'EuclideanDist2', 'EuclideanDist3', 'EuclideanDist4', 'EuclideanDist5',
    'EuclideanDist6', 'EuclideanDist7', 'EuclideanDist8', 'EuclideanDist9', 'EuclideanDist10',
    'DeviationAngle1', 'DeviationAngle2', 'DeviationAngle3', 'DeviationAngle4', 'DeviationAngle5',
    'DeviationAngle6', 'DeviationAngle7', 'DeviationAngle8', 'DeviationAngle9', 'DeviationAngle10',
    'Velocity', 'VelocityX', 'VelocityY', 'VelocityZ',
    'EngineRPM', 'DistanceFromCenter', 'Angle', 'SteeringCommand', 'Reward', 'EpisodeReward', 'TotalReward'
    ])
    return csv_writer, csv_file

def initialize_termination_log(csv_termination_log):
    # Set up CSV writer
    csv_file = open(csv_termination_log, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(['Episode', 'Reason', 'TotalReward'])
    return csv_writer, csv_file

def initialize_euc_log(csv_euc_waypoints_log):
    # Set up CSV writer
    csv_file = open(csv_euc_waypoints_log, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(['CurrentWaypoint', 'Waypoint 1', 'Waypoint 2', 'Waypoint 3', 'Waypoint 4',
                          'Waypoint 5', 'Waypoint 6', 'Waypoint 7', 'Waypoint 8', 'Waypoint 9', 'Waypoint 10'])
    return csv_writer, csv_file

def initialize_reward_log(csv_reward_log):
    # Set up CSV writer
    csv_file = open(csv_reward_log, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(['Episode', 'NextWaypoint', 'd_norm', 'theta_norm', 'v_norm', 'ProgressReward', 'SteeringPenalty', 'Reward'])
    return csv_writer, csv_file

def initialize_metrics_log(csv_metrics_log):
    # Set up CSV writer
    csv_file = open(csv_metrics_log, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(['Episode', 'YawAcceleration (deg/s^2)', 'SteeringAngle (scalar [-1, 1])', 
                         'AngleDeviation (rad)', 'DistanceDeviation (m)'])
    return csv_writer, csv_file

def initialize_trajectory_log(csv_trajectory_log):
    # Set up CSV writer
    csv_file = open(csv_trajectory_log, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(['Episode', 'LocationX', 'LocationY'])
    return csv_writer, csv_file

def initialize_waypoints_log(csv_waypoints_log):
    # Set up CSV writer
    csv_file = open(csv_waypoints_log, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(['Episode', 'Waypoint', 'WaypointLocationX', 'WaypointLocationY', 'WaypointLocationZ'])
    return csv_writer, csv_file

def initialize_shap_log(csv_shap_log):
    # Set up CSV writer
    csv_file = open(csv_shap_log, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(['Episode', 'Waypoint', 'Global Time Step', 'Dominant Feature', 'SHAP Value'])
    return csv_writer, csv_file

# def log_termination(episode, reason, total_reward, file_path='termination_log.csv'):
#     """Logs the episode termination details to a CSV file."""
#     with open(file_path, 'a') as file:
#         #file.write(f"{episode},{reason},{total_reward}\n")
#         self.csv_writer.writerow([time.time(), ep, self.closest_waypoint_index,
#                                             *state, steer, reward, episode_reward, self.total_rewards])



def set_random_weather(world):
    # Create a WeatherParameters object
    weather = carla.WeatherParameters()

    # Randomize weather conditions
    weather.sun_azimuth_angle = random.uniform(0, 360)    # Sun position around the horizon
    weather.sun_altitude_angle = random.uniform(-30, 90)  # Sun position above the horizon
    weather.cloudiness = random.uniform(0, 100)           # Cloud cover
    weather.precipitation = random.uniform(0, 100)        # Rain intensity
    weather.fog_density = random.uniform(0, 100)          # Fog concentration
    weather.fog_distance = random.uniform(0, 100)         # Fog start distance
    weather.fog_falloff = random.uniform(0, 5)            # Fog density falloff
    weather.wetness = random.uniform(0, 100)              # Wetness intensity
    weather.precipitation_deposits = random.uniform(0, 100) # Puddles on the road
    weather.wind_intensity = random.uniform(0, 100)       # Wind effect
    weather.scattering_intensity = random.uniform(0, 1)   # Volumetric fog light scattering
    weather.mie_scattering_scale = random.uniform(0, 1)   # Light scattering from large particles
    weather.rayleigh_scattering_scale = random.uniform(0, 0.1) # Light scattering from small particles

    # Apply the randomized weather conditions to the world
    world.set_weather(weather)


def set_overcast_weather(world):
    # Create a WeatherParameters object with overcast settings
    weather = carla.WeatherParameters(
        cloudiness=80.0,               # High cloud cover to diffuse sunlight
        precipitation=0.0,             # No rain
        precipitation_deposits=0.0,    # No puddles
        wind_intensity=10.0,           # Mild wind
        sun_azimuth_angle=90.0,        # Sun position (can adjust to minimize direct light)
        sun_altitude_angle=15.0,       # Low sun altitude to avoid strong sunlight
        fog_density=0.0,               # Light fog to further diffuse light
        fog_distance=100.0,            # Fog start distance
        fog_falloff=0.1,               # Fog density falloff
        wetness=0.0,                   # No wet roads
        scattering_intensity=0.3,      # Mild scattering for a soft light effect
        mie_scattering_scale=0.1,      # Mild scattering due to particles
        rayleigh_scattering_scale=0.0331 # Standard value for normal atmospheric scattering
    )

    # Apply the weather conditions to the world
    world.set_weather(weather)