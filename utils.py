import os
import cv2
import pygame
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_img(image, dim_x=128, dim_y=128):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    # scale_percent = 25
    # width = int(array.shape[1] * scale_percent/100)
    # height = int(array.shape[0] * scale_percent/100)

    # dim = (width, height)
    dim = (dim_x, dim_y)  # set same dim for now
    resized_img = cv2.resize(array, dim, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    scaledImg = img_gray/255.

    # normalize
    mean, std = 0.5, 0.5
    normalizedImg = (scaledImg - mean) / std

    return normalizedImg

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.
        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) # UNIT: km/h

def correct_yaw(x):
    return(((x%360) + 360) % 360)

def create_folders(folder_names):
    for directory in folder_names:
        if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)

def get_latest_checkpoint(checkpoint_dir):
    list_of_files = os.listdir(checkpoint_dir)
    full_path = [os.path.join(checkpoint_dir, file) for file in list_of_files]
    latest_file = max(full_path, key=os.path.getctime)
    return latest_file
    
def update_min_max(new_data, state_min, state_max):
    # This function can be called every time new data is collected
    np.minimum(state_min, new_data, out=state_min)
    np.maximum(state_max, new_data, out=state_max)

def normalize_state(state, state_min, state_max):
    # Prevent division by zero with a small epsilon
    eps = 1e-10
    range = np.maximum(state_max - state_min, eps)
    normalized_state = (state - state_min) / range
    return normalized_state

def calculate_performance_metrics(num_pid_sets, pid_parameters_list):
    """
    Calculates performance metrics for each PID parameter set.

    Parameters:
    - num_pid_sets: Number of PID parameter sets to analyze.
    - pid_parameters_list: List of PID parameters [K_P, K_I, K_D] for each set.
    """
    metrics = []

    for idx in range(1, num_pid_sets + 1):
        df = pd.read_csv(f'pid_{idx}_data.csv')

        # Retrieve the corresponding PID parameters
        k_p, k_i, k_d = pid_parameters_list[idx - 1]

        # Calculate metrics
        mae = np.mean(np.abs(df['Error']))
        rmse = np.sqrt(np.mean(df['Error'] ** 2))
        max_error = np.max(np.abs(df['Error']))
        total_throttle = np.sum(df['Throttle']) * (df['Time'][1] - df['Time'][0])
        total_brake = np.sum(df['Brake']) * (df['Time'][1] - df['Time'][0])
        std_throttle = np.std(df['Throttle'])
        std_brake = np.std(df['Brake'])

        metrics.append({
            'PID_Index': idx,
            'K_P': k_p,
            'K_I': k_i,
            'K_D': k_d,
            'MAE': mae,
            'RMSE': rmse,
            'MaxError': max_error,
            'TotalThrottle': total_throttle,
            'TotalBrake': total_brake,
            'StdThrottle': std_throttle,
            'StdBrake': std_brake
        })

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('pid_performance_metrics.csv', index=False)
    print("Performance metrics calculated and saved to 'pid_performance_metrics.csv'.")

    # Sort the metrics DataFrame based on RMSE
    metrics_df_sorted = metrics_df.sort_values(by='RMSE')

    # Save the sorted metrics to a CSV file
    metrics_df_sorted.to_csv('pid_performance_metrics_sorted.csv', index=False)
    print("Performance metrics sorted and saved to 'pid_performance_metrics_sorted.csv'.")

    print("PID Parameter Sets Ranked from Best to Worst based on RMSE:")
    print(metrics_df_sorted[['PID_Index', 'K_P', 'K_I', 'K_D', 'RMSE']])



def plot_pid_results(num_pid_sets):
    """
    Plots the results for each PID parameter set.

    Parameters:
    - num_pid_sets: Number of PID parameter sets to plot.
    """
    for idx in range(1, num_pid_sets + 1):
        # Read the data
        df = pd.read_csv(f'pid_{idx}_data.csv')

        # Create plots
        plt.figure(figsize=(12, 8))

        # Plot Speed vs. Time
        plt.subplot(2, 2, 1)
        plt.plot(df['Time'], df['Speed'], label='Actual Speed')
        plt.plot(df['Time'], df['TargetSpeed'], label='Target Speed', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (km/h)')
        plt.title(f'PID Set {idx}: Speed vs. Time')
        plt.legend()

        # Plot Speed Error vs. Time
        plt.subplot(2, 2, 2)
        plt.plot(df['Time'], df['Error'])
        plt.xlabel('Time (s)')
        plt.ylabel('Speed Error (km/h)')
        plt.title(f'PID Set {idx}: Speed Error vs. Time')

        # Plot Throttle vs. Time
        plt.subplot(2, 2, 3)
        plt.plot(df['Time'], df['Throttle'])
        plt.xlabel('Time (s)')
        plt.ylabel('Throttle')
        plt.title(f'PID Set {idx}: Throttle vs. Time')

        # Plot Brake vs. Time
        plt.subplot(2, 2, 4)
        plt.plot(df['Time'], df['Brake'])
        plt.xlabel('Time (s)')
        plt.ylabel('Brake')
        plt.title(f'PID Set {idx}: Brake vs. Time')

        plt.tight_layout()
        plt.savefig(f'pid_{idx}_plots.png')
        plt.close()
        print(f"Plots saved for PID parameter set {idx}.")

    print("All plots generated.")
