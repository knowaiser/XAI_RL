import os
import sys
import glob

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
from collections import deque
from utils import get_speed

class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """
    def __init__(self, vehicle, max_throttle=0.75, max_brake=0.3, K_P=0.299999997117411,
                K_I=0.499999995195557, K_D=0.049999999519492, dt=0.03):
    #the above are best results based on 1-initial manual tuning 2- L-BFGS-B tuning with the manual results as initials
    #def __init__(self, vehicle, max_throttle=0.75, max_brake=0.3, K_P=0.3, K_I=0.5, K_D=0.05, dt=0.03):
    #the above is somewhat tuned but needs improvement
    #def __init__(self, vehicle, max_throttle=0.75, max_brake=0.3, K_P=0.3, K_I=0.5, K_D=0.05, dt=0.03):
        """
        Constructor method.
            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self.max_throttle = max_throttle 
        self.max_brake = max_brake 
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.
            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        acceleration = self._pid_control(target_speed, current_speed)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)
        return control

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations
            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        # error = target_speed - current_speed
        # self._error_buffer.append(error)

        # if len(self._error_buffer) >= 2:
        #     _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
        #     _ie = sum(self._error_buffer) * self._dt
        # else:
        #     _de = 0.0
        #     _ie = 0.0

        # return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

        # New implementation to perform anti-windup
        error = target_speed - current_speed # UNIT: km/h

        # Proportional term
        p_term = self._k_p * error # UNIT: control output per km/h

        # Derivative term
        if len(self._error_buffer) >= 2:
            _de = (error - self._error_buffer[-1]) / self._dt
        else:
            _de = 0.0
        d_term = self._k_d * _de # UNIT: control output per (km/h per second)

        # Anti-windup: Only integrate when control output is not saturated
        control = p_term + d_term
        control = np.clip(control, -1.0, 1.0)

        if abs(control) < 1.0:
            self._error_buffer.append(error)
            _ie = sum(self._error_buffer) * self._dt
        else:
            _ie = 0.0

        i_term = self._k_i * _ie # UNIT: control output per (km/h-seconds)

        # Recalculate control output with integral term
        control = p_term + i_term + d_term # UNIT: unitless
        control = np.clip(control, -1.0, 1.0) # UNIT: unitless

        return control 

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt # UNIT: seconds


