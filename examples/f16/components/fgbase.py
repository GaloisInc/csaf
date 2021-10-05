import threading
import socket
import time
import math
import typing

import numpy as np
import pymap3d as pm

from abc import ABC
from scipy.spatial.transform import Rotation

import csaf.core.trace

class Dubins2DConverter():
    """
    Originally from John McCarroll, modified by Michal Podhradsky

    Converts orientation and rotation from ENU to ECEF
    """

    @staticmethod
    def quaternion_from_lon_lat(lon: float, lat: float) -> typing.List[float]:
        """
        A helper function to calculate a quaternion representation of a rotation from ENU to ECEF
        parameters: longitude and latitude (radians)
        returns: list of quaternion components (scalar last)
        """
        zd2 = 0.5 * lon
        yd2 = -0.25 * math.pi - 0.5 * lat
        Szd2 = math.sin(zd2)
        Syd2 = math.sin(yd2)
        Czd2 = math.cos(zd2)
        Cyd2 = math.cos(yd2)
        w = Czd2 * Cyd2
        x = -Szd2 * Syd2
        y = Czd2 * Syd2
        z = Szd2 * Cyd2
        return [x, y, z, w]
    
    @classmethod
    def convert_to_ecef(cls, pn_m: float, pe_m: float, pu_m: float,
                        phi_rad: float, theta_rad: float, psi_rad: float,
                        lat0_rad: float, lon0_rad: float, h0_m: float) -> typing.Tuple[float]:
        """
        This method takes in a dictionary of "raw" 2D Dubins log data, as read from the LogReader class,
        and returns a populated Episode object.
        """
        # position conversions

        ## ENU to ECEF
        # convert position to geocentric (Earth-centered) reference frame
        ecef_x, ecef_y, ecef_z = pm.enu2ecef(pe_m, pn_m, pu_m, lat0_rad, lon0_rad, h0_m, ell=None, deg=False)

        # orientation conversions

        ## ECEF
        # 1st rotation (frame alignment)
        global_rotation = Rotation.from_quat(Dubins2DConverter.quaternion_from_lon_lat(
            lon0_rad, lat0_rad))
        # 2nd rotation (from data)
        local_rotation = Rotation.from_euler('xyz', [phi_rad, theta_rad, psi_rad], degrees=False)

        # multiply
        rotation = global_rotation * local_rotation

        quaternion = rotation.as_quat()
        angle = 2 * math.acos(quaternion[3])  # cos(a / 2) = w
        direction = quaternion / (math.sin(angle / 2))  # [Vx,Vy,Vz] * sin(a / 2) = [x,y,z]
        ecef_x_orientation = direction[0] * angle
        ecef_y_orientation = direction[1] * angle
        ecef_z_orientation = direction[2] * angle

        return (
            ecef_x, ecef_y, ecef_z,
            ecef_x_orientation,
            ecef_y_orientation,
            ecef_z_orientation
        )

class FlightGearBase(ABC):
    # Start position of the aircraft
    DEFAULT_FG_LAT = 35.802117
    DEFAULT_FG_LON = -117.806717
    DEFAULT_FG_GROUND_LEVEL = 1500 # m
    # Default max values for actutors
    DEFAULT_FG_AILERON_MAX_DEG = 21.5
    DEFAULT_FG_ELEVATOR_MAX_DEG = 25
    DEFAULT_FG_RUDDER_MAX_DEG = 30.0

    FG_FT_IN_M = 3.2808
    # Networking variables
    DEFAULT_FG_IP = "192.168.40.219"#"127.0.0.1"

    DEFAULT_DELTA_T = 0.5

    # Class variables
    reset_flag = False
    plant = None
    controller = None

    lag = DEFAULT_DELTA_T
    speed = 1.0
    initial_time = None
    sim_flag = False
    stopped = False
    main_loop = None

    lat0 = np.deg2rad(DEFAULT_FG_LAT)
    lon0 = np.deg2rad(DEFAULT_FG_LON)
    h0 = DEFAULT_FG_GROUND_LEVEL
    sock_args = (socket.AF_INET, socket.SOCK_DGRAM)  # UDP

    def __init__(self) -> None:
        self.sock = socket.socket(*self.sock_args)

    def reset(self):
        """
        Set the aircrat at the beginning of the trajectory
        """
        self.reset_flag = True

    def set_trajs(self, plant: csaf.trace.TimeTrace, controller: csaf.trace.TimeTrace):
        """
        Set trajectories
        """
        self.plant = plant
        self.controller = controller

    def simulate(self, delta_t: float =0.1, speed: float =1.0):
        """
        Start simulation, assuming trajectories are properly set
        """
        self.lag = delta_t
        self.speed = speed
        self.initial_time = time.monotonic()
        self.sim_flag = True

    def start(self):
        """
        Start the main loop of the component
        """
        if self.main_loop is None:
            self.main_loop = threading.Thread(target=self.sim_loop, args=[], daemon=True)
        self.main_loop.start()

    def stop(self):
        """
        Stop the main loop of the component
        """
        self.stopped = True

    def pack_to_struct(self):
        """
        Package the data into a network compatible struct
        """
        pass

    def update_and_send(self, inputs: typing.Optional[typing.List[float]] =None):
        """
        Update the internal values and send a FG compatible packet

        The expected format of `inputs` is:
        - float64 vt 0
        - float64 alpha 1
        - float64 beta 2
        - float64 phi 3
        - float64 theta 4
        - float64 psi 5
        - float64 p 6
        - float64 q 7
        - float64 r 8
        - float64 pn 9
        - float64 pe 10
        - float64 h 11
        - float64 pow 12
        - float64 delta_e 13
        - float64 delta_a 14
        - float64 delta_r 15
        - float64 throttle 16
        """
        pass

    def get_format_string(self) -> str:
        """
        Returns format string for the network packet
        """
        pass

    def sim_loop(self):
        """
        Main simulation loop
        """
        print(f"<{self.__class__.__name__}> Starting main loop!")
        while not self.stopped:
            updated_input = None
            if self.sim_flag:
                real_time = time.monotonic()
                sim_time = (real_time - self.initial_time)*self.speed
                timestamp = next(filter(lambda x: x > sim_time, self.plant.times), None)
                if timestamp:
                    # Plant states
                    idx = self.plant.times.index(timestamp)
                    states = self.plant.states[idx]
                    # Controller output
                    # TODO: if no controller is present, just fill in zeros
                    try:
                        idx = self.controller.times.index(timestamp)
                    except ValueError:
                        idx = 0
                    ctrls = self.controller.states[idx]
                    updated_input = np.concatenate((np.asarray(states),ctrls))
                else:
                    self.sim_flag = False
                    self.lag = self.DEFAULT_DELTA_T
            elif self.reset_flag:
                idx = 0
                states = self.plant.states[idx]
                ctrls = self.controller.states[idx]
                updated_input = np.concatenate((np.asarray(states),ctrls))
                self.reset_flag = False
            self.update_and_send(updated_input)
            time.sleep(self.lag)
        print(f"<{self.__class__.__name__}> Main loop stopped.")