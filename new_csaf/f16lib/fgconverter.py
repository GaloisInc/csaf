"""
ACT3 RL RTA: FlightGear Visualization System

This class extends the Converter class, implementing logic to convert raw 2D Dubins Rejoin task experiment state data
into FlightGear interpretable values. Each environment object - timestep pair contains unique state data which, after
being converted into FlightGear-friendly values, are packaged into an Episode object.

John McCarroll
02/24/2021
"""
import pymap3d as pm
from numpy import deg2rad
import math
from scipy.spatial.transform import Rotation

class Dubins2DConverter():
    FG_FT_IN_M = 3.2808

    @classmethod
    def convert_data(cls, pn_m, pe_m, alt_m, psi_rad, lat0_rad, lon0_rad, ground_level_m) -> list:
        """
        This method takes in a dictionary of "raw" 2D Dubins log data, as read from the LogReader class,
        and returns a populated Episode object.
        """
        # position conversions

        ## ENU to geodetic
        pu = ground_level_m
        latitude, longitude, _alt = \
            pm.enu2geodetic(pe_m, pn_m, pu, lat0_rad,
                            lon0_rad, ground_level_m, ell=None, deg=False)

        ## ENU to ECEF
        # convert position to geocentric (Earth-centered) reference frame
        ecef_x, ecef_y, ecef_z = \
            pm.enu2ecef(pe_m, pn_m, pu, lat0_rad,
                        lon0_rad, ground_level_m, ell=None, deg=False)

        # orientation conversions

        ## ECEF
        # 1st rotation (frame alignment)
        global_rotation = Rotation.from_quat(Dubins2DConverter.quaternion_from_lon_lat(
            lon0_rad, lat0_rad))
        # 2nd rotation (from data)
        ## just yaw for 2D Dubins
        local_rotation = Rotation.from_euler('z', psi_rad, degrees=False)

        # multiply
        rotation = global_rotation * local_rotation

        quaternion = rotation.as_quat()
        angle = 2 * math.acos(quaternion[3])  # cos(a / 2) = w
        direction = quaternion / (math.sin(angle / 2))  # [Vx,Vy,Vz] * sin(a / 2) = [x,y,z]
        ecef_x_orientation = direction[0] * angle
        ecef_y_orientation = direction[1] * angle
        ecef_z_orientation = direction[2] * angle

        return [
            ecef_x, ecef_y, ecef_z,
            ecef_x_orientation,
            ecef_y_orientation,
            ecef_z_orientation
        ]


    @classmethod
    def quaternion_from_lon_lat(cls, lon, lat):
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
