"""
ACT3 RL RTA: FlightGear Visualization System

A class to generate UDP Packets that follow MP protocol used by FGMS for rendering multiplayer planes in FlightGear
The packet structure for MP Protocol is defined here: http://wiki.flightgear.org/Multiplayer_protocol#Messages

Author: John McCarroll
03/03/2021
"""
import time
import struct
import socket
import math
import threading

from datetime import datetime, timezone
import pymap3d as pm
import numpy as np
from scipy.spatial.transform import Rotation

from fgconverter import Dubins2DConverter

class FGMSImposter:
    """
    Multiplayer airplane/object

    It can either `idle` and only stays alive, or actively update its position
    """
    # header values
    MAGIC = 0x46474653                  # "FGFS"
    FG_MPP_VERSION = 0x00010001         # 1.1
    MSG_ID = 0x00000007                 # positional data
    REQ_RANGE_NUM = 0x000000a0          # visibility range in nm
    REPLY_PORT = 0x00000000             # deprecated

    DEFAULT_DELTA_T = 0.5

    # Socket info
    DEFAULT_FG_IP = "127.0.0.1"
    DEFAULT_FG_MP_PORT = 5001
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

    # TODO: fold this into some config
    DEFAULT_FG_LAT = 35.802117
    DEFAULT_FG_LON = -117.806717
    DEFAULT_FG_GROUND_LEVEL = 705 # m
    FG_FT_IN_M = 3.2808

    def __init__(self, callsign: str, model_path: str, fallback_model_index: int):
        """
        * callsign -  first 7 chars of vehicle name
        * model_path - path to the model, in the form of `Aircraft/f16/Models/F-16.xml`
                       The base directory is assumed to be set with `--fg-aircraft`, typically
                       set to `--fg-aircraft=/home/mpodhradsky/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020`
        * fallback_model_index - AI model that should be used as a low-res substitute. Typical values:
                      1 - c172
                     56 - Cessna 337
                    516 - F-16
                    702 - air balloon
        """
        self.callsign = callsign[0:7].encode('utf-8')       
        self.lag = FGMSImposter.DEFAULT_DELTA_T
        self.model_path = model_path.encode('utf-8')
        self.fallback_model_index = fallback_model_index

        # init vars
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.x_ori = 0.0
        self.y_ori = 0.0
        self.z_ori = 0.0
        self.lat0 = np.deg2rad(self.DEFAULT_FG_LAT)
        self.lon0 = np.deg2rad(self.DEFAULT_FG_LON)
        self.h0 = self.DEFAULT_FG_GROUND_LEVEL

        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        self.pn = 0.0
        self.pe = 0.0
        self.pu = self.h0

        self.states = None
        self.ctrls = None
        self.stopped = False
        self.sim_flag = False
        self.initial_time = None
        self.plant = None
        self.controller = None
        self.main_loop = threading.Thread(target=self.run_main, args=[], daemon=True)
        self.reset_flag = False


    def set_trajs(self, plant, controller):
        self.plant = plant
        self.controller = controller

    def reset(self):
        """
        Set the aircrat at the beginning of the trajectory
        """
        self.reset_flag = True

    def simulate(self, delta_t=0.1, speed=1.0):
        """
        Commence simulation, assuming trajectories are properly set
        """
        self.lag = delta_t
        self.speed = speed
        self.initial_time = time.monotonic()
        self.sim_flag = True

    def start(self):
        self.main_loop.start()

    def stop(self):
        self.stopped = True

    def run_main(self):
        print("Starting main loop")
        while not self.stopped:
            intruder_input = None
            if self.sim_flag:
                real_time = time.monotonic()
                sim_time = (real_time - self.initial_time)*self.speed
                timestamp = next(filter(lambda x: x > sim_time, self.plant.times), None)
                if timestamp:
                    # Plant states
                    idx = self.plant.times.index(timestamp)
                    states = self.plant.states[idx]
                    # Controller output
                    idx = self.controller.times.index(timestamp)
                    ctrls = self.controller.states[idx]
                    intruder_input = np.concatenate((np.asarray(states),ctrls))
                else:
                    self.sim_flag = False
                    self.lag = FGMSImposter.DEFAULT_DELTA_T
            elif self.reset_flag:
                idx = 0
                states = self.plant.states[idx]
                ctrls = self.controller.states[idx]
                intruder_input = np.concatenate((np.asarray(states),ctrls))
                self.reset_flag = False
            self.update_and_send(intruder_input)
            time.sleep(self.lag)

    def update_and_send(self, inputs=None):
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
        # update:
        if inputs is not None:
            self.phi = inputs[3]
            self.theta = inputs[4]
            self.psi = inputs[5]
            self.pn = inputs[9]
            self.pe = inputs[10]
            self.pu = inputs[11]/self.FG_FT_IN_M

        ecef_x, ecef_y, ecef_z = \
            pm.enu2ecef(self.pe, self.pn, self.pu, self.lat0, self.lon0, self.h0, ell=None, deg=False)
        # position
        self.x = ecef_x
        self.y = ecef_y
        self.z = ecef_z
        
        # orientation
        
        ## ECEF
        # 1st rotation (frame alignment)
        global_rotation = Rotation.from_quat(Dubins2DConverter.quaternion_from_lon_lat(
            self.lon0, self.lat0))
        # 2nd rotation (from data)
        ## just yaw for 2D Dubins
        #local_rotation = Rotation.from_euler('z', self.psi, degrees=False)
        local_rotation = Rotation.from_euler('xyz', [self.phi,self.theta,self.psi], degrees=False)


        # multiply
        rotation = global_rotation * local_rotation

        quaternion = rotation.as_quat()
        angle = 2 * math.acos(quaternion[3])  # cos(a / 2) = w
        direction = quaternion / (math.sin(angle / 2))  # [Vx,Vy,Vz] * sin(a / 2) = [x,y,z]
        ecef_x_orientation = direction[0] * angle
        ecef_y_orientation = direction[1] * angle
        ecef_z_orientation = direction[2] * angle

        self.x_ori = ecef_x_orientation
        self.y_ori = ecef_y_orientation
        self.z_ori = ecef_z_orientation

        # time
        cur_date = datetime.fromtimestamp(time.time(), timezone.utc)
        self.time = (
                cur_date.hour * 3600.0 + cur_date.minute * 60.0 + cur_date.second + cur_date.microsecond * 0.000001)

        # TODO: what to do with control surfaces?

        # send
        self.sock.sendto(self.pack_to_struct(), (FGMSImposter.DEFAULT_FG_IP, FGMSImposter.DEFAULT_FG_MP_PORT))

    def get_format_string(self):
        """
        Because we have so many variables, getting the correct
        format string for packing is handled in this function
        """
        format_string = '!'                 # byte alignment and size (network, standard)
        format_string += 'I I I I I I 8s'   # Header: magic, version, msgID, msgLen, reqRangeNum, ReplyPort, callsign
        format_string += '96s'              # Position: ModelName (96 bytes string)
        format_string += '5d'               # time, lag, pos_x, pos_y, pos_z (5 8-byte doubles)
        format_string += '15f'              # ori_[xyz], vel_[xyz], av[123], acc[123], ang_acc[123]
        format_string += 'I'#'4x'               # padding (4 bytes)

        # # Properties
        format_string += 'HH' # protocol version + value
        format_string += 'II'               # sim/model/fallback-model-index | SHORTINT (00000204)
        #format_string += '10I' # generic float - whaterver it is
        # format_string += '58I'              # 29 ID|Value pairs (4|4 bytes) - not sure if unsign int is right for all?
        # format_string += 'I 9I'             # sim/model/livery/file | STRING (but encoded as 4 byte ints?)
        # format_string += 'I I'              # environment/wildfire/data | STRING (empty... only 1 int)
        # format_string += '3I I'             # 1 prop + scenery/events | STRING (empty)
        # format_string += '12I'              # 6 ID|Value pairs (SHORTINTs)
        # format_string += 'I 9I'             # sim/multiplay/chat | STRING ('Hello')
        # format_string += 'I 5I'             # sim/multiplay/generic/string[4] | STRING ('071')
        # format_string += 'I 17I'            # sim/multiplay/generic/string[5] | STRING ('00--++00--Empty')
        # format_string += 'I I'              # sim/multiplay/generic/string[6] | STRING (empty)
        # format_string += 'I f'              # sim/multiplay/generic/float[0-22] | FLOAT
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += 'I f'
        # format_string += '12I'              # sim/multiplay/generic/ints & shorts + sim/multiplay/mp-clock-mode
        # format_string += '6I'               # sim/multiplay/emesary/bridge[17-19] | STRING (all empty)
        # format_string += '2I'               # sim/model/fallback-model-index | SHORTINT (00000204)
        # format_string += '4I'               # sim/multiplay/comm-transmit-frequency-hz & comm-transmit-power-norm (both zero)
        # format_string += 'I I'              # sim/multiplay/generic/bool[0] | BOOL_ARRAY (hex encoded array of 31 bools)
        # format_string += 'I I'              # sim/multiplay/generic/bool[31] | BOOL_ARRAY (hex encoded array of 31 bools)

        return format_string

    def pack_to_struct(self):
        """
        Actual packing to a FGMS compatible struct
        Resulting struct is 1000 bytes long
        """
        s = struct.Struct(self.get_format_string())
        struct_len = struct.calcsize(self.get_format_string())
        values = (
            # Header
            FGMSImposter.MAGIC,             # FGFS
            FGMSImposter.FG_MPP_VERSION,    # MP Version
            FGMSImposter.MSG_ID,            # Msg ID
            struct_len,
            FGMSImposter.REQ_RANGE_NUM,
            FGMSImposter.REPLY_PORT,
            self.callsign,                  # FGMS identifier
            self.model_path,                # path to model file of vehicle
            self.time,
            self.lag,                       # time between packets

            # Position Data
            self.x,
            self.y,
            self.z,
            # Orientation Data
            self.x_ori,                     # x ori
            self.y_ori,                     # y ori
            self.z_ori,                     # z ori
            0.0,                          # x, y, z velocities* (local frame?)        (was vcas)
            0.0,
            0.0,
            0.0,                          # angular velocities (local frame?)
            0.0,
            0.0,
            0.0,                          # accelerations
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0x1face002, # padding val
            10, # protocol version ID
            2, # protocol version value
            0x000032c8, # sim/model/fallback-model-index
            self.fallback_model_index, # fallback value

            # 0x000a0002,
            # # # Properties
            # 0x000027d8, # 10200 + 4b float
            # 0x3f800000,
            # #10201 + 4b float
            # 0x000027d9,
            # 0x3f800000,  
            # # 10202 + 4b float
            # 0x000027da,
            # 0x3f800000,
            # # 10203 + 4b float
            # 0x000027db,
            # 0x3f800000,
            # # 102034 + 4b float
            # 0x000027dc,
            # 0x00000000,

            # ## control surfaces
            # 0x00000064,             # surface-positions/left-aileron-pos-norm
            # 0x00000000,
            # 0x00000065,             # surface-positions/right-aileron-pos-norm
            # 0x00000000,
            # 0x00000066,             # surface-positions/elevator-pos-norm
            # 0x80000000,
            # 0x00000067,             # surface-positions/rudder-pos-norm
            # 0x00000000,
            # 0x00000068,             # surface-positions/flap-pos-norm
            # 0x00000000,
            # 0x00000069,             # surface-positions/speedbrake-pos-norm
            # 0x00000000,
            # 0x0000006a,             # gear/tailhook/position-norm
            # 0x00000000,
            # 0x0000006c,             # gear/launchbar/state
            # 0x00000000,
            # 0x0000006e,             # canopy/position-norm
            # 0x00000000,
            # 0x000000c8,             # gear/gear[0]/compression-norm
            # 0x00000000,
            # 0x000000c9,             # gear/gear[0]/position-norm
            # 0x00000000,
            # 0x000000d2,             # gear/gear[1]/compression-norm
            # 0x00000000,
            # 0x000000d3,             # gear/gear[1]/position-norm
            # 0x00000000,
            # 0x000000dc,             # gear/gear[2]/compression-norm
            # 0x00000000,
            # 0x000000dd,             # gear/gear[2]/position-norm
            # 0x00000000,
            # 0x0000012c,             # engines/engine[0]/n1
            # 0x00000000,
            # 0x0000012d,             # engines/engine[0]/n2
            # 0x00000000,
            # 0x0000012e,             # engines/engine[0]/rpm
            # 0x45bb8000,
            # 0x0000032a,             # rotors/main/blade[0]/position-deg
            # 0x00000000,
            # 0x0000032b,             # rotors/main/blade[1]/position-deg
            # 0x00000000,
            # 0x0000032c,             # rotors/main/blade[2]/position-deg
            # 0x00000000,
            # 0x0000032d,             # rotors/main/blade[3]/position-deg
            # 0x00000000,
            # 0x00000337,             # rotors/main/blade[3]/flap-deg
            # 0x00000000,
            # 0x000003e9,             # controls/flight/slats
            # 0x00000000,
            # 0x000003ea,             # controls/flight/speedbrake
            # 0x00000000,
            # 0x000003eb,             # controls/flight/spoilers
            # 0x00000000,
            # 0x000003ec,             # controls/gear/gear-down
            # 0x3f800000,
            # 0x000003ed,             # controls/lighting/nav-lights
            # 0x3f800000,
            # 0x000003ee,             # controls/armament/station[0]/jettison-all
            # 0x00000000,

            # 0x0000044d,             # sim/model/livery/file | STRING
            # 0x00000007,             # values: int char encodings of '80-3602'
            # 0x00000038,
            # 0x00000030,
            # 0x0000002d,
            # 0x00000033,
            # 0x00000036,
            # 0x00000030,
            # 0x00000032,
            # 0x00000000,

            # 0x000004b0,             # environment/wildfire/data | STRING
            # 0x00000000,             # values: int char encodings of ''

            # 0x000004b1,             # environment/contrail
            # 0x00000000,

            # 0x00000578,             # scenery/events | STRING
            # 0x00000000,             # values: int char encodings of ''

            # 0x000005dc,             # instrumentation/transponder/transmitted-id
            # 0xffffd8f1,
            # 0x000005dd,             # instrumentation/transponder/altitude
            # 0xffffd8f1,
            # 0x000005de,             # instrumentation/transponder/ident
            # 0x00000000,
            # 0x000005df,             # instrumentation/transponder/inputs/mode
            # 0x00000001,
            # 0x000005e0,             # instrumentation/transponder/ground-bit
            # 0x00000000,
            # 0x000005e1,             # instrumentation/transponder/airspeed-kt
            # 0xffffd8f1,

            # 0x00002712,             # sim/multiplay/chat | STRING
            # 0x00000005,             # Values: int char encodings of 'Hello'
            # 0x00000048,
            # 0x00000065,
            # 0x0000006c,
            # 0x0000006c,
            # 0x0000006f,
            # 0x00000000,
            # 0x00000000,
            # 0x00000000,

            # 0x00002778,             # sim/multiplay/generic/string[4] | STRING
            # 0x00000003,             # Values: int char encodings of '071'
            # 0x00000062,
            # 0x00000037,
            # 0x00000039,
            # 0x00000000,

            # 0x00002779,             # sim/multiplay/generic/string[5] | STRING
            # 0x0000000f,             # Values: int char encodings of '00--++00--Empty'
            # 0x00000030,
            # 0x00000030,
            # 0x0000002d,
            # 0x0000002d,
            # 0x0000002b,
            # 0x0000002b,
            # 0x00000030,
            # 0x00000030,
            # 0x0000002d,
            # 0x0000002d,
            # 0x00000045,
            # 0x0000006d,
            # 0x00000070,
            # 0x00000074,
            # 0x00000079,
            # 0x00000000,

            # 0x0000277a,             # sim/multiplay/generic/string[6] | STRING
            # 0x00000000,             # Values: int char encodings of ''

            # 0x000027d8,             # sim/multiplay/generic/float[0]
            # 0x00000000,
            # 0x000027da,             # sim/multiplay/generic/float[2]
            # 0x00000000,
            # 0x000027db,             # sim/multiplay/generic/float[3]
            # 0x4291b92e,
            # 0x000027dc,             # sim/multiplay/generic/float[4]
            # 0x00000000,
            # 0x000027dd,             # sim/multiplay/generic/float[5]
            # 0x00000000,
            # 0x000027de,             # sim/multiplay/generic/float[6]
            # 0x00000000,
            # 0x000027df,             # sim/multiplay/generic/float[7]
            # 0x00000000,
            # 0x000027e0,             # sim/multiplay/generic/float[8]
            # 0x3a83126f,
            # 0x000027e1,             # sim/multiplay/generic/float[9]
            # 0x3a83126f,
            # 0x000027e2,             # sim/multiplay/generic/float[10]
            # 0x3f800000,
            # 0x000027e3,             # sim/multiplay/generic/float[11]
            # 0x3f600000,
            # 0x000027e4,             # sim/multiplay/generic/float[12]
            # 0x3f200000,
            # 0x000027e5,             # sim/multiplay/generic/float[13]
            # 0x3ecccccd,
            # 0x000027e6,             # sim/multiplay/generic/float[14]
            # 0x3f000000,
            # 0x000027e7,             # sim/multiplay/generic/float[15]
            # 0x3a83126f,
            # 0x000027e8,             # sim/multiplay/generic/float[16]
            # 0x00000000,
            # 0x000027e9,             # sim/multiplay/generic/float[17]
            # 0x00000000,
            # 0x000027eb,             # sim/multiplay/generic/float[19]
            # 0x00000000,
            # 0x000027ec,             # sim/multiplay/generic/float[20]
            # 0x3a83126f,
            # 0x000027ed,             # sim/multiplay/generic/float[21]
            # 0x3a83126f,
            # 0x000027ee,             # sim/multiplay/generic/float[22]
            # 0x00000000,
            # 0x0000283e,             # sim/multiplay/generic/int[2]
            # 0x00000000,
            # 0x00002845,             # sim/multiplay/generic/int[9]
            # 0x00000002,
            # 0x00002909,             # sim/multiplay/generic/short[5]
            # 0x00000000,
            # 0x0000290a,             # sim/multiplay/generic/short[6]
            # 0x00000000,
            # 0x0000290b,             # sim/multiplay/generic/short[7]
            # 0x00000000,
            # 0x00002ed6,             # sim/multiplay/mp-clock-mode
            # 0x00000001,

            # 0x00002ef1,             # sim/multiplay/emesary/bridge[17] | STRING
            # 0x00000000,

            # 0x00002ef2,             # sim/multiplay/emesary/bridge[18]
            # 0x00000000,

            # 0x00002ef3,             # sim/multiplay/emesary/bridge[19]
            # 0x00000000,

            # 0x000032c9,             # sim/multiplay/comm-transmit-frequency-hz
            # 0x00000000,
            # 0x000032ca,             # sim/multiplay/comm-transmit-power-norm
            # 0x00000000,
            # 0x00002af8,             # sim/multiplay/generic/bool[0] | BOOL (BOOLARRAY)
            # 0x00000006,
            # 0x00002b20,             # sim/multiplay/generic/bool[31] | BOOL (BOOLARRAY)
            # 0x00000000,
            )

        packed_data = s.pack(*values)
        return packed_data
