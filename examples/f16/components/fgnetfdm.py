import struct
import typing
import time

import pymap3d as pm

from numpy import deg2rad

from fgbase import FlightGearBase

class FGNetFDM(FlightGearBase):
    """
    Flight Dynamic Model (FDM) class for FlightGear

    FlightGear expects a particar network packet when using external FDM
    The packet structure is defined in: https://github.com/FlightGear/flightgear/blob/next/src/Network/net_fdm.hxx
    """
    FG_NET_FDM_VERSION = 24
    DEFAULT_DELTA_T = 0.01
    DEFAULT_FG_PORT = 5505
    DEFAULT_FG_GENERIC_PORT = 5506

    version: int = FG_NET_FDM_VERSION # increment when data values change
    padding: int = 0 #padding

    # Positions
    longitude: float = 0           # geodetic (radians)
    latitude: float = 0            # geodetic (radians)
    altitude: float = 0            # above sea level (meters)
    agl: float = 0                  # above ground level (meters)
    phi: float = 0                  # roll (radians)
    theta: float = 0                # pitch (radians)
    psi: float = 0                   # yaw or true heading (radians)
    alpha: float = 0               # angle of attack (radians)
    beta: float = 0                # side slip angle (radians)
    pe: float = 0
    pn: float = 0

    # Velocities
    phidot: float = 0               # roll rate (radians/sec)
    thetadot: float = 0             # pitch rate (radians/sec)
    psidot: float = 0               # yaw rate (radians/sec)
    vcas: float = 0                 # calibrated airspeed
    climb_rate: float = 0           # feet per second
    v_north: float = 0              # north velocity in local/body frame, fps
    v_east: float = 0               # east velocity in local/body frame, fps
    v_down: float = 0               # down/vertical velocity in local/body frame, fps
    v_body_u: float = 0             # ECEF velocity in body frame
    v_body_v: float = 0             # ECEF velocity in body frame
    v_body_w: float = 0             # ECEF velocity in body frame

    # Accelerations
    A_X_pilot: float = 0            # X accel in body frame ft/sec^2
    A_Y_pilot: float = 0            # Y accel in body frame ft/sec^2
    A_Z_pilot: float = 0            # Z accel in body frame ft/sec^2
    # Stall
    stall_warning: float = 0        # 0.0 - 1.0 indicating the amount of stall
    slip_deg: float = 0             # slip ball deflection

    # Engine status
    num_engines: int = 0                       # Number of valid engines
    eng_state: typing.List[int] # Engine state (off, cranking, running)
    rpm: typing.List[int]          # Engine RPM rev/min
    fuel_flow: typing.List[int]    # Fuel flow gallons/hr
    fuel_px: typing.List[int]      # Fuel pressure psi
    egt: typing.List[int]          # Exhuast gas temp deg F
    cht: typing.List[int]          # Cylinder head temp deg F
    mp_osi: typing.List[int]       # Manifold pressure
    tit: typing.List[int]          # Turbine Inlet Temperature
    oil_temp: typing.List[int]     # Oil temp deg F
    oil_px: typing.List[int]       # Oil pressure psi

    # Consumables
    num_tanks: int = 0         # Max number of fuel tanks
    fuel_quantity: typing.List[float]

    # Gear status
    num_wheels: int = 0
    wow: typing.List[int]
    gear_pos: typing.List[float]
    gear_steer: typing.List[float]
    gear_compression: typing.List[float]

    # Environment
    cur_time: int = 0           # current unix time
    warp: int = 0                # offset in seconds to unix time
    visibility: float = 10000.0            # visibility in meters (for env. effects)

    # Control surface positions (normalized values)
    elevator: float = 0
    elevator_trim_tab: float = 0
    left_flap: float = 0
    right_flap: float = 0
    left_aileron: float = 0
    right_aileron: float = 0
    rudder: float = 0
    nose_wheel: float = 0
    speedbrake: float = 0
    spoilers: float = 0

    def __init__(self, parameters: dict = {}):
        """
        Set everything to zeros, and set lists to the correct length
        If parameters are supplied, initialize from parameters
        """
        self.eng_state = [0,0,0,0]
        self.rpm = [0,0,0,0]
        self.fuel_flow = [0,0,0,0]
        self.fuel_px = [0,0,0,0]
        self.egt = [0,0,0,0]
        self.cht = [0,0,0,0]
        self.mp_osi = [0,0,0,0]
        self.tit = [0,0,0,0]
        self.oil_temp = [0,0,0,0]
        self.oil_px = [0,0,0,0]

        self.fuel_quantity = [0.0, 0.0, 0.0, 0.0]
        self.wow = [0,0,0]
        self.gear_pos = [0.0, 0.0, 0.0]
        self.gear_steer = [0.0, 0.0, 0.0]
        self.gear_compression = [0.0, 0.0, 0.0]
        self.num_engines = 1

        self.lat0 = deg2rad(float(parameters.get("FG_LAT", FlightGearBase.DEFAULT_FG_LAT)))
        self.lon0 = deg2rad(float(parameters.get("FG_LON", FlightGearBase.DEFAULT_FG_LON)))
        self.h0 = parameters.get("FG_GROUND_LEVEL", FlightGearBase.DEFAULT_FG_GROUND_LEVEL)
        # TODO: the altitude is not matching very well until initialized with traces
        self.agl = self.h0

        self.elevator_max = deg2rad(parameters.get("elevator_max", FlightGearBase.DEFAULT_FG_ELEVATOR_MAX_DEG))
        self.runnder_max = deg2rad(parameters.get("rudder_max", FlightGearBase.DEFAULT_FG_RUDDER_MAX_DEG))
        self.aileron_max = deg2rad(parameters.get("aileron_max", FlightGearBase.DEFAULT_FG_AILERON_MAX_DEG))

        self.FG_PORT = parameters.get("FG_PORT", self.DEFAULT_FG_PORT)
        self.FG_GENERIC_PORT = parameters.get("FG_GENERIC_PORT", self.DEFAULT_FG_GENERIC_PORT)

    def update_and_send(self, inputs: typing.Optional[typing.List[float]] =None):
        if inputs is not None:
            self.vcas = inputs[0]
            self.alpha = inputs[1]
            self.beta = inputs[2]
            self.phi = inputs[3]
            self.theta = inputs[4]
            self.psi = inputs[5]
            self.phidot = inputs[6]
            self.thetadot = inputs[7]
            self.psidot = inputs[8]
            self.pn = inputs[9]
            self.pe = inputs[10]

            self.agl = inputs[11]/FlightGearBase.FG_FT_IN_M
            self.altitude = self.h0 + self.agl

            self.eng_state = [1,0,0,0] # Dummy values
            self.rpm = [6000,1,0,0,] # Dummy values

            self.elevator = -1.0*inputs[13]*self.elevator_max
            self.rudder = inputs[15]*self.runnder_max
            self.left_aileron = inputs[14]*self.aileron_max
            self.right_aileron = self.left_aileron

        lat, lon, _alt = pm.enu2geodetic(self.pe, self.pn, self.agl, self.lat0, self.lon0, self.h0, ell=None, deg=False)
        self.latitude = lat
        self.longitude = lon
        self.cur_time = int(time.time())

        # Send FDM
        self.sock.sendto(self.pack_to_struct(), (FlightGearBase.DEFAULT_FG_IP, self.FG_PORT))

        # Send Control surfaces
        # flaperons is in radians:
        # surface-positions/leftrad2
        # surface-positions/leftrad
        # surface-positions/rightrad2
        # surface-positions/rightrad
        val = struct.pack('!fffff', -1*self.elevator,
            self.left_aileron, self.left_aileron, self.right_aileron, self.right_aileron)
        self.sock.sendto(val, (FlightGearBase.DEFAULT_FG_IP, self.FG_GENERIC_PORT))

    def get_format_string(self) -> str:
        """
        Because we have so many variables, getting the correct
        format string for packing is handled in this function
        """
        format_string = '!'
        format_string +=  'I I' # version, padding
        format_string +=  'd d d' # lon,lat,alt
        format_string +=  'f f f' # agl, phi, theta
        format_string +=  'f f f' # psi, alpha, beta
        format_string +=  'f f f' # phidot, thetadot, psidot
        format_string +=  'f f'   # vcasm climb_rate
        format_string +=  'f f f' # v_n, v_e, v_d
        format_string +=  'f f f' # v_u, v_v, v_w
        format_string +=  'f f f' # A_Y_pilot, A_Y_pilot, A_Z_pilot
        format_string +=  'f f' # stall, slip
        format_string +=  'I' # num_engines
        format_string +=  'IIII' # eng_state[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # rpm[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # fuel_flow[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # fuel_px[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # egt[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # cht[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # mp_osi[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # tit[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # oil_temp[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'ffff' # oil_px[FG_NET_FDM_MAX_ENGINES]
        format_string +=  'I' # num_tanks
        format_string +=  'ffff' # fuel_quantity[FG_NET_FDM_MAX_TANKS]
        format_string +=  'I' # num wheels
        format_string +=  'III' # wow[FG_NET_FDM_MAX_WHEELS]
        format_string +=  'fff' # gear_pos[FG_NET_FDM_MAX_WHEELS]
        format_string +=  'fff' # gear_steer[FG_NET_FDM_MAX_WHEELS]
        format_string +=  'fff' # gear_compression[FG_NET_FDM_MAX_WHEELS]
        format_string +=  'I' # cur time
        format_string +=  'i' # warp
        format_string +=  'f' # visibility
        format_string +=  'f f f f' # elevator, elevator_trim, left_flap, right_flap
        format_string +=  'f f f' # left aileron, right aileron, rudder
        format_string +=  'f f f' # nose_wheel, speedbrake, spoilers
        return format_string

    def unpack_from_struct(self, data):
        """
        Unpack data from a network compatible struct
        """
        s = struct.Struct(self.get_format_string())
        values = s.unpack(data)
        idx = 2
        self.longitude = values[idx]; idx += 1 # lon,lat,alt
        self.latitude = values[idx]; idx += 1
        self.altitude = values[idx]; idx += 1
        self.agl = values[idx]; idx += 1 # agl, phi, theta
        self.phi = values[idx]; idx += 1
        self.theta = values[idx]; idx += 1
        self.psi = values[idx]; idx += 1 # psi, alpha, beta
        self.alpha = values[idx]; idx += 1
        self.beta = values[idx]; idx += 1
        self.phidot = values[idx]; idx += 1# phidot, thetadot, psidot
        self.thetadot = values[idx]; idx += 1
        self.psidot = values[idx]; idx += 1
        self.vcas = values[idx]; idx += 1 # vcas climb_rate
        self.climb_rate = values[idx]; idx += 1
        self.v_north = values[idx]; idx += 1 # v_n, v_e, v_d
        self.v_east = values[idx]; idx += 1
        self.v_down = values[idx]; idx += 1
        self.v_body_u = values[idx]; idx += 1 # v_u, v_v, v_w
        self.v_body_v = values[idx]; idx += 1
        self.v_body_w = values[idx]; idx += 1
        self.A_X_pilot = values[idx]; idx += 1 # A_X_pilot, A_Y_pilot, A_Z_pilot
        self.A_Y_pilot = values[idx]; idx += 1
        self.A_Z_pilot = values[idx]; idx += 1
        self.stall_warning = values[idx]; idx += 1 # stall, slip
        self.slip_deg = values[idx]; idx += 1
        self.num_engines = values[idx]; idx += 1 # num_engines
        self.eng_state[0] = values[idx]; idx += 1 # eng_state[FG_NET_FDM_MAX_ENGINES]
        self.eng_state[1] = values[idx]; idx += 1
        self.eng_state[2] = values[idx]; idx += 1
        self.eng_state[3] = values[idx]; idx += 1
        self.rpm[0] = values[idx]; idx += 1 # rpm[FG_NET_FDM_MAX_ENGINES]
        self.rpm[1] = values[idx]; idx += 1
        self.rpm[2] = values[idx]; idx += 1
        self.rpm[3] = values[idx]; idx += 1
        self.fuel_flow[0] = values[idx]; idx += 1 # fuel_flow[FG_NET_FDM_MAX_ENGINES]
        self.fuel_flow[1] = values[idx]; idx += 1
        self.fuel_flow[2] = values[idx]; idx += 1
        self.fuel_flow[3] = values[idx]; idx += 1
        self.fuel_px[0] = values[idx]; idx += 1 # fuel_px[FG_NET_FDM_MAX_ENGINES]
        self.fuel_px[1] = values[idx]; idx += 1
        self.fuel_px[2] = values[idx]; idx += 1
        self.fuel_px[3] = values[idx]; idx += 1
        self.egt[0] = values[idx]; idx += 1 # egt[FG_NET_FDM_MAX_ENGINES]
        self.egt[1] = values[idx]; idx += 1
        self.egt[2] = values[idx]; idx += 1
        self.egt[3] = values[idx]; idx += 1
        self.cht[0] = values[idx]; idx += 1 # cht[FG_NET_FDM_MAX_ENGINES]
        self.cht[1] = values[idx]; idx += 1
        self.cht[2] = values[idx]; idx += 1
        self.cht[3] = values[idx]; idx += 1
        self.mp_osi[0] = values[idx]; idx += 1 # mp_osi[FG_NET_FDM_MAX_ENGINES]
        self.mp_osi[1] = values[idx]; idx += 1
        self.mp_osi[2] = values[idx]; idx += 1
        self.mp_osi[3] = values[idx]; idx += 1
        self.tit[0] = values[idx]; idx += 1 # tit[FG_NET_FDM_MAX_ENGINES]
        self.tit[1] = values[idx]; idx += 1
        self.tit[2] = values[idx]; idx += 1
        self.tit[3] = values[idx]; idx += 1
        self.oil_temp[0] = values[idx]; idx += 1 # oil_temp[FG_NET_FDM_MAX_ENGINES]
        self.oil_temp[1] = values[idx]; idx += 1
        self.oil_temp[2] = values[idx]; idx += 1
        self.oil_temp[3] = values[idx]; idx += 1
        self.oil_px[0] = values[idx]; idx += 1 # oil_px[FG_NET_FDM_MAX_ENGINES]
        self.oil_px[1] = values[idx]; idx += 1
        self.oil_px[2] = values[idx]; idx += 1
        self.oil_px[3] = values[idx]; idx += 1
        self.num_tanks = values[idx]; idx += 1 # num_tanks
        self.fuel_quantity[0] = values[idx]; idx += 1 # fuel_quantity[FG_NET_FDM_MAX_TANKS]
        self.fuel_quantity[1] = values[idx]; idx += 1
        self.fuel_quantity[2] = values[idx]; idx += 1
        self.fuel_quantity[3] = values[idx]; idx += 1
        self.num_wheels = values[idx]; idx += 1 # num wheels
        self.wow[0] = values[idx]; idx += 1 # wow[FG_NET_FDM_MAX_WHEELS]
        self.wow[1] = values[idx]; idx += 1
        self.wow[2] = values[idx]; idx += 1
        self.gear_pos[0] = values[idx]; idx += 1 # gear_pos[FG_NET_FDM_MAX_WHEELS]
        self.gear_pos[1] = values[idx]; idx += 1
        self.gear_pos[2] = values[idx]; idx += 1
        self.gear_steer[0] = values[idx]; idx += 1 # gear_steer[FG_NET_FDM_MAX_WHEELS]
        self.gear_steer[1] = values[idx]; idx += 1
        self.gear_steer[2] = values[idx]; idx += 1
        self.gear_compression[0] = values[idx]; idx += 1 # gear_compression[FG_NET_FDM_MAX_WHEELS]
        self.gear_compression[1] = values[idx]; idx += 1
        self.gear_compression[2] = values[idx]; idx += 1
        self.cur_time = values[idx]; idx += 1 # cur time
        self.warp = values[idx]; idx += 1 # warp
        self.visibility = values[idx]; idx += 1 # visibility
        self.elevator = values[idx]; idx += 1 # elevator, elevator_trim, left_flap, right_flap
        self.elevator_trim_tab = values[idx]; idx += 1
        self.left_flap = values[idx]; idx += 1
        self.right_flap = values[idx]; idx += 1
        self.left_aileron = values[idx]; idx += 1 # left aileron, right aileron, rudder
        self.right_aileron = values[idx]; idx += 1
        self.rudder = values[idx]; idx += 1
        self.nose_wheel = values[idx]; idx += 1 # nose_wheel, speedbrake, spoilers
        self.speedbrake = values[idx]; idx += 1
        self.spoilers = values[idx]; idx += 1

    def pack_to_struct(self):
        """
        Actual packing to a network compatible struct
        Resulting struct is 408 bytes long
        """
        s = struct.Struct(self.get_format_string())
        values = (self.FG_NET_FDM_VERSION, 0, # version, padding
            self.longitude, # lon,lat,alt
            self.latitude,
            self.altitude, 
            self.agl, # agl, phi, theta
            self.phi,
            self.theta,
            self.psi, # psi, alpha, beta
            self.alpha,
            self.beta,
            self.phidot,# phidot, thetadot, psidot
            self.thetadot,
            self.psidot,
            self.vcas, # vcas climb_rate
            self.climb_rate,
            self.v_north, # v_n, v_e, v_d
            self.v_east,
            self.v_down,
            self.v_body_u, # v_u, v_v, v_w
            self.v_body_v,
            self.v_body_w,
            self.A_X_pilot, # A_X_pilot, A_Y_pilot, A_Z_pilot
            self.A_Y_pilot,
            self.A_Z_pilot,
            self.stall_warning, # stall, slip
            self.slip_deg,
            self.num_engines, # num_engines
            self.eng_state[0], # eng_state[FG_NET_FDM_MAX_ENGINES]
            self.eng_state[1],
            self.eng_state[2],
            self.eng_state[3],
            self.rpm[0], # rpm[FG_NET_FDM_MAX_ENGINES]
            self.rpm[1],
            self.rpm[2],
            self.rpm[3],
            self.fuel_flow[0], # fuel_flow[FG_NET_FDM_MAX_ENGINES]
            self.fuel_flow[1],
            self.fuel_flow[2],
            self.fuel_flow[3],
            self.fuel_px[0], # fuel_px[FG_NET_FDM_MAX_ENGINES]
            self.fuel_px[1],
            self.fuel_px[2],
            self.fuel_px[3],
            self.egt[0], # egt[FG_NET_FDM_MAX_ENGINES]
            self.egt[1],
            self.egt[2],
            self.egt[3],
            self.cht[0], # cht[FG_NET_FDM_MAX_ENGINES]
            self.cht[1],
             self.cht[2],
            self.cht[3],
            self.mp_osi[0], # mp_osi[FG_NET_FDM_MAX_ENGINES]
            self.mp_osi[1],
            self.mp_osi[2],
            self.mp_osi[3],
            self.tit[0], # tit[FG_NET_FDM_MAX_ENGINES]
            self.tit[1],
            self.tit[2],
            self.tit[3],
            self.oil_temp[0], # oil_temp[FG_NET_FDM_MAX_ENGINES]
            self.oil_temp[1],
            self.oil_temp[2],
            self.oil_temp[3],
            self.oil_px[0], # oil_px[FG_NET_FDM_MAX_ENGINES]
            self.oil_px[1],
            self.oil_px[2],
            self.oil_px[3],
            self.num_tanks, # num_tanks
            self.fuel_quantity[0], # fuel_quantity[FG_NET_FDM_MAX_TANKS]
            self.fuel_quantity[1],
            self.fuel_quantity[2],
            self.fuel_quantity[3],
            self.num_wheels, # num wheels
            self.wow[0], # wow[FG_NET_FDM_MAX_WHEELS]
            self.wow[1],
            self.wow[2],
            self.gear_pos[0], # gear_pos[FG_NET_FDM_MAX_WHEELS]
            self.gear_pos[1],
            self.gear_pos[2],
            self.gear_steer[0], # gear_steer[FG_NET_FDM_MAX_WHEELS]
            self.gear_steer[1],
            self.gear_steer[2],
            self.gear_compression[0], # gear_compression[FG_NET_FDM_MAX_WHEELS]
            self.gear_compression[1],
            self.gear_compression[2],
            self.cur_time, # cur time
            self.warp, # warp
            self.visibility, # visibility
            self.elevator, # elevator, elevator_trim, left_flap, right_flap
            self.elevator_trim_tab,
            self.left_flap,
            self.right_flap,
            self.left_aileron, # left aileron, right aileron, rudder
            self.right_aileron,
            self.rudder,
            self.nose_wheel, # nose_wheel, speedbrake, spoilers
            self.speedbrake,
            self.spoilers)
        packed_data = s.pack(*values)
        return packed_data
