import time
import os
import math

import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
import matplotlib.animation as animation
import numpy as np

from plot import plot_component

from components import fgnetfdm

def render_in_flightgear(trajs):
    """
    Render the trajectory in FlightGear
    """
    fdm = fgnetfdm.FGNetFDM()
    fdm.init_from_params({})

    initial_time = time.monotonic()
    while True:
        real_time = time.monotonic()
        sim_time = real_time - initial_time
        timestamp = next(filter(lambda x: x > sim_time, trajs['plant'].times), None)
        if timestamp:
            # Plant states
            idx = trajs['plant'].times.index(timestamp)
            states = trajs['plant'].states[idx]
            # Controller output
            idx = trajs['controller'].times.index(timestamp)
            ctrls = trajs['controller'].states[idx]

            # Update
            input_f16 = np.concatenate((np.asarray(states),ctrls))
            fdm.update_and_send(input_f16)

            # Check if we crashed
            if fdm.agl <= 0:
                print(fdm.agl)
                print("CRASH!")
                break

            # Delay
            time.sleep(fgnetfdm.FGNetFDM.FG_SLEEP_S)
        else:
            print("Done!")
            break

def plot_shield(trajs):
    """
    Plot results for GCSA shield autopilot
    """
    fig, ax = plt.subplots(figsize=(25, 15), nrows=4, ncols=3, sharex=True)
    ax[0][0].set_title("F16 Plant")
    plot_component(ax[0][0], trajs, "plant", "states", 11, "height (ft)")
    plot_component(ax[1][0], trajs, "plant", "states", 0, "airspeed (ft/s)")
    plot_component(ax[2][0], trajs, "plant", "states", 3, "roll (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 4, "pitch (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 5, "yaw (degrees)")
    plot_component(ax[3][0], trajs, "plant", "states", 12, "power (%)")

    ax[0][1].set_title("Low Level Controller")
    plot_component(ax[0][1], trajs, "controller", "outputs", 0, "e ()")
    plot_component(ax[1][1], trajs, "controller", "outputs", 1, "a ()")
    plot_component(ax[2][1], trajs, "controller", "outputs", 2, "r ()")
    plot_component(ax[3][1], trajs, "controller", "outputs", 3, "throttle ()")

    ax[0][2].set_title("Autopilots")
    plot_component(ax[0][2], trajs, "monitor_ap", "outputs", 0, "autopilot selected ()", do_schedule=True)
    plot_component(ax[1][2], trajs, "autopilot", "fdas", 0, "GCAS State ()", do_schedule=True)
    ax[1][2].set_title("GCAS Finite Discrete State")
    ax[2][2].axis('off')
    ax[3][2].axis('off')
    ax[1][2].set_xlabel('Time (s)')

    [ax[3][idx].set_xlabel('Time (s)') for idx in range(2)]

    return fig

def plot_simple(trajs):
    """
    Show results for a simulated F16 plant, controller and an autopilot
    """
    fig, ax = plt.subplots(figsize=(25, 15), nrows=4, ncols=3, sharex=True)
    ax[0][0].set_title("F16 Plant")
    plot_component(ax[0][0], trajs, "plant", "states", 11, "height (ft)")
    plot_component(ax[1][0], trajs, "plant", "states", 0, "airspeed (ft/s)")
    plot_component(ax[2][0], trajs, "plant", "states", 3, "roll (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 4, "pitch (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 5, "yaw (degrees)")
    plot_component(ax[3][0], trajs, "plant", "states", 12, "power (%)")

    ax[0][1].set_title("Low Level Controller")
    plot_component(ax[0][1], trajs, "controller", "outputs", 0, "s0 ()")
    plot_component(ax[1][1], trajs, "controller", "outputs", 1, "s1 ()")
    plot_component(ax[2][1], trajs, "controller", "outputs", 2, "s2 ()")
    plot_component(ax[3][1], trajs, "controller", "outputs", 3, "s3 ()")

    ax[0][2].set_title("Autopilot")
    plot_component(ax[0][2], trajs, "autopilot", "outputs", 0, "a0 ()")
    plot_component(ax[1][2], trajs, "autopilot", "outputs", 1, "a1 ()")
    plot_component(ax[2][2], trajs, "autopilot", "outputs", 2, "a2 ()")
    plot_component(ax[3][2], trajs, "autopilot", "outputs", 3, "a3 ()")

    [ax[3][idx].set_xlabel('Time (s)') for idx in range(3)]

    return fig

def plot_llc(trajs):
    """
    Plot reference tracking of LLC
    """
    fig, ax = plt.subplots(figsize=(10, 6), nrows=3, ncols=1, sharex=True)
    ax[0].set_title("F16 LLC controller")
    plot_component(ax[0], trajs, "autopilot", "outputs", 0, "Nz_ref ()")
    plot_component(ax[0], trajs, "plant", "outputs", 0, "Nz ()")
    plot_component(ax[1], trajs, "autopilot", "outputs", 2, "Ny_r_ref ()")
    plot_component(ax[1], trajs, "plant", "outputs", 1, "Ny+r ()")
    plot_component(ax[2], trajs, "autopilot", "outputs", 1, "ps_ref (rad/s)")
    plot_component(ax[2], trajs, "plant", "states", 6, "ps (rad/s)")
    return fig

def plot_llc_shield(trajs):
    """
    Plot reference tracking of LLC
    """
#     fig, ax = plt.subplots(figsize=(10, 6), nrows=3, ncols=1, sharex=True)
#     ax[0].set_title("F16 LLC controller")
#     plot_component(ax[0], trajs, "autopilot", "outputs", 0, "Nz_ref ()")
#     plot_component(ax[0], trajs, "plant", "outputs", 0, "Nz ()")
#     plot_component(ax[1], trajs, "autopilot", "outputs", 2, "Ny_r_ref ()")
#     plot_component(ax[1], trajs, "plant", "outputs", 1, "Ny+r ()")
#     plot_component(ax[2], trajs, "autopilot", "outputs", 1, "ps_ref (rad/s)")
#     plot_component(ax[2], trajs, "plant", "states", 6, "ps (rad/s)")
#     return fig
    fig, ax = plt.subplots(figsize=(25, 15), nrows=4, ncols=3, sharex=True)
    ax[0][0].set_title("F16 Plant")
    plot_component(ax[0][0], trajs, "plant", "states", 11, "height (ft)")
    plot_component(ax[1][0], trajs, "plant", "states", 1, "alpha (ft/s)")
    plot_component(ax[2][0], trajs, "plant", "states", 3, "roll (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 4, "pitch (degrees)")
    plot_component(ax[2][0], trajs, "plant", "states", 5, "yaw (degrees)")
    plot_component(ax[3][0], trajs, "plant", "states", 12, "power (%)")

    ax[0][1].set_title("Low Level Controller")
    plot_component(ax[0][1], trajs, "shield_llc", "outputs", 0, "s0 ()")
    plot_component(ax[1][1], trajs, "shield_llc", "outputs", 1, "s1 ()")
    plot_component(ax[2][1], trajs, "shield_llc", "outputs", 2, "s2 ()")
    plot_component(ax[3][1], trajs, "shield_llc", "outputs", 3, "s3 ()")

    ax[0][2].set_title("Autopilot")
    plot_component(ax[0][2], trajs, "autopilot", "outputs", 0, "a0 ()")
    plot_component(ax[1][2], trajs, "autopilot", "outputs", 1, "a1 ()")
    plot_component(ax[2][2], trajs, "autopilot", "outputs", 2, "a2 ()")
    plot_component(ax[3][2], trajs, "autopilot", "outputs", 3, "a3 ()")

    [ax[3][idx].set_xlabel('Time (s)') for idx in range(3)]

    return fig


def scale3d(pts, scale_list):
    """
    scale a 3d ndarray of points, and return the new ndarray
    """

    assert len(scale_list) == 3

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        for d in range(3):
            rv[i, d] = scale_list[d] * pts[i, d]

    return rv

def rotate3d(pts, theta, psi, phi):
    """
    rotates an ndarray of 3d points, returns new list
    """

    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    sinPsi = math.sin(psi)
    cosPsi = math.cos(psi)
    sinPhi = math.sin(phi)
    cosPhi = math.cos(phi)

    transform_matrix = np.array([ \
        [cosPsi * cosTheta, -sinPsi * cosTheta, sinTheta], \
        [cosPsi * sinTheta * sinPhi + sinPsi * cosPhi, \
        -sinPsi * sinTheta * sinPhi + cosPsi * cosPhi, \
        -cosTheta * sinPhi], \
        [-cosPsi * sinTheta * cosPhi + sinPsi * sinPhi, \
        sinPsi * sinTheta * cosPhi + cosPsi * sinPhi, \
        cosTheta * cosPhi]])

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        rv[i] = np.dot(pts[i], transform_matrix)

    return rv


def plot3d_anim(trace, filename=None):
    """
    make a 3d plot of the GCAS maneuver
    """

    skip = 1
    full_plot = True

    times = trace.times
    states = trace.states
    assert len(times) == len(states)

    try:
        modes = trace.modes
    except AttributeError:
        modes = [None]*len(times)

    #TODO: Improve this interface?
    op_array = np.vstack(trace.outputs)
    Nz_list, ps_list = op_array[:,0], op_array[:,1]

    if filename == None: # plot to the screen
        skip = 20
        full_plot = False
    elif filename.endswith('.gif'):
        skip = 5
    else:
        skip = 1 # plot every frame


    start = time.time()

    times = times[0::skip]
    states = states[0::skip]
    modes = modes[0::skip]
    ps_list = ps_list[0::skip]
    Nz_list = Nz_list[0::skip]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 45)

    pos_xs = [pt[9] for pt in states]
    pos_ys = [pt[10] for pt in states]
    pos_zs = [pt[11] for pt in states]

    trail_line, = ax.plot([], [], [], color='r', lw=1)

    data = loadmat(os.path.dirname(os.path.realpath(__file__))+ '/f-16.mat')
    f16_pts = data['V']
    f16_faces = data['F']

    plane_polys = Poly3DCollection([], color=None if full_plot else 'k')
    ax.add_collection3d(plane_polys)

    ax.set_xlim([min(pos_xs), max(pos_xs)])
    ax.set_ylim([min(pos_ys), max(pos_xs)])
    ax.set_zlim([min(pos_zs), max(pos_zs)])

    ax.set_xlabel('X [ft]')
    ax.set_ylabel('Y [ft]')
    ax.set_zlabel('Altitude [ft] ')
    frames = len(times)

    # text
    fontsize = 14
    time_text = ax.text2D(0.05, 1.07, "", transform=ax.transAxes, fontsize=fontsize)
    mode_text = ax.text2D(0.95, 1.07, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    alt_text = ax.text2D(0.05, 1.00, "", transform=ax.transAxes, fontsize=fontsize)
    v_text = ax.text2D(0.95, 1.00, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    alpha_text = ax.text2D(0.05, 0.93, "", transform=ax.transAxes, fontsize=fontsize)
    beta_text = ax.text2D(0.95, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    nz_text = ax.text2D(0.05, 0.86, "", transform=ax.transAxes, fontsize=fontsize)
    ps_text = ax.text2D(0.95, 0.86, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    ang_text = ax.text2D(0.5, 0.79, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')

    def anim_func(frame):
        'updates for the animation frame'

        speed = states[frame][0]
        alpha = states[frame][1]
        beta = states[frame][2]
        alt = states[frame][11]
        phi = states[frame][3]
        theta = states[frame][4]
        psi = states[frame][5]
        dx = states[frame][9]
        dy = states[frame][10]
        dz = states[frame][11]

        time_text.set_text('t = {:.2f} sec'.format(times[frame]))
        mode_text.set_text('Mode: {}'.format(modes[frame]))

        alt_text.set_text('h = {:.2f} ft'.format(alt))
        v_text.set_text('V = {:.2f} ft/sec'.format(speed))

        alpha_text.set_text('$\\alpha$ = {:.2f} deg'.format(np.rad2deg(alpha)))
        beta_text.set_text('$\\beta$ = {:.2f} deg'.format(np.rad2deg(beta)))

        nz_text.set_text('$N_z$ = {:.2f} g'.format(Nz_list[frame]))
        ps_text.set_text('$p_s$ = {:.2f} deg/sec'.format(np.rad2deg(ps_list[frame])))

        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(\
            np.rad2deg(phi), np.rad2deg(theta), np.rad2deg(psi)))

        # do trail
        trail_len = 200 // skip
        start_index = max(0, frame-trail_len)
        trail_line.set_data(pos_xs[start_index:frame], pos_ys[start_index:frame])
        trail_line.set_3d_properties(pos_zs[start_index:frame])

        scale = 25
        pts = scale3d(f16_pts, [-scale, scale, scale])

        pts = rotate3d(pts, theta, -psi, phi)

        size = 1000
        minx = dx - size
        maxx = dx + size
        miny = dy - size
        maxy = dy + size
        minz = dz - size
        maxz = dz + size

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        ax.set_zlim([minz, maxz])

        verts = []
        fc = []
        count = 0

        for face in f16_faces:
            face_pts = []

            count = count + 1

            if not full_plot and count % 10 != 0:
                continue

            for index in face:
                face_pts.append((pts[index-1][0] + dx, \
                    pts[index-1][1] + dy, \
                    pts[index-1][2] + dz))

            verts.append(face_pts)
            fc.append('k')

        # draw ground
        if minz <= 0 and maxz >= 0:
            z = 0
            verts.append([(minx, miny, z), (maxx, miny, z), (maxx, maxy, z), (minx, maxy, z)])
            fc.append('0.8')

        plane_polys.set_verts(verts)
        plane_polys.set_facecolors(fc)

        return None

    anim_obj = animation.FuncAnimation(fig, anim_func, frames, interval=30, \
        blit=False, repeat=True)

    if filename is not None:

        if filename.endswith('.gif'):
            print("\nSaving animation to '{}' using 'imagemagick'...".format(filename))
            anim_obj.save(filename, dpi=80, writer='imagemagick')
            print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
        else:
            fps = 50
            codec = 'libx264'

            print("\nSaving '{}' at {:.2f} fps using ffmpeg with codec '{}'.".format(
                filename, fps, codec))

            # if this fails do: 'sudo apt-get install ffmpeg'
            try:
                extra_args = []

                if codec is not None:
                    extra_args += ['-vcodec', str(codec)]

                anim_obj.save(filename, fps=fps, extra_args=extra_args)
                print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
            except AttributeError:
                print("\nSaving video file failed! Is ffmpeg installed? Can you run 'ffmpeg' in the terminal?")
    else:
        return anim_obj
