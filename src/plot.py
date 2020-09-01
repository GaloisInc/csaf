"""
Stanley Bak
Python code for F-16 animation video output
"""

import os
import math
import time
import numpy as np
from numpy import rad2deg

from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.io import loadmat


def scale3d(pts, scale_list):
    'scale a 3d ndarray of points, and return the new ndarray'

    assert len(scale_list) == 3

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        for d in range(3):
            rv[i, d] = scale_list[d] * pts[i, d]

    return rv

def rotate3d(pts, theta, psi, phi):
    'rotates an ndarray of 3d points, returns new list'

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

        alpha_text.set_text('$\\alpha$ = {:.2f} deg'.format(rad2deg(alpha)))
        beta_text.set_text('$\\beta$ = {:.2f} deg'.format(rad2deg(beta)))

        nz_text.set_text('$N_z$ = {:.2f} g'.format(Nz_list[frame]))
        ps_text.set_text('$p_s$ = {:.2f} deg/sec'.format(rad2deg(ps_list[frame])))

        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(\
            rad2deg(phi), rad2deg(theta), rad2deg(psi)))

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


def plot_spec2plot_data_list(plot_spec, traj):
    plot_data_list = [(traj[e[0]], e[1]) for e in plot_spec]
    return plot_data_list

def plot2d_traj(traj, plot_spec, title, filename=None, fullscreen=False):
    plot_data_list = plot_spec2plot_data_list(plot_spec, traj)
    return plot2d(traj.times, plot_data_list, title)

def plot2d_multiple_simple(trajs, plot_spec, legend=None, title=None, filename=None, fullscreen=False):
    """
    A simpler version of `plot2d_multiple` that doesn't require the demo registry
    """
    fig = None
    ax = None

    for traj in trajs:
        plot_data_list = plot_spec2plot_data_list(plot_spec, traj)
        fig, ax = plot2d(traj.times, plot_data_list, None, fig, ax)

    if legend:
        ax[0].legend(legend)

    # set the time labels
    ax[-1].set_xlabel('Time [s]', fontsize=16)

    # apply grid to each axis
    [a.grid() for a in ax]

    # set title
    if title:
        ax[0].set_title(f"F16 Demo Plot for {title}")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        if fullscreen:
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()

        plt.show()


#TODO: should only take trajectories and elements
def plot2d_multiple(demo_registry, trajs, plot_spec, demo_ids, filename=None, fullscreen=False):
    # TODO: Modify this by extending trace class
    # plotspec: [(element_name, [(idx, label)])]
    fig = None
    ax = None
    do_legend = True
    if demo_ids is None:
        demo_ids = [None, ] * len(trajs)
        do_legend = False
    if len(demo_ids) == 1:
        do_legend = False
    assert len(demo_ids) == len(trajs)

    for traj, tid in zip(trajs, demo_ids):
        plot_data_list = plot_spec2plot_data_list(plot_spec, traj)
        fig, ax = plot2d(traj.times, plot_data_list, tid, fig, ax)

    if do_legend:
        ax[0].legend([demo_registry.get(t, t).descr for t in demo_ids])

    # set the time labels
    ax[-1].set_xlabel('Time [s]', fontsize=16)

    # apply grid to each axis
    [a.grid() for a in ax]

    # set title
    if len(demo_ids) > 1:
        ax[0].set_title("F16 Demo Plot for Multiple Controllers")
    else:
        ax[0].set_title(f"F16 Demo Plot for {demo_registry[demo_ids[0]].descr}")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        if fullscreen:
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()

        plt.show()

def plot2d(times, plot_data_list, id_name, fig=None, ax=None):
    """plot state variables in 2d

    plot data list of is a list of (values_list, var_data),
    where values_list is an 2-d array, the first is time step, the second is a state vector
    and each var_data is a list of tuples: (state_index, label)
    """

    labels = [var_name for (_, var_data) in plot_data_list for _, var_name in var_data]
    num_plots = len(labels)

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, num_plots*2), nrows=num_plots, sharex=True)

    for plot_index, label in enumerate(labels):
        ax[plot_index].tick_params(axis='both', which='major', labelsize=16)

        sum_plots = 0
        states = None
        state_var_data = None

        for values_list, var_data in plot_data_list:
            if plot_index < sum_plots + len(var_data):
                states = values_list
                state_var_data = var_data
                break

            sum_plots += len(var_data)

        state_index, label = state_var_data[plot_index - sum_plots]

        if state_index == 0 and isinstance(states[0], float): # state is just a single number
            ys = states
        else:
            ys = [state[state_index] for state in states]

        ax[plot_index].plot(times, ys, '-', label=id_name)
        ax[plot_index].set_ylabel(label, fontsize=16)
        ax[plot_index].set_xlim(min(times), max(times))

    return fig, ax

def plot2d_live(labels, fig=None, ax=None, lines=None, ys=[], times=[]):
    """
    Live plot2d except plots live
    """
    if fig is None:
        num_plots = len(labels)
        fig, ax = plt.subplots(figsize=(10, num_plots*2), nrows=num_plots, sharex=True)
        lines = []

        for plot_index, label in enumerate(labels):
            ax[plot_index].tick_params(axis='both', which='major', labelsize=16)
            ax[plot_index].set_ylabel(label, fontsize=16)
            line, = ax[plot_index].plot([])
            lines.append(line)
            ax[plot_index].set_autoscale_on(True)
            ax[plot_index].autoscale_view(True,True,True)
            ys.append([])
    else:
        for plot_index in range(0,len(lines)):
            lines[plot_index].set_data(times, ys[plot_index])
            ax[plot_index].set_xlim(0, max(times))
            ax[plot_index].set_ylim(min(ys[plot_index]), max(ys[plot_index]))
    return fig, ax, lines, ys, times
