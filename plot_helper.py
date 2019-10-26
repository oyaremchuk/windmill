import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def prepare_plot(title=None):
    fig = plt.figure(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel('X Label', color='r')
    ax.set_ylabel('Y Label', color='g')
    ax.set_zlabel('Z Label', color='b')

    return ax


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_point(ax, point, color):
    ax.scatter(point[0], point[1], point[2], color=color)

def plot_plane(ax, point, normal, color):
    x = np.linspace(point[0] - 20, point[0] + 20, 10)
    y = np.linspace(point[1] - 20, point[1] + 20, 10)

    X, Y = np.meshgrid(x, y)
    if (normal[2] == 0):
        Z = point[2] + 0 * X + 0 * Y
    else:
        Z = - (normal[0] * (X - point[0]) + normal[1] * (Y - point[1])) / normal[2] + point[2]
    ax.plot_surface(X, Y, Z, alpha=0.2, color=color)

def plot_vector(ax, center, vector, color):
    ax.quiver(center[0], center[1], center[2], vector[0], vector[1], vector[2], color=color)

def plot_line(ax, p1, p2, color,linestyle):
    x = np.concatenate((p1[0], p2[0]))
    y = np.concatenate((p1[1], p2[1]))
    z = np.concatenate((p1[2], p2[2]))
    ax.plot(xs=x, ys=y, zs=z, color=color, linestyle=linestyle)

def plot_camera(ax, cameraPosition, R, r_multiplier):
    plot_vector(ax, cameraPosition, R[0] * r_multiplier, colors['r'])
    plot_vector(ax, cameraPosition, R[1] * r_multiplier, colors['g'])
    plot_vector(ax, cameraPosition, R[2] * r_multiplier, colors['b'])