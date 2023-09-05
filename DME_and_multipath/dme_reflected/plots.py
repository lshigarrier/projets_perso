import math
import scipy
from geometry import *


def plot_instance(instance, fig, ax):
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(instance.a*np.cos(t), instance.b*np.sin(t), c='blue')
    ax.scatter(instance.beacon.x, instance.beacon.y, marker='+', c='black')
    ax.scatter(instance.beacon.x + instance.dist, instance.beacon.y, marker='+', c='black')
    ax.axline((instance.beacon.x, instance.beacon.y), slope=np.tan(-instance.elev), c='green')
    ax.axline((instance.beacon_h.x, instance.beacon_h.y), slope=np.tan(-instance.elev), c='green')
    return fig, ax


def plot_func(instance, func, fig, ax):
    x_array = np.linspace(-1e5, 1e5, 1000)
    y_array = np.linspace(-1e5, 1e5, 10)
    z_array = np.linspace(1e-5, 1e5, 10)
    points = [[], []]
    for x in x_array:
        for y in y_array:
            for z in z_array:
                pts = func(x, y, z)
                points[0].append(pts[0])
                points[1].append(pts[1])
    ax.scatter(points[0], points[1], s=1e-1, c='red')
    return fig, ax


def plot_inter(instance, fig, ax):
    x1_array = np.linspace(-1e5, 1e5, 10)
    y1_array = np.linspace(-1e5, 1e5, 10)
    y2_array = np.linspace(-1e5, 1e5, 10)
    z2_array = np.linspace(1e-5, 1e5, 10)
    points = [[], []]
    for x1 in x1_array:
        for y1 in y1_array:
            for y2 in y2_array:
                for z2 in z2_array:
                    pts = instance.space_to_inter(x1, y1, y2, z2)
                    points[0].append(pts[0])
                    points[1].append(pts[1])
    ax.scatter(points[0], points[1], s=1e-1, c='red')
    return fig, ax


def plot_circle(dist: float, radius: float, fig, ax):
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(radius * np.cos(t), radius * np.sin(t), c='blue')
    r = radius
    x_array = []
    y_array = []
    while r > 0:
        print(r, flush=True)
        n = math.floor(2*np.pi * r / dist)
        a = 2*np.pi / n
        for i in range(n):
            x_array.append(r*np.cos(a*i))
            y_array.append(r*np.sin(a*i))
        r = np.sqrt(r*(r-2) - dist**2)
    ax.scatter(x_array, y_array, s=1e-1, c='red')
    return fig, ax


def plot_ellipsoid(a, b, c, fig, ax):
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    x = a * np.outer(np.ones_like(theta), np.cos(phi))
    z = c * np.outer(np.cos(theta), np.sin(phi))
    y = b * np.outer(np.sin(theta), np.sin(phi))
    ax.plot_wireframe(x, y, z, alpha=0.1)
    max_radius = max(a, b, c)
    '''
    max_range = 2*max_radius
    xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten()
    yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten()
    zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten()
    for x, y, z in zip(xb, yb, zb):
        ax.plot([x], [y], [z], 'w')
    '''
    for axis in 'xy':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
    getattr(ax, 'set_{}lim'.format('z'))((-max_radius*0.8, max_radius*0.8))
    return fig, ax
