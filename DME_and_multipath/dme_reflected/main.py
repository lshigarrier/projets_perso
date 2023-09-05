import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plots import *


DIST = 20000
DT = 395
h = 100

ELEV = 0.152996
# ELEV = np.pi/4

H = 20000
D = 100
R = 4532

a = np.sqrt(DT*(2*DIST + DT))/2
b = (DIST + DT)/2
c = a

A = 0
B = np.sin(ELEV)
C = np.cos(ELEV)  # not 0
x0 = 0
y0 = -DIST/2 + h*np.sin(ELEV)
z0 = h*np.cos(ELEV)
# A = 1
# B = 1
# C = 1
# x0 = 1000
# y0 = 0
# z0 = 0


def test_projection():
    beacon = Point2D(-DIST/2, 0)
    instance = Instance2D(beacon=beacon, dist=DIST, dt=DIST+DT, elev=ELEV, h=H)
    fig, ax = plt.subplots()
    fig, ax = plot_instance(instance, fig, ax)
    fig, ax = plot_inter(instance, fig, ax)
    plt.show()


def test_circle():
    fig, ax = plt.subplots()
    fig, ax = plot_circle(dist=D, radius=R, fig=fig, ax=ax)
    plt.axis('equal')
    plt.show()


def test_ellipsoid():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig, ax = plot_ellipsoid(a, b, c, fig, ax)
    plt.show()


def test_plane_ellipsoid_intersect():
    const = (A*a + B*b)/(C*c)
    norm_n = np.sqrt((A*a)**2 + (B*b)**2 + (C*c)**2)
    rho = (A*x0 + B*y0 + C*z0)/norm_n
    norm_f = np.sqrt(2 + const**2)
    param = np.linspace(0, 2*np.pi, 100)
    radius = 1-rho**2
    if radius < 0:
        print("No intersection")
        return
    radius = np.sqrt(radius)

    x = a*(rho/norm_n*A*a + radius/norm_f*(np.cos(param) - np.sin(param)/norm_n*(b*B*const+c*C)))
    y = b*(rho/norm_n*B*b + radius/norm_f*(np.cos(param) + np.sin(param)/norm_n*(c*C+a*A*const)))
    z = c*(rho/norm_n*C*c - radius/norm_f*(np.cos(param)*const - np.sin(param)/norm_n*(a*A-b*B)))

    (x1, y1) = np.meshgrid(np.arange(x0-DIST//4, x0+DIST//4+1, DIST//8), np.arange(y0-np.sin(ELEV)*DIST//4, y0+DIST//2+np.cos(ELEV)**2*DIST//2+1, DIST//8))
    z1 = (A * (x0 - x1) + B * (y0 - y1) + C * z0) / C

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig, ax = plot_ellipsoid(a, b, c, fig, ax)
    ax.plot(x, y, z, color='r')
    ax.plot_surface(x1, y1, z1, color='y', alpha=0.3)

    ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    plt.show()

    # print(f"K={const}, rho={rho}, n={norm_n}, f={norm_f}, R={radius}")
    # print(x[-1], y[-1], z[-1])


if __name__ == "__main__":
    test_plane_ellipsoid_intersect()
