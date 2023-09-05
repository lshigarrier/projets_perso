import numpy as np


class Point2D:

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, real: float):
        return Point2D(real*self.x, real*self.y)

    def __abs__(self):
        return np.sqrt(self.x**2 + self.y**2)


class Point3D:

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, real: float):
        return Point3D(real*self.x, real*self.y, real*self.z)

    def __abs__(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)


def sigmoid(x: float, a: float, b: float):
    return (b-a)/(1+np.exp(-x)) + a


class Instance2D:

    def __init__(self, beacon: Point2D, dist: float, dt: float, elev: float, h: float):
        """
        beacon: location of the DME beacon as a Point2D. It is assumed that beacon.y = 0
        dist: distance between the beacon and the aircraft
        dt: delay
        h: height between the lower plane (i.e., the ground) and the higher plane
        elev: elevation angle in [0, pi/2] radians
        """
        self.beacon = beacon
        self.dist = dist
        self.dt = dt
        self.h = h
        self.elev = elev
        self.co = np.cos(elev)
        self.si = np.sin(elev)
        self.a = dist/2 + dt
        self.b = np.sqrt(dt*dist/2)  # approx
        self.new_org = beacon + Point2D(self.si, self.co)*(h/2)
        self.beacon_h = beacon + Point2D(self.si, self.co)*h

    def space_to_disk(self, x: float, y: float, z: float):
        p = Point3D(x, y, z)
        r = abs(p)
        return self.a*p.x/r, self.b*p.y/r

    def space_to_strip(self, x: float, y: float, z:float):
        p = Point2D(y, z)
        r = abs(p)
        return self.co*x + self.si*y/r*self.h/2 + self.new_org.x, self.co*y/r*self.h/2 - self.si*x + self.new_org.y

    def space_to_inter(self, x1: float, y1: float, y2: float, z2: float):
        r2 = abs(Point2D(y2, z2))
        temp = (self.h*y2/2/r2 + self.si*self.new_org.x + self.co*self.new_org.y)**2
        coef_a = (self.si*self.a)**2 - temp
        coef_b = 2*self.si*self.a*self.co*self.b*y1
        coef_c = y1**2*((self.co*self.b)**2 - temp)
        delta = coef_b**2 - 4*coef_a*coef_c
        root1 = (coef_b - np.sqrt(delta))/(2*coef_a)
        root2 = (coef_b + np.sqrt(delta)) / (2 * coef_a)
        if coef_a <= 0:
            x1 = sigmoid(x1, root1, root2)
        else:
            mean = (root1+root2)/2
            if x1 <= mean:
                x1 -= mean
            else:
                x1 += mean
        z1 = np.sqrt((self.si*self.a*x1+self.co*self.b*y1)**2/temp - x1**2 - y1**2)
        r1 = abs(Point3D(x1, y1, z1))
        # x2 = (self.a*self.x1/r1 - self.si*self.h*self.y2/r2/2 - self.new_org.x)/self.si
        return self.a * x1 / r1, self.b * y1 / r1

