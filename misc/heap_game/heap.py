import argparse
import itertools
import tkinter
import time
import numpy as np
import matplotlib.pyplot as plt


def increment_increasing(pos, stone):
    flag = True
    if pos[2] < stone:
        pos[2] += 1
    elif pos[1] < stone:
        pos[1] += 1
        pos[2]  = pos[1]
    elif pos[0] < stone:
        pos[0] += 1
        pos[1]  = pos[0]
        pos[2]  = pos[0]
        print(f'Iter: {pos[0]}')
    else:
        flag = False
    return pos, flag


def test_pairs(current_pairs, pairs):
    for pair in current_pairs:
        if pair in pairs:
            return False
    return True


def get_lose(stone):
    lose, lose_plot = [[0, 0, 0]], [(0, 0, 0)]
    current = [0, 0, 0]
    pairs   = {(0, 0)}
    current, flag = increment_increasing(current, stone)
    while flag:
        current_pairs = {(current[0], current[1]), (current[0], current[2]), (current[1], current[2])}
        lost = test_pairs(current_pairs, pairs)
        if lost:
            lose.append(current[:])
            kayak = True
            if (current[0] == current[1]) or (current[0] == current[2]) or (current[1] == current[2]):
                kayak = False
            for perm in itertools.permutations(current):
                if kayak or (perm[:] not in lose_plot):
                    lose_plot.append(perm[:])
            for pair in current_pairs:
                pairs.add(pair)
        current, flag = increment_increasing(current, stone)
    return lose, np.array(lose_plot)


def fractal(current, points, rot, order):
    if order == 0:
        points.append(current.tolist()[0])
    else:
        for i in range(4):
            fractal(current+int(2**order-1)*rot[i], points, rot-rot[i], order-1)


def geo_method(order):
    rot = np.array([[0, 0, 0],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]])
    current = np.array([[0, 0, 0]])
    points = []
    fractal(current, points, rot, order)
    return np.array(points)


def remove_duplicates(points):
    new_points = points[:]
    for pts in points[:]:
        if (pts[0] > pts[1]) or (pts[0] > pts[2]) or (pts[1] > pts[2]):
            new_points.remove(pts)
    return new_points


def losing(x):
    n   = len(x)
    pos = np.array([0, 0, 0])
    old = np.array([0, 0, 0])
    for k in range(n):
        # temp = np.array([0, 0, 0])
        # for i in range(k+1):
            # temp += ((-1)**(k+i)/2*np.array(
            #     [x[i]*(3-x[i]), x[i]*(x[i]-2)*(4*x[i]-10)/3, x[i]*(x[i]-1)*(7-2*x[i])/3])).astype(int)
            # temp += (np.array([x[i]*(3-x[i]), x[i]*(x[i]-2)*(4*x[i]-10)/3, x[i]*(x[i]-1)*(7-2*x[i])/3])/2).astype(int)
        new  = (np.array([x[k]*(3-x[k]), x[k]*(x[k]-2)*(4*x[k]-10)/3, x[k]*(x[k]-1)*(7-2*x[k])/3])/2).astype(int)
        pos += (2**(n-k)-1)*(new - old)
        old  = new[:]
    return pos


def update(x, pivot, base):
    if x[pivot] < base - 1:
        x[pivot] += 1
        for i in range(pivot+1, len(x)):
            x[i] = 0
        return x, False
    if pivot == 0:
         raise RuntimeError
    return x, True


def increment(x, base):
    flag  = True
    pivot = len(x) - 1
    while flag:
        x, flag = update(x, pivot, base)
        pivot -= 1
    return x


def fct_method(order):
    points = []
    x      = [0 for _ in range(order)]
    x[-1]  = -1
    for _ in range(4**order):
        x = increment(x, 4)
        points.append(losing(x).tolist())
    return np.array(points)


def plot_lose(pts, size):
    root = tkinter.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    fig = plt.figure(figsize = (width/200, height/200))
    ax  = fig.add_subplot(projection='3d')
    fig.subplots_adjust(top=1.1, bottom=-.1)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='green', s=size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig


def test_methods(points, test_points):
    flag = True
    if len(test_points) != len(points):
         flag = False
         print(f'Current method length: {len(points)}')
         print(f'Other method length: {len(test_points)}')
    for pts in test_points:
        if pts not in points:
            flag = False
            print(f'Incorrect position: {pts}')
            break
    if flag:
        print('Test result: same')
    else:
        print('Test result: different')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='geo')  # 'geo' or 'pairs'
    parser.add_argument('--stone', type=int, default=31)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--order', type=int, default=5)
    parser.add_argument('--test', type=bool, default=False)
    args   = parser.parse_args()
    for key in vars(args):
        print(f'{key}: {getattr(args, key)}')
    method = args.method
    stone  = args.stone
    size   = args.size
    order  = args.order
    test   = args.test
    tic    = time.time()
    if method == 'pairs':
        print(f'3 heaps with at most {stone} stones')
        points, points_scatter = get_lose(stone)
        if test:
            test_points = geo_method(order)
            test_points = remove_duplicates(test_points.tolist())
            test_methods(points, test_points)
    elif method == 'geo':
        print(f'3 heaps with at most {int(2**order-1)} stones')
        points_scatter = geo_method(order)
        points = remove_duplicates(points_scatter.tolist())
        if test:
            test_points, _ = get_lose(stone)
            test_methods(points, test_points)
    elif method == 'fct':
        print(f'3 heaps with at most {int(2**order-1)} stones')
        points_scatter = fct_method(order)
        points = remove_duplicates(points_scatter.tolist())
        if test:
            test_points = geo_method(order)
            test_points = remove_duplicates(test_points.tolist())
            test_methods(points, test_points)
    else:
        raise NotImplementedError
    print(f'Execution time: {time.time()-tic:.3f}s')
    print(f'Losing positions:\n{points}')
    print(f'Length: {len(points)}')
    _ = plot_lose(points_scatter, size)
    # _ = plot_lose(points, size)
    plt.show()


if __name__ == '__main__':
    main()