import numpy as np
import random
import math

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculating distance in R^3"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

def random_versor():
    """Returns a random versor (vector of length 1)"""
    angle1 = random.uniform(0, 2 * math.pi)
    angle2 = random.uniform(0, math.pi)

    projection = math.cos(angle1)
    x = math.cos(angle2) * projection
    y = math.sin(angle2) * projection
    z = math.sin(angle1)

    return np.array([x, y, z])

def move_in_lattice():
    """Returns a random vector that corresponds to a move in 3D lattice"""
    directions = [-1, 0, 1]
    
    while True:
        move = np.array([random.choice(directions) for _ in range(3)])
        if not np.all(move == 0):
            return move

if __name__ == '__main__':
    pass