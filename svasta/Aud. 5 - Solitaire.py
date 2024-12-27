from searching_framework import *


class Solitaire(Problem):

    def __init__(self, size, w, initial, goal=None):
        super().__init__(initial, goal)
        self.walls = w
        self.size = size

    def successor(self, state):
        successors = dict()
        balls = list(state)
        for x, y in balls:
            # Gore Levo
            new_x, new_y = GoreLevo(x, y, balls, self.walls, self.size)
            if new_x != x or new_y != y:
                new_balls = [(a, b) for a, b in balls if (a, b) != (x - 1, y + 1) and (a, b) != (x, y)]
                new_balls.append((new_x, new_y))
                successors[f'Gore Levo: (x={x},y={y})'] = tuple(new_balls)

            # Gore Desno
            new_x, new_y = GoreDesno(x, y, balls, self.walls, self.size)
            if new_x != x or new_y != y:
                new_balls = [(a, b) for a, b in balls if (a, b) != (x + 1, y + 1) and (a, b) != (x, y)]
                new_balls.append((new_x, new_y))
                successors[f'Gore Desno: (x={x},y={y})'] = tuple(new_balls)

            # Dolu Levo
            new_x, new_y = DoluLevo(x, y, balls, self.walls, self.size)
            if new_x != x or new_y != y:
                new_balls = [(a, b) for a, b in balls if (a, b) != (x - 1, y - 1) and (a, b) != (x, y)]
                new_balls.append((new_x, new_y))
                successors[f'Dolu Levo: (x={x},y={y})'] = tuple(new_balls)

            # Dolu Desno
            new_x, new_y = DoluDesno(x, y, balls, self.walls, self.size)
            if new_x != x or new_y != y:
                new_balls = [(a, b) for a, b in balls if (a, b) != (x + 1, y - 1) and (a, b) != (x, y)]
                new_balls.append((new_x, new_y))
                successors[f'Dolu Desno: (x={x},y={y})'] = tuple(new_balls)

            # Levo
            new_x, new_y = Levo(x, y, balls, self.walls, self.size)
            if new_x != x or new_y != y:
                new_balls = [(a, b) for a, b in balls if (a, b) != (x - 1, y) and (a, b) != (x, y)]
                new_balls.append((new_x, new_y))
                successors[f'Levo: (x={x},y={y})'] = tuple(new_balls)

            # Desno
            new_x, new_y = Desno(x, y, balls, self.walls, self.size)
            if new_x != x or new_y != y:
                new_balls = [(a, b) for a, b in balls if (a, b) != (x + 1, y) and (a, b) != (x, y)]
                new_balls.append((new_x, new_y))
                successors[f'Desno: (x={x},y={y})'] = tuple(new_balls)

        return successors

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        return self.successor(state)[action]

    def goal_test(self, state):
        return state[0] == (self.size//2, self.size - 1)


def GoreLevo(x, y, balls, walls, size):
    if x - 2 >= 0 and \
            y + 2 < size and \
            (x - 1, y + 1) in balls and \
            (x - 2, y + 2) not in walls:
        x -= 2
        y += 2
    return x, y


def GoreDesno(x, y, balls, walls, size):
    if x + 2 < size and \
            y + 2 < size and \
            (x + 1, y + 1) in balls and \
            (x + 2, y + 2) not in walls:
        x += 2
        y += 2
    return x, y


def DoluLevo(x, y, balls, walls, size):
    if x - 2 >= 0 and \
            y - 2 >= 0 and \
            (x - 1, y - 1) in balls and \
            (x - 2, y - 2) not in walls:
        x -= 2
        y -= 2
    return x, y


def DoluDesno(x, y, balls, walls, size):
    if x + 2 < size and \
            y - 2 >= 0 and \
            (x + 1, y - 1) in balls and \
            (x + 2, y - 2) not in walls:
        x += 2
        y -= 2
    return x, y


def Levo(x, y, balls, walls, size):
    if x - 2 >= 0 and \
            (x - 1, y) in balls and \
            (x - 2, y) not in walls:
        x -= 2
    return x, y


def Desno(x, y, balls, walls, size):
    if x + 2 < size and \
            (x + 1, y) in balls and \
            (x + 2, y) not in walls:
        x += 2
    return x, y


if __name__ == '__main__':
    ball = []
    wall = []
    size = int(input())
    n = int(input())
    for i in range(0, n):
        parts = input().split(",")
        ball.append((int(parts[0]), int(parts[1])))

    n = int(input())
    for i in range(0, n):
        parts = input().split(",")
        wall.append((int(parts[0]), int(parts[1])))

    solitaire = Solitaire(size, tuple(wall), tuple(ball))
    print(breadth_first_graph_search(solitaire).solution())
