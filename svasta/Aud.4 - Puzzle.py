from searching_framework import *


class Puzzle(Problem):

    def __init__(self, initial, goal=None):
        super().__init__(initial, goal)

    def successor(self, state):
        successors = dict()
        # 0 1 2
        # 3 4 5
        # 6 7 8

        i = state.index('*')

        # if index of * is > 2 it can go up
        if i > 2:
            tmp = list(state)
            tmp[i], tmp[i - 3] = tmp[i - 3], tmp[i]
            new_state = ''.join(tmp)
            successors["UP"] = new_state

        if i < 6:
            tmp = list(state)
            tmp[i], tmp[i + 3] = tmp[i + 3], tmp[i]
            new_state = ''.join(tmp)
            successors["DOWN"] = new_state

        if i % 3 > 0:
            tmp = list(state)
            tmp[i], tmp[i - 1] = tmp[i - 1], tmp[i]
            new_state = ''.join(tmp)
            successors["LEFT"] = new_state

        if i % 3 < 2:
            tmp = list(state)
            tmp[i], tmp[i + 1] = tmp[i + 1], tmp[i]
            new_state = ''.join(tmp)
            successors["RIGHT"] = new_state

        return successors

    def h(self, node):
        sum_manhattan = 0

        for x in '12345678':
            sum_manhattan += manhattan_distance(node.state.index(x), int(x))
        return sum_manhattan

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        return self.successor(state)[action]

    def goal_test(self, state):
        return self.goal == state


def manhattan_distance(current, goal):
    coordinates = {0: (0, 2), 1: (1, 2), 2: (2, 2), 3: (0, 1),
                   4: (1, 1), 5: (2, 1), 6: (0, 0), 7: (1, 0),
                   8: (2, 0)}
    x1, y1 = coordinates[current]
    x2, y2 = coordinates[goal]
    return abs(x1 - x2) + abs(y1 - y2)


if __name__ == "__main__":
    puzzle = Puzzle('*32415678', '*12345678')

    print(greedy_best_first_graph_search(puzzle).solution())
