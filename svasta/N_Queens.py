from constraint import *


def horizontal_vertical_diagonal(a, b):
    if a[0] == b[0] or a[1] == b[1] or abs(a[0] - b[0]) == abs(a[1] - b[1]):
        return False
    return True


if __name__ == '__main__':
    problem = Problem(BacktrackingSolver())

    n = int(input())
    variables = []
    domain = []

    for i in range(1, n + 1):
        variables.append(i)

    for i in range(0, n):
        for j in range(0, n):
            domain.append((i, j))

    problem.addVariables(variables, domain)

    # ---Tuka dodadete gi ogranichuvanjata----------------
    for x in variables:
        for y in variables:
            if x == y:
                continue
            problem.addConstraint(horizontal_vertical_diagonal, (x, y))
    # ----------------------------------------------------
    if n <= 6:
        print(len(problem.getSolutions()))
    else:
        print(problem.getSolution())
