from searching_framework.utils import Problem
from searching_framework.uninformed_search import *


class Container(Problem):
    def __init__(self, capacities, initial, goal=None):
        super().__init__(initial, goal)
        self.capacities = capacities

    def successor(self, state):
        """За дадена состојба, врати речник од парови {акција : состојба}
        достапни од оваа состојба. Ако има многу следбеници, употребете
        итератор кој би ги генерирал следбениците еден по еден, наместо да
        ги генерирате сите одеднаш.
        :param state: дадена состојба
        :return:  речник од парови {акција : состојба} достапни од оваа
                  состојба
        :rtype: dict
        """
        successors = dict()

        first, second = state
        firstGoal, secondGoal = self.goal

        if first < firstGoal:
            if second > secondGoal:
                delta = min(firstGoal - first, second - secondGoal)
                successors['Preturi od J1 vo J0 '+str(delta)+' litri'] = (first + delta, second - delta)
            else:
                successors['Dopolni go sadot J0 '+str(firstGoal-first)+' litri'] = (first + (firstGoal - first), second)

        if second < secondGoal:
            if first > firstGoal:
                delta = min(firstGoal - first, second - secondGoal)
                successors['Preturi od J0 vo J1 '+str(delta)+' litri'] = (first - delta, second + delta)
            else:
                successors['Dopolni go sadot J1 '+str(secondGoal-second)+' litri'] \
                    = (first, second + (secondGoal - second))

        if first > firstGoal:
            successors['Isprazni go sadot J0 '+str(first-firstGoal)+' litri'] = (first - (first - firstGoal), second)
        if second > secondGoal:
            successors['Isprazni go sadot J1 '+str(second-secondGoal)+' litri'] \
                = (first, second - (second - secondGoal))

        return successors

    def actions(self, state):
        """За дадена состојба state, врати листа од сите акции што може да
        се применат над таа состојба
        :param state: дадена состојба
        :return: листа на акции
        :rtype: list
        """
        return self.successor(state).keys()

    def result(self, state, action):
        """За дадена состојба state и акција action, врати ја состојбата
        што се добива со примена на акцијата над состојбата
        :param state: дадена состојба
        :param action: дадена акција
        :return: резултантна состојба
        """
        return self.successor(state)[action]

    def goal_test(self, state):
        """Врати True ако state е целна состојба. Даденава имплементација
        на методот директно ја споредува state со self.goal, како што е
        специфицирана во конструкторот. Имплементирајте го овој метод ако
        проверката со една целна состојба self.goal не е доволна.
        :param state: дадена состојба
        :return: дали дадената состојба е целна состојба
        :rtype: bool
        """
        return state == self.goal


if __name__ == '__main__':
    container = Container([15, 5], (8, 8), (10, 5))

    print(breadth_first_graph_search(container).solution())
    print(breadth_first_graph_search(container).solve())
