from constraint import *

if __name__ == '__main__':
    problem = Problem(RecursiveBacktrackingSolver())
    n = int(input())

    participants = ['1', '2', '3', '4', '5']
    leader = ['Team leader']
    dict_participant = dict()
    dict_leader = dict()

    domain_part = []
    domain_leader = []

    for i in range(0, n):
        participant = input().split(" ")
        dict_participant[float(participant[0])] = participant[1]
        domain_part.append(float(participant[0]))
    problem.addVariables(participants, domain_part)

    n = int(input())
    for i in range(0, n):
        participant = input().split(" ")
        dict_leader[float(participant[0])] = participant[1]
        domain_leader.append(float(participant[0]))

    problem.addVariables(leader, domain_leader)

    problem.addConstraint(MaxSumConstraint(100))
    problem.addConstraint(AllDifferentConstraint())

    solutions = problem.getSolutions()
    solutions.sort(key=lambda x: sum(x.values()))
    solution = solutions[-1]

    suma = 0
    for x in solution.items():
        suma += x[1]

    print(f'Total score: {round(suma, 1)}')
    print(f'Team leader: {dict_leader[solution["Team leader"]]}')
    for i in range(1, 6):
        print(f'Participant {i}: {dict_participant[solution[str(i)]]}')

