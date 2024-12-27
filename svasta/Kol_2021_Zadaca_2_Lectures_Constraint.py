from constraint import *


def twoHoursTerm(x, y):
    a = int(x[-2]) * 10 + int(x[-1])
    b = int(y[-2]) * 10 + int(y[-1])
    day_1 = x[:3]
    day_2 = y[:3]
    if day_2 != day_1:
        return True
    return abs(a - b) >= 2


def mlConstraint(x, y):
    a = int(x[-2]) * 10 + int(x[-1])
    b = int(y[-2]) * 10 + int(y[-1])

    return a != b


if __name__ == '__main__':
    problem = Problem(BacktrackingSolver())
    casovi_AI = int(input())
    casovi_ML = int(input())
    casovi_R = int(input())
    casovi_BI = int(input())

    AI_predavanja_domain = ["Mon_11", "Mon_12", "Wed_11", "Wed_12", "Fri_11", "Fri_12"]
    ML_predavanja_domain = ["Mon_12", "Mon_13", "Mon_15", "Wed_12", "Wed_13", "Wed_15", "Fri_11", "Fri_12", "Fri_15"]
    R_predavanja_domain = ["Mon_10", "Mon_11", "Mon_12", "Mon_13", "Mon_14", "Mon_15", "Wed_10", "Wed_11", "Wed_12",
                           "Wed_13", "Wed_14", "Wed_15", "Fri_10", "Fri_11", "Fri_12", "Fri_13", "Fri_14", "Fri_15"]
    BI_predavanja_domain = ["Mon_10", "Mon_11", "Wed_10", "Wed_11", "Fri_10", "Fri_11"]

    AI_vezbi_domain = ["Tue_10", "Tue_11", "Tue_12", "Tue_13", "Thu_10", "Thu_11", "Thu_12", "Thu_13"]
    ML_vezbi_domain = ["Tue_11", "Tue_13", "Tue_14", "Thu_11", "Thu_13", "Thu_14"]
    BI_vezbi_domain = ["Tue_10", "Tue_11", "Thu_10", "Thu_11"]

    # ---Tuka dodadete gi promenlivite--------------------
    all_subjects = []
    ml = []
    for i in range(0, casovi_AI):
        problem.addVariable(f"AI_cas_{i + 1}", AI_predavanja_domain)
        all_subjects.append(f"AI_cas_{i + 1}")
    for i in range(0, casovi_R):
        problem.addVariable(f"R_cas_{i + 1}", R_predavanja_domain)
        all_subjects.append(f"R_cas_{i + 1}")
    for i in range(0, casovi_BI):
        problem.addVariable(f"BI_cas_{i + 1}", BI_predavanja_domain)
        all_subjects.append(f"BI_cas_{i + 1}")
    for i in range(0, casovi_ML):
        problem.addVariable(f"ML_cas_{i + 1}", ML_predavanja_domain)
        all_subjects.append(f"ML_cas_{i + 1}")
        ml.append(f"ML_cas_{i + 1}")
    problem.addVariable("AI_vezbi", AI_vezbi_domain)
    problem.addVariable("ML_vezbi", ML_vezbi_domain)
    problem.addVariable("BI_vezbi", BI_vezbi_domain)
    all_subjects.append("AI_vezbi")
    all_subjects.append("ML_vezbi")
    all_subjects.append("BI_vezbi")
    ml.append("ML_vezbi")
    # ---Tuka dodadete gi ogranichuvanjata----------------
    for x in all_subjects:
        for y in all_subjects:
            if x != y:
                problem.addConstraint(twoHoursTerm, (x, y))

    for x in ml:
        for y in ml:
            if x != y:
                problem.addConstraint(mlConstraint, (x, y))
    # ----------------------------------------------------
    solution = problem.getSolution()

    print(solution)
