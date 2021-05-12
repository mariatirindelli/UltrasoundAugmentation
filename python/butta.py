def german2italian_grade(german_grade):
    xa = 1
    xb = 4

    ya = 30
    yb = 18

    italian_grade = ( (german_grade-xa)*yb - (german_grade - xb)*ya ) / (xb - xa)

    return italian_grade

print(german2italian_grade(1.7))
