import numpy as np

def check_win(T):

    def check_sum(row):
        row = row.tolist()
        if row.count(1) == len(row): return True, 1
        elif row.count(2) == len(row): return True, 2

    # check rows
    for row in T:
        check_sum(row)
    # check columns
    for row in np.transpose(T):
        check_sum(row)
    # check diagonals
    if [T[0, 0], T[1, 1], T[2, 2]].count(1) == 3: return True, 1
    elif [T[0, 2], T[1, 1], T[2, 0]].count(2) == 3: return True, 2
    else: return False, 0

T = np.zeros((3,3))
T[0, 2] = 2
T[1, 1] = 2
T[2, 0] = 2
print(check_win(T))