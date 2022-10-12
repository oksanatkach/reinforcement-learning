import numpy as np
T = np.zeros([3, 3])
all_states = {0 : [T]}
state_lst = []
indices=list(np.ndindex(T.shape))

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

def make_turn(indices, step_lst, player):
    results = []
    for T in step_lst:
        if check_win(T)[0] == False:
            for loc in indices:
                if T[loc] == 0:
                    copy_X = np.array(list(map(list, T)))
                    copy_X[loc] = player
                    results.append(copy_X)
    return results

# collect all possible states at each turn
for i in range(1, 9):
    if i%2 == 1:
        results = make_turn(indices, all_states[i-1], 1)
        all_states[i] = results
        state_lst += results

    else:
        results = make_turn(indices, all_states[i - 1], 2)
        all_states[i] = results
        state_lst += results

# calculate what is the prob that choosing a particular action (cell) at a particular state will lead to a win
# matrix states X actions for each computer's turn
# 0 if the cell is filled
# 1 if playing a cell leads to a win
# 0.5 otherwise
all_weights = {}
for num in range(1, 10, 2):
    current_states = all_states[num-1]
    weights = np.zeros((len(current_states), len(indices)))

    for i, j in np.ndindex(weights.shape):
        state = current_states[i]
        loc = indices[j]
        if state[loc] == 0:
            copy_state = np.array(list(map(list, state)))
            copy_state[loc] = 1
            check, player = check_win(copy_state)
            if check == True and player == 1:
                weights[i, j] = 1
            else:
                weights[i, j] = 0.5

    all_weights[num] = weights

# choose the action that maximizes win
def next_action(current_state, turn, all_weights, all_states, indices):
    turn_matrix = all_weights[turn]
    turn_states = all_states[turn-1]
    for ind in range(len(turn_states)):
        if np.array_equal(current_state, turn_states[ind]):
            row = turn_matrix[ind].tolist()
            argmax = max(row)
            j = row.index(argmax)
            loc = indices[j]
            return loc

# initialize game
state = T
win = False
turn = 1
print('start:')

while win == False:
    # computer's turn
    action = next_action(state, turn, all_weights, all_states, indices)
    state[action] = 1
    print('computer\'s turn:')
    print(state)
    turn += 1
    win, player = check_win(state)
    if win:
        print("Computer won")
        break
    elif not win and 0 not in state:
        print("It's a tie")
        break

    # human turn
    row = int(input())
    col = int(input())

    while state[row, col] != 0:
        print('Choose an empty cell')
        row = int(input())
        col = int(input())
    else:
        state[row, col] = 2
        print('your turn:')
        print(state)
        turn += 1
        win, player = check_win(state)
print("You won")

# reinforcement:
# instead of print return reward 1 for win and -1 for fail, 0 for a tie
# one game is one episode
# matrices are weights that need to be updated according to the rewards