import numpy as np
import random

# 0 is empty, 1 is X, 2 is O

T = np.zeros([3, 3])
all_states = []
# step 0
all_states.append(T)
# step 1
indices=list(np.ndindex(T.shape))
# for loc in indices:
#     copy_X = np.array(list(map(list, T)))
#     copy_O = np.array(list(map(list, T)))
#     copy_X[loc] = 1
#     all_states.append(copy_X)
#     copy_O[loc] = 2
#     all_states.append(copy_O)
#
# print(len(all_states))

def loop_till_win(indices, all_states):
    for loc in indices:
        temp = []

        for T in all_states:
            for i in range(len(T)):
                if sum(T[i]) == 3: continue
                elif sum(np.transpose(T)[i]) == 3: continue
            if T[0, 0] + T[1, 1] + T[2, 2] == 3: continue
            elif T[0, 2] + T[1, 1] + T[2, 0] == 3: continue

            copy_X = np.array(list(map(list, T)))
            copy_O = np.array(list(map(list, T)))
            copy_X[loc] = 1
            temp.append(copy_X)
            copy_O[loc] = 2
            temp.append(copy_O)

            all_states += temp

    return loop_till_win(indices, all_states)

arrays = loop_till_win(indices, all_states)