import numpy as np

lists = [
    [3,2,4],
    [1,1,1]
]
ar = np.array(lists)
means = [ sum(row)/len(row) for row in ar ]
elite = sorted(means, reverse=True)[:1]
print(elite)