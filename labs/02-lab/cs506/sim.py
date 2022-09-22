import math
from multiprocessing.sharedctypes import Value
def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x, y):
    # raise NotImplementedError()
    acc = 0
    for i in range(len(x)):
        acc += abs(x[i]-y[i])
    return acc


def jaccard_dist(x, y):
    # i = len(set(x).intersection(y))
    cnt = 0
    for i in range(len(x)):
        if x[i]==y[i]:
            cnt+=1
    
    # n = (len(x)+len(y))-i
    n = len(x)
    if n == 0:
        return ValueError
    return 1-cnt/n

def cosine_sim(x, y):
    normx = 0
    normy = 0
    xdoty = sum([i*j for (i, j) in zip(x, y)])
    for i in range(len(x)):
       normx += (x[i])**2 
    normx = math.sqrt(normx)

    for i in range(len(y)):
        normy += (y[i])**2
    normy = math.sqrt(normy)
    if normx*normy == 0:
        return ValueError
    return  xdoty/normx*normy

# def jaccard_dist(x, y):
    

# def cosine_sim(x, y):
#     raise NotImplementedError()

# Feel free to add more
