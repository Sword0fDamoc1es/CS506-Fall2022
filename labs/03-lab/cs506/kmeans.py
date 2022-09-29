from collections import defaultdict
from math import inf
import random
import csv

from scipy import rand


def get_centroid(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    p = len(points) # the number of the points
    d = len(points[0]) # the dimensions of the points
    
    center = [0] * (d)

    for i in range(p):
        for j in range(d):
            center[j] += points[i][j]
    
    for n in range(len(center)):
        center[n] = center[n] / 2
    
    return tuple(center)
    #raise NotImplementedError()


def get_centroids(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the centroid for each of the assigned groups.
    Return `k` centroids in a list
    """
    data_len = len(dataset)
    dd = len(dataset[0])
    ass_len = len(assignments)
    ass_holder = [[]*ass_len]

    SMALL = inf
    SMALL_idx = -1

    for i in range(data_len):
        data = dataset[i]

        for j in range(ass_len):
            dis = distance(assignments[i],data)
            if dis<SMALL:
                SMALL_idx=j
                SMALL=dis
        ass_holder[SMALL_idx].append(data)
        SMALL_idx=-1
        SMALL = inf
    res = []
    for i in range(ass_len):
        cluster = ass_holder[i]
        cur=get_centroid(cluster)
        res.append(cur)
    return res
    
    


    

def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    acc = 0
    d = len(a)
    for i in range(d):
        dif = abs(a[i]-b[i])
        acc += dif**2
    return acc**0.5


def distance_squared(a, b):
    return distance(a,b)**2

def cost_function(clustering):
    cost = 0
    for key in clustering:
        for value in clustering[key]:
            cost += distance_squared(list(key),value)
    return cost


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    set= random.sample(dataset,k)
    return set


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    raise NotImplementedError()


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = get_centroids(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
