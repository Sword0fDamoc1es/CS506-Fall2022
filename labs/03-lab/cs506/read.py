# import pandas as pd
import csv
# from numpy import genfromtxt
def read_csv(csv_file_path):
    """
        Given a path to a csv file, return a matrix (list of lists)
        in row major.
    """

    # my_data = genfromtxt(csv_file_path, delimiter=',')
    readline = []
    my_data = []
    with open(csv_file_path) as file:
        readline = list(csv.reader(file))
        for row in readline:
            buffer = []
            for ele in row:
                if ele.isnumeric():
                    ele = int(ele)
                buffer.append(ele)
            my_data.append(buffer)
    

    return my_data

