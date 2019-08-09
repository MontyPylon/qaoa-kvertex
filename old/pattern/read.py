import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def read_data():
    for gi in range(1, 995):
        path = 'data/' + str(gi) + '.angles'
        data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(e)
            if data:
                print('\n' + str(gi))
                for i in range(len(data[0])):
                    print(data[0][i])
                    for j in range(len(data[1][i])):
                        print('\t' + str(data[1][i][j]))
                print('\n')

if __name__ == '__main__':
    read_data()
