import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

def matMult(A,x):
    ''' Multiply a 2by2 matrix with a 2by1 vector '''
    return [ A[0][0]*x[0]+A[0][1]*x[1] , A[1][0]*x[0] + A[1][1]*x[1]  ]

def prettyPrint(A):
    ''' Print arrays in a pretty way '''
    import pandas as pd
    pd.set_option("display.precision",8)
    Table = pd.DataFrame(A)
    Table.columns = ['']*Table.shape[1]
    print(Table.to_string(index=False))


