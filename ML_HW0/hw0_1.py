import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from heapq import nlargest

def main():
	data = io.loadmat('hw0data.mat')
	matrix = data['M'].astype(int)
	print(matrix)
	print(matrix.shape) #Question (ii)
	print(matrix[3][4]) #Question (iii)
	fifthColumn = matrix[:,4]
	print(np.mean(fifthColumn)) #Question (iv)
	fourthRow = matrix[3,:]
	plt.hist(fourthRow,bins=[0,10,20,30,40,50,60,70,80,90,100])
	plt.show() #Question (v)
	transpose = matrix.transpose()
	#result = np.dot(transpose,matrix)
	result = np.dot(matrix.T,matrix)
	eigenvalues = linalg.eigvals(a=result, b=None, overwrite_a=False, check_finite=True)
	print(nlargest(3,eigenvalues)) #Question (vi)

if __name__ == "__main__":
	main()