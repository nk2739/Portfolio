import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from heapq import nlargest

def main():
	matrixL = np.matrix([[1.25,-1.5],[-1.5,5]]) #Question (i)
	vectorsR = []
	for num in range(0,500): #Question (ii)
		sample1 = np.random.randn()
		sample2 = np.random.randn()
		length = np.power(np.power(sample1,2)+np.power(sample2,2),0.5)
		newVector = [[sample1/length],[sample2/length]]
		vectorsR.append(newVector)
	distortedVectors = []
	distortedLengths = []
	xValues = []
	yValues = []
	for val in vectorsR: #Question (iii)
		newR = np.dot(matrixL,val)
		lengthR = np.power(np.power(newR.tolist()[0][0],2)+np.power(newR.tolist()[1][0],2),0.5) #Question (v)
		xValues.append(newR.tolist()[0][0])
		yValues.append(newR.tolist()[1][0])
		distortedVectors.append(newR)
		distortedLengths.append(lengthR)
	eigenvalues = linalg.eigvals(a=matrixL, b=None, overwrite_a=False, check_finite=True) #Question (iv)
	print(eigenvalues)
	eigenvectors = linalg.eig(matrixL) #Question (viii)
	vMax = eigenvectors[1][:,1] 
	product = np.dot(matrixL,vMax)
	print("vmax",vMax)
	plt.hist(distortedLengths,bins=50)
	plt.show() #Quesion (vi)
	plt.plot(xValues,yValues,'bo',product.item(0),product.item(1),'ro')
	plt.axis([-10,10,-10,10])
	plt.show() #Question (ix)

if __name__ == "__main__":
	main()