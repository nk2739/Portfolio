import sys
import numpy as np
from numpy import linalg as LA
import scipy
import matplotlib
from matplotlib import pyplot as plt
import scipy.io as sio
import math
#from sklearn.preprocessing import normalize

## Main entry point
def main(filename):

	X, Y = loadData(filename)	
	print(X.shape)
	print(Y.shape)
	# print(X[5])
	# print(createTagDictionary(Y))
	#findProbability(X,Y)
	# print(findTagValues(X,Y))
	doClassification(X,Y,1)

tags = [0,1,2,3,4,5,6,7,8,9]

def createTagDictionary(data):
	tagDictionary = {}

	for n in range(0,len(data)):
		tagDictionary.setdefault(data[n][0],[]).append(n)

	return tagDictionary


def findTagValues(imageData,tagData):
	tagDictionary = createTagDictionary(tagData)
	tagValueDict = {}

	#For each tag in tags, find its mu and variance
	for tag in tags:
		tagIndices = tagDictionary[tag]

		uML = np.zeros(784) #imageData[tagIndices[0],:]
		# print(uML)
		n = len(tagIndices)
		#Only look at values in that tag's list 
		for num in range(0,n):
			temp = imageData[tagIndices[num],:]
			uML = np.add(uML,temp)
		# print(uML)
		uML = uML / n
		print(uML)

		# sumML = np.dot(np.add(imageData[tagIndices[0],:],-1 * uML),np.add(imageData[tagIndices[0],:],-1*uML).transpose())
		sumML = np.zeros(784,784)
		# print(sumML)

		for num in range(0,n):
			diff = np.add(imageData[tagIndices[num],:], -1 * uML)
			diffTranspose = diff.transpose()
			sumML = np.add(sumML,np.dot(diff,diffTranspose))
		sumML = sumML / n
		print("sum:" + str(sumML))
		#Populate value dictionary here
		tagValueDict[tag] = (uML,sumML)
	# print(tagValueDict)
	return tagValueDict

def doClassification(imageData,tagData, image):

	tagValueDict = findTagValues(imageData,tagData)
	finalPredictions = {}

	for tag in tags:
		curMu = tagValueDict[tag][0]
		curVariance = tagValueDict[tag][1]
		# print('mu:' + str(curMu))
		coef = 1 / math.sqrt(math.pow((2 * math.pi),2) * LA.det(curVariance))
		probability = math.exp( -0.5 * np.dot(np.dot((image-curMu).transpose(),LA.inv(sumML))),image-curMu)
		#test with built-in multivariate gaussian 
		finalPredictions[tag] = probability

	#Find max among tags here
	#Return that tag
	# return finalPredictions
	
## Helper Methods

def loadData(mat_file):
	mat_file = sio.loadmat(mat_file)	
	X = mat_file['X']
	Y = mat_file['Y']
	X.astype(float)
	Y.astype(float)	
	
	return X, Y



if __name__ == '__main__':	
	main(sys.argv[1])