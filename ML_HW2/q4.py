############################
#                          #
# authors:                 #
# Zixi Huang(zh2313)       # 
# Neil Kumar(nk2739)       # 
# Yichen Pan(yp2450)       #
#                          #
############################

#Necessary imports 
import sys
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
import scipy.io as sio
import math
from collections import defaultdict as dd

tags = [0,1,2,3,4,5,6,7,8,9]

'''
The main method to test the perceptron algorithms. 
@param filename - the hw1data.mat file 
'''
def main(filename):

	X, Y = loadData(filename)

	for i in range(1,10):
		print(str(i*10) + "%")
		testKernelPerceptron(X,Y,i*0.1)

'''
Helper method to convert a current tag to its encoded version.
Ex: 9 --> [0,0,0,0,0,0,0,0,0,1]
@param curTag - the tag being looked at 
@return tag_encoded - the array like above
'''
def encodeTag(curTag):
	tag_encoded = []
	for tag in tags:
		if tag == curTag:
			tag_encoded.append(1)
		else:
			tag_encoded.append(0)
	return tag_encoded

'''
Helper method to encode a whole tag list.
@param tagData - the tag list
@return newData - the list where each tag is now a (1,10) array 
'''
def encodeTagData(tagData):
	newData = []
	for data in tagData:
		newData.append(encodeTag(data))
	return newData

'''
The perceptron V0 algorithm.
@param imageData- the training data
@param tagaData - the tag data 
@return curW - the resulting weight vector of dim (10,784) to test against 
'''
def perceptronV0(imageData,tagData):
 	w_0 = np.zeros((10,784),dtype=float)
	T = len(imageData)*10
	n = len(imageData)
	prevW = w_0
	

	for t in range(1,T+1):
		i = (t % n) + 1
		
		#The tag data encoded as a list of 1's and 0's to account for all classes 
		y_i = np.asmatrix(encodeTag(tagData[i-1])) 
	
		#The imageData of the current index
		x_i = np.asmatrix(imageData[i-1]) 
		
		if y_i * (np.dot(prevW,x_i.T)) <= 0:
			curW = prevW + (y_i.T * x_i)

		prevW = curW
	
	return curW

'''
The perceptron V1 algorithm. 
@param imageData - the training data
@param tagData - the tag data
@return curW - the resulting weight vector 
'''
def perceptronV1(imageData,tagData):
	w_0 = np.zeros((10,784),dtype=float)
	T = len(imageData)*10
	n = len(imageData)
	prevW = w_0
	
	for t in range(1,T+1):
		
		#The image and tag data as numpy matrices,respectively 
		imageMatrix = np.asmatrix(imageData) 
		tagMatrix = tagData[:len(imageData)]
		encodedTags = np.asmatrix(encodeTagData(tagMatrix))
		
		#Matrix multiplication of all data to speed up computation
		product = np.multiply(encodedTags, np.dot(prevW, imageMatrix.T).T)
		
		#From the (len(imageData), 10) product choose the i'th index that classifies
		#poorly over all classes 

		#Find the i'th index by getting each row's sum and finding the min sum
		sums = np.sum(product, axis=1).tolist()
		minVal = min(sums)
		i = sums.index(minVal)

		y_i = np.asmatrix(encodeTag(tagData[i])) 
		
		x_i = np.asmatrix(imageData[i]) 
		
		if y_i * (np.dot(prevW,x_i.T)) <= 0: 
			curW = prevW + (y_i.T * x_i)
		else:
			return curW
		prevW = curW
	
	return curW

'''
The preceptron V2 algorithm.
@param imageData - the training data.
@param tagData - tbe tag data.
@return c_k, w_k, k - three parameters to classify the test data
'''
def perceptronV2(imageData,tagData):
	w_k = dd(lambda: np.zeros((10,784),dtype=float))
	T = len(imageData)*10
	n = len(imageData)
	c_k = dd(lambda: 0)
	k = 1

	for t in range(1,T+1):
		i = (t % n) + 1
		y_i = np.asmatrix(encodeTag(tagData[i-1])) 
		x_i = np.asmatrix(imageData[i-1]) 
		
		#Now update the hyperparameters instead of one weight vector 
		curW = w_k[k]
		if y_i * (np.dot(curW,x_i.T)) <= 0: 
			w_k[k+1] = curW + (y_i.T * x_i)
			c_k[k+1] = 1
			k = k + 1
		else:
			c_k[k] = c_k[k] + 1
	
	return c_k,w_k,k 

'''
The kernel perceptron algorithm: https://en.wikipedia.org/wiki/Polynomial_kernel.
@param imageData - the training data.
@param tagData0 - the tag data.
@return w - the weight vector after transforming the data. 
'''
def kernelPerceptron(imageData,tagData):
	c = 0 #C = 0 so in this case homogenous 
	d = 10 #d-degree kernel

	w = np.zeros((10,784),dtype=float)

	for i in range(0,len(imageData)):
		x_i = np.asmatrix(imageData[i]).T 
		y_i = np.asmatrix(encodeTag(tagData[i]))

		w += np.power(np.dot(x_i,y_i).T + c, d)

	return w

'''
A method to test the perceptron V0 algorithm.
@param imageData - the whole set of images
@param tagData - all the tags
@param trainPercentage - the percentage of training data
'''
def testPerceptronV0(imageData,tagData, trainPercentage):

	trainLength = trainPercentage * len(imageData)
	
	trainData = imageData[:trainLength]
	testData = imageData[trainLength:len(imageData)]
	
	correctTags = tagData[trainLength:len(tagData)]
	
	numCorrect = 0
	
	#The perceptron from the V0 algorithm 
	weightVect = perceptronV0(trainData,tagData)
	
	for i in range(0,len(testData)):
		correctTag = correctTags[i][0]
		
		#For each image, do the dot product with the weight vector and choose the max index
		val = np.dot(np.asmatrix(weightVect),testData[i].T)
		valAsList = val.tolist()[0]
		maxVal = max(valAsList)
		
		#The max index correlates to the best predicted tag 
		max_index = valAsList.index(maxVal) 
		
		if (max_index == correctTag):
			numCorrect += 1

	print(numCorrect)
	print(float(numCorrect / float(len(testData))))

'''
A method to test the perceptron V1 algorithm. 
@param imageData - the whole set of images
@param tagData - all the tags
@param trainPercentage - the percentage of training data
'''
def testPerceptronV1(imageData,tagData, trainPercentage):

	trainLength = trainPercentage * len(imageData)
	
	trainData = imageData[:trainLength]
	testData = imageData[trainLength:len(imageData)]
	
	correctTags = tagData[trainLength:len(tagData)]

	numCorrect = 0
	
	#The perceptron from the V1 algorithm 
	weightVect = perceptronV1(trainData,tagData)
	
	for i in range(0,len(testData)):
		correctTag = correctTags[i][0]
		
		#For each image find the dot product with the weight vector and choose the max index 
		val = np.dot(np.asmatrix(weightVect),testData[i].T)
		valAsList = val.tolist()[0]
		maxVal = max(valAsList)
		
		max_index = valAsList.index(maxVal) 
		
		if (max_index == correctTag):
			numCorrect += 1

	print(numCorrect)
	print(float(numCorrect / float(len(testData))))

'''
Tests the perceptronV2 algorithm for different splits.
@param imageData - the whole set of images
@param tagData - all the tags
@param trainPercentage - the percentage of training data
'''
def testPerceptronV2(imageData,tagData, trainPercentage):

	trainLength = trainPercentage * len(imageData)
	
	trainData = imageData[:trainLength]
	testData = imageData[trainLength:len(imageData)]
	
	correctTags = tagData[trainLength:len(tagData)]
	
	numCorrect = 0
	
	#Here the perceptron V2 algorithm returns 3 hyperparameters 
	weightVect = perceptronV2(trainData,tagData)
	c_k, w_k, k = weightVect[0], weightVect[1], weightVect[2]

	
	for i in range(0,len(testData)):
		correctTag = correctTags[i][0]
		
		#For every image, find the sum from the algorithm  
		total = np.zeros((10,1),dtype=float)
		for j in range(1,k+1):

			total += c_k[j] * np.dot(np.asmatrix(w_k[j]), np.asmatrix(testData[i]).T)
		
		#Then choose the index that contains the max dot product 
		valAsList = total.T.tolist()[0]
		maxVal = max(valAsList)
		
		max_index = valAsList.index(maxVal) 

		if (max_index == correctTag):
			numCorrect += 1

	print(numCorrect)
	print(float(numCorrect / float(len(testData))))

'''
Tests the kernel perceptron algorithm on different splits of the data.
@param imageData - the whole set of images
@param tagData - all the tags
@param trainPercentage - the percentage of training data
'''
def testKernelPerceptron(imageData,tagData,trainPercentage):
	trainLength = trainPercentage * len(imageData)
	
	trainData = imageData[:trainLength]
	
	testData = imageData[trainLength:len(imageData)]
	
	#Tags associated with the test data
	correctTags = tagData[trainLength:len(tagData)]
	
	numCorrect = 0
	
	#The perceptron trained on the training data 
	weightVect = kernelPerceptron(trainData,tagData)
	
	for i in range(0,len(testData)):
		correctTag = correctTags[i][0]
		
		#For each image find the dot product with the weight vector and choose the 
		# tag with the highest value
		val = np.dot(np.asmatrix(weightVect),testData[i].T)
		valAsList = val.tolist()[0]
		maxVal = max(valAsList)
		
		max_index = valAsList.index(maxVal) 
		
		if (max_index == correctTag):
			numCorrect += 1

	#Prints the number of correct guesses and the % correct
	print(numCorrect)
	print(float(numCorrect / float(len(testData))))

'''
Helper method to load the data file
@param mat_file - hw1data.mat
'''
def loadData(mat_file):
	mat_file = sio.loadmat(mat_file)	
	X = mat_file['X']
	Y = mat_file['Y']
	X.astype(float)
	Y.astype(float)	
	
	return X, Y

if __name__ == '__main__':	
	main(sys.argv[1])