'''
Neil Kumar(nk2739)
AI Assignment 3
Programming Problem 2
4/15/18
'''
import pandas
import sys
import copy
import numpy as np

# Opens the given CSV file and returns the data
def openFileAndReturnData(file):

	inputData = pandas.read_csv(file,header=None)

	return inputData

# Prints the requested data [alpha, iterations, etc.]
def printRegressionData(outputFile,regressionData):

	fileToWrite = open(outputFile,'w')

	for data in regressionData:
		fileToWrite.write(str(data[0]) + "," + str(data[1]) + "," + str(round(data[2][0],5)) + "," + str(round(data[2][1],5)) + "," + str(round(data[2][2],5)) + '\n')

	fileToWrite.close()

# Calculates the mean of the requested column of data 
def calcMean(data,column):

	vals = []

	for index,row in data.iterrows():
		val = float(row[column])
		vals.append(val)

	return np.mean(vals)

# Calculates the standard deviation of the requested column of data
def calcStdDev(data,column):
	
	vals = []

	for index,row in data.iterrows():
		val = float(row[column])
		vals.append(val)

	return np.std(vals)

# Calculates the total 'loss' for a column 
def calcTotalLoss(data,column,ageData,weightData,weights):

	n = data.shape[0]
	totalLoss = 0.0
	
	for index,row in data.iterrows():

		# Scale the age and weight values to means of 0
		scaledAge = (row[1] - ageData[0])/ageData[1]
		scaledWeight = (row[2] - weightData[0])/weightData[1]

		val = weights[0] + weights[1]*scaledAge + weights[2]*scaledWeight

		labelVal = row[3]

		if column == 0:
			curX = row[0]

		elif column == 1:
			curX = scaledAge

		else:
			curX = scaledWeight

		curLoss = (val - labelVal)*curX

		# Total loss aggregates each point's loss 
		totalLoss += curLoss
		  
	# Return the average loss for all points 
	return totalLoss / float(n)

# Returns the final weights computed by linear regression
def doLinearRegression(fileData,alpha,numIterations):

	df = pandas.DataFrame(fileData)

	# Weights B0 (Bias), B_Age, B_Weight
	weights = [0.0,0.0,0.0] 

	ageMean = calcMean(df,0)
	ageStdDev = calcStdDev(df,0)

	ageData = (ageMean,ageStdDev)

	weightMean = calcMean(df,1)
	weightStdDev = calcStdDev(df,1)

	weightData = (weightMean,weightStdDev)

	# Adds a column of 1's to the beginning for the intercept
	df.insert(0,-1,1.0)
	for i in range(0,len(df.columns.values)):
		df.columns.values[i-1] = df.columns.values[i-1] + 1

	# Runs for the given number of iterations 
	for i in range(0,numIterations):
		weights_copy = copy.copy(weights)

		# Updates the weights simultaneously 
		weights[0] = weights[0] - alpha * calcTotalLoss(df,0,ageData,weightData,weights_copy)
		weights[1] = weights[1] - alpha * calcTotalLoss(df,1,ageData,weightData,weights_copy)
		weights[2] = weights[2] - alpha * calcTotalLoss(df,2,ageData,weightData,weights_copy)

	regressionData = (alpha,numIterations,weights)
	return regressionData

def main(arg1, arg2):

	inputFile = arg1
	outputFile = arg2

	fileData = openFileAndReturnData(inputFile)

	alphaList = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]
	numIterations = 100

	regressionData = []

	# Runs linear regression for the requested alpha/number of iterations
	for alpha in alphaList:
		data_copy = copy.copy(fileData)

		linearData = doLinearRegression(data_copy,alpha,numIterations)
		regressionData.append(linearData)

	data_copy = copy.copy(fileData)

	# Then runs it on values of alpha/number of iterations I think work well 
	linearData = doLinearRegression(data_copy,1.1,150)
	regressionData.append(linearData)

	printRegressionData(outputFile,regressionData)

if __name__ == '__main__':	
	main(sys.argv[1], sys.argv[2])