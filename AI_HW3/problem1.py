'''
Neil Kumar(nk2739)
AI Assignment 3
Programming Problem 1
4/15/18
'''
import pandas
import sys
import copy
import random

# Returns the CSV data of the given file 
def openFileAndReturnData(file):

	inputData = pandas.read_csv(file,header=None)

	return inputData

# Prints out the weights for each iteration of the perceptron algorithm
def printWeightData(outputFile,weightData):

	fileToWrite = open(outputFile,'w')

	for data in weightData:

		# Prints as W1, W2, B
		fileToWrite.write(str(data[1]) + "," + str(data[2]) + "," + str(data[0]) + '\n')

	fileToWrite.close()

# Runs the perceptron algorithm on the given data
def runPerceptronAlg(fileData):

	df = pandas.DataFrame(fileData)

	weightData = []

	# Initializes W0(B), W1, and W2 to random valus b/w 0 and 1
	weights = [random.random(),random.random(),random.random()]
	converged = False

	# Runs until no errors occur during an iteration
	while (converged == False):

		df_copy = copy.copy(df)
		error_occurred = False

		for index,row in df_copy.iterrows():

			feature0 = 1
			feature1 = row[0]
			feature2 = row[1]
			label = row[2]

			# The total weight of a data point 
			total = weights[0]*feature0 + weights[1] * feature1 + weights[2]*feature2

			if total > 0:
				sign = 1
			else:	
				sign = -1

			# Adjusts the perceptrons if an error occurs 
			if label * sign <= 0:
				weights[0] = weights[0] + label * feature0
				weights[1] = weights[1] + label * feature1
				weights[2] = weights[2] + label * feature2
				error_occurred = True
		
		weightData.append(copy.copy(weights))

		if (error_occurred == False):
			converged = True

	return weightData

def main(arg1, arg2):

	inputFile = arg1
	outputFile = arg2

	fileData = openFileAndReturnData(inputFile)

	weightData = runPerceptronAlg(fileData)

	printWeightData(outputFile,weightData)

if __name__ == '__main__':	
	main(sys.argv[1], sys.argv[2])