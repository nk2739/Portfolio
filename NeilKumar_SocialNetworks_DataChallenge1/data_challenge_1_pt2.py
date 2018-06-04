#Neil Kumar(nk2739)
#Data Challenge 1 (Part 2)
#4/8/18

import networkx as nx
from random import *
from operator import itemgetter
import math
import collections
import random
import requests
import json 
import csv

'''
Gets a node's information for a specific node ID.
'''
def getSpecificNode(request,uni,key,nodeID):
	
	request = request + nodeID + '/NodeInfo'

	headers = {"uni": uni, "key": key}

	node = requests.get(request,headers=headers)

	json_data = json.loads(node.text)

	return json_data

'''
Returns a random node's information.
'''
def getRandomNode(request,uni,key):
	headers = {"uni": uni, "key": key}

	node = requests.get(request,headers=headers)

	json_data = json.loads(node.text)

	return json_data

'''
Requests the API key given my uni and password.
''' 
def getAPIKey(request,uni,password):

	headers = {"uni": uni, "pass": password}

	key = requests.get(request,headers=headers)

	json_data = json.loads(key.text)

	return json_data

'''
Calculates a node's "value" based on a heuristic.
Value = Degree * (1 + clusteringCoef)
'''
def calculateNodeValue(nodeVals):

	degree = nodeVals[1]
	clusteringCoef = nodeVals[3]

	if (is_float(clusteringCoef)):
		value = float(degree * float(1+float(clusteringCoef)))

	else:
		value = float(degree)

	return value 

'''
Determines whether a string can be cast to a float or not.
'''
def is_float(val):

	try:
		num = float(val)

	except ValueError:
		return False
	return True

'''
Calculates the total value of all neighbors of a node.
Value (Each Node) = Degree * (1+clusteringCoef) * labelValue (A = 1, B = 0.3)
Total = Sum of all neighbor values
'''
def calcNeighborValue(neighbors):

	totalVal = 0.0

	for neighbor in neighbors:

		label = neighbor[2]

		if label == 'B':
			coef = 0.3
		else:
			coef = 1

		adjustedVal = calculateNodeValue(neighbor)*coef

		totalVal += adjustedVal

	return totalVal

'''
Returns data for a node and its neighbors to write to the csv file. 
'''
def createNodeData(extractInfo):

	nodeData = [] 
	mainNode = extractInfo[0]

	neighborNodes = extractInfo[1]

	mainNodeData = (mainNode[0],mainNode[1],mainNode[2],mainNode[3],calculateNodeValue(mainNode),calcNeighborValue(neighborNodes))

	nodeData.append(mainNodeData)

	for neighborNode in neighborNodes:
		neighborData = (neighborNode[0],neighborNode[1],neighborNode[2],neighborNode[3],calculateNodeValue(neighborNode),0.0)
		nodeData.append(neighborData)

	return nodeData

'''
Writes the current node and its neighbors' data into the appropriate csv file.
'''
def writeToFile(nodeData):

	curFile = 'data_challenge_data.csv'

	file = open(curFile,'a')

	for data in nodeData:
		file.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "\n")

	file.close()

'''
Returns all the node information in the data csv and returns the node in order.
Order determined by a node's curNodeVal + neighborsVal. 
'''
def getNodesInOrder():

	nodes = []

	curFile = 'data_challenge_data.csv'

	file = open(curFile,'r')

	reader = csv.DictReader(file,delimiter=',')

	for row in reader:
		
		nodeID = row['NodeID']

		total = float(row['CurNodeVal']) + float(row['NeighborsVal'])

		node = (nodeID,total)

		nodes.append(node)

	sortedNodes = sorted(nodes,key=itemgetter(1),reverse=True)

	return sortedNodes

'''
Finds the next biggest node to explore in the data csv file.
A node has not been seen if its 'neighborsVal' is still 0.
'''
def findNextBiggestNode():

	curFile = 'data_challenge_data.csv'

	file = open(curFile,'r')

	reader = csv.DictReader(file,delimiter=',')
	seen = set()

	for row in reader:
		if (float(row['NeighborsVal'])!= 0.0):
			seen.add(row['NodeID'])

	file.close()

	file = open(curFile,'r')
	reader = csv.DictReader(file,delimiter=',')

	maxVal = -1*float("inf")
	maxNode = ''
	vals = []

	for row in reader:
		if row['NodeID'] not in seen:

			if float(row['CurNodeVal']) > maxVal:
				maxVal = float(row['CurNodeVal'])
				maxNode = row['NodeID']

	file.close()
	return maxNode

'''
Gets random nodes and writes their information into the data csv file. 
'''
def getAndWriteRandomCalls(url,uni,key,center,numCalls):

	for i in range(0,numCalls):

		randomNode = getRandomNode(url,uni,key)

		info = extractInfo(randomNode,center)

		nodeData = createNodeData(info)

		writeToFile(nodeData)

'''
Returns the number of calls left to the 'api/nodes' access point. 
'''
def getNumberOfCallsLeft(request,uni,key):

	headers = {"uni": uni, "key": key}

	key = requests.get(request,headers=headers)

	json_data = json.loads(key.text)

	return json_data

'''
Gets the next biggest 'node' and writes the info to the data csv file. 
Does this for a certain number of calls. 
'''
def getAndWriteSpecificCalls(url,uni,key,center,numCalls):

	for i in range(0,numCalls):

		nextBiggestNode = findNextBiggestNode()

		specificNode = getSpecificNode(url,uni,key,str(nextBiggestNode))

		info = extractInfo(specificNode,center)

		nodeData = createNodeData(info)

		writeToFile(nodeData)

'''
Gets a specific node's information and writes it to the data csv file. 
'''
def getAndWriteSpecificCall(url,uni,key,center,nodeid):

	specificNode = getSpecificNode(url,uni,key,str(nodeid))

	info = extractInfo(specificNode,center)

	nodeData = createNodeData(info)

	writeToFile(nodeData)

'''
Gets the top 250 distinct nodes from all nodes.
'''
def getTopNodes(sortedNodes):

	topNodes = set()
	seen = set()
	count = 0

	for node in sortedNodes:

		if count < 250 and node[0] not in seen:
			seen.add(node[0])
			count += 1
			topNodes.add(node[0])

	return topNodes

'''
Prints the top 250 nodes to the seed set file. 
'''
def printTopNodes(topNodes):

	file = open("seedset.txt","w")

	for node in topNodes:
		file.write(node + '\n')

	file.close()

'''
Gets all the relevant information for a node and its neighbors from the returned JSON. 
'''
def extractInfo(nodeJSON,center):

	curNodeID = nodeJSON['nodeid']
	curDegree = nodeJSON['degree']
	curLabel = nodeJSON['label']

	curClusteringCoef = nodeJSON['clusteringCoef']

	curNode = (curNodeID,curDegree,curLabel,curClusteringCoef)

	neighbors = nodeJSON['neighbors']
	neighborVals = []
	
	for neighbor in neighbors:

		neighborID = nodeJSON['neighbors'][neighbor]['nodeid']
		neighborDegree = nodeJSON['neighbors'][neighbor]['degree']
		neighborLabel = nodeJSON['neighbors'][neighbor]['label']

		neighborClusteringCoef = nodeJSON['neighbors'][neighbor]['clusteringCoef']

		neighborVal = (neighborID,neighborDegree,neighborLabel,neighborClusteringCoef)

		neighborVals.append(neighborVal)

	info = (curNode,neighborVals)

	return info

def main():
	
	SF = (37.773972,-122.431297)

	IP = '167.99.225.109'
	port = '5000'
	uni = 'nk2739'
	apiKey = '93731'
	password = '22'

	keyUrl = 'http://' + IP + ':' + port + '/api/getKey' 
	randomNodeURL = 'http://' + IP + ':' + port + '/api/nodes/getRandomNode' 
	specificNodeURL = 'http://' + IP + ':' + port + '/api/nodes/' 
	callsURL = 'http://' + IP + ':' + port + '/api/getCalls' 

	# Get the API key initially to query the server. 
	apiKey = getAPIKey(keyUrl,uni,password)	

	# Keeps track of how many calls I have left to the server. 
	numCallsLeft = getNumberOfCallsLeft(callsURL,uni,apiKey)

	# Initially populate the data file by getting information for the seed nodes. 
	seedNodes = ['12080','266','3762','4059','4109','53419','6228','77253','53419']

	for seed in seedNodes:

		getAndWriteSpecificCall(specificNodeURL,uni,apiKey,SF,seed)

	# Alternate between gaining information from random nodes or the best nodes I have already seen. 
	getAndWriteRandomCalls(randomNodeURL,uni,apiKey,SF,25)
	# getAndWriteSpecificCalls(specificNodeURL,uni,apiKey,SF,25)

	# Finally get the top 250 nodes and print them to the seed set. 
	sortedNodes = getNodesInOrder()
	topNodes = getTopNodes(sortedNodes)
	printTopNodes(topNodes)

if __name__ == '__main__':	
	main()