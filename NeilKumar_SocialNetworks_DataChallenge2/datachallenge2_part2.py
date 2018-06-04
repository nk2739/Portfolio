#Neil Kumar(nk2739)
#Social Networks 
#Data Challenge 2 Part 2
#4/26/18

import networkx as nx
from random import *
from operator import itemgetter
import random
import math

# Creates the graph of the blog data 
def createNewGraph():
	graph = nx.read_gml('polblogs.gml',label="id")
	return graph

# Selects 20% of the edges in the blog graph randomly and prints them to a file.
def pruneAndPrint(predictionNodeFile,correctNodeFile,graph):

	graph_edges = list(graph.edges)
	numToDelete = int(0.2*len(graph_edges))

	predictionNodeFile = open(predictionNodeFile,'w')
	correctNodeFile = open(correctNodeFile,'w')

	# Selects 20% of the edges using random sampling 
	edgesToDelete = set(random.sample(range(0,len(graph_edges)),numToDelete))
	
	for edge in edgesToDelete:
		removed_edge = graph_edges[edge]
		fromNode = removed_edge[0]
		toNode = removed_edge[1]
		value = removed_edge[2]
		predictionNodeFile.write(str(fromNode) + "\n")
		correctNodeFile.write(str(fromNode) + "," + str(toNode) + "," + str(value) + "\n")

	predictionNodeFile.close()
	correctNodeFile.close()

# Gets a pruned blog graph by removing the edges printed in 'edgeFile'
def getPrunedGraph(edgeFile,graph):
	
	input = open(edgeFile,'r')
	lines = input.readlines()

	for line in lines:
		parts = line.strip().replace('\t',",").split(",")

		fromNode = int(parts[0])
		toNode = int(parts[1])
		value = int(parts[2])

		graph.remove_edge(*(fromNode,toNode,value))

	return graph

# Selects a portion of the Instagram graph to delete and prints those edges to a file.
def pruneIGAndPrint(predictionNodeFile,correctNodeFile,graph):

	graph_edges = list(graph.edges)
	numToDelete = int(0.03*len(graph_edges))

	predictionNodeFile = open(predictionNodeFile,'w')
	correctNodeFile = open(correctNodeFile,'w')

	edgesToDelete = set(random.sample(range(0,len(graph_edges)),numToDelete))
	
	for edge in edgesToDelete:
		removed_edge = graph_edges[edge]
		fromNode = removed_edge[0]
		toNode = removed_edge[1]
		predictionNodeFile.write(str(fromNode) + "\n")
		correctNodeFile.write(str(fromNode) + "," + str(toNode) + "\n")

	predictionNodeFile.close()
	correctNodeFile.close()

# Returns a graph of the pruned Instagram data based on the edges in 'edgeFile'.
def getPrunedIGGraph(edgeFile,graph):
	
	input = open(edgeFile,'r')
	lines = input.readlines()

	for line in lines:
		parts = line.strip().replace('\t',",").split(",")

		fromNode = int(parts[0])
		toNode = int(parts[1])

		graph.remove_edge(*(fromNode,toNode))

	return graph

# Creates and returns the Instagram graph (from CSV data).
def createGraph():
	
	input = open("graph_edges.csv",'r')
	lines = input.readlines()
	G = nx.Graph()

	for line in lines:
		parts = line.strip().replace('\t',",").split(",")
		
		if parts[0].isdigit() and parts[1].isdigit():

			fromNode = int(parts[0])
			toNode = int(parts[1])

			G.add_edge(fromNode,toNode)

	return G

# Returns the 'graph_gender.csv' data as a dictionary of [node]:gender
def getGenderData(genderFile):

	genderFile = open(genderFile,'r')
	lines = genderFile.readlines()
	genderDict = {}

	for line in lines:
		parts = line.strip().replace('\t',",").split(",")

		node = int(parts[0])
		gender = str(parts[1])
		genderDict[node] = gender

	return genderDict

# Returns the number of recommendations of nodes as a dict - [node]:# times recommended
def getRWealthValues(predictionFile):

	input = open(predictionFile,'r')
	lines = input.readlines()
	rWealthDict = {}

	for line in lines:
		parts = line.strip().replace('\t',",").split(",")
		
		if parts[0].isdigit() and parts[1].isdigit():
		
			node = int(parts[0])
			prediction = int(parts[1])

			if prediction not in rWealthDict:
				rWealthDict[prediction] = 1
			else:
				rWealthDict[prediction] += 1

	return rWealthDict

# Returns a list of all node's data as (node,wealth,r_wealth)
def getNodeWealthData(predictionFile,graph):

	nodeWealthData = []
	r_wealthValues = getRWealthValues(predictionFile)

	for node in list(graph.nodes):

		wealth = len(set(graph.neighbors(node)))

		if (node in r_wealthValues):
			r_wealth = r_wealthValues[node]
		else:
			r_wealth = 0

		# Does not include nodes that have a wealth or r_wealth of 0 
		if wealth > 0 and r_wealth > 0:
			nodeData = (node,wealth,r_wealth)
			nodeWealthData.append(nodeData)

	return nodeWealthData

# Calculates the F score for the data given the % of data being looked at
# Returns the % female among the top % of this data 
def getFValue(nodeWealthData,genderData,is_prime,n):

	# Sorts based on wealth if not prime 
	if (is_prime == False):
		top_nodes = sorted(nodeWealthData,key=lambda x: x[1],reverse=True)[:n]
		
	# Sorts based on r_wealth if prime 
	else:
		top_nodes = sorted(nodeWealthData,key=lambda x: x[2],reverse=True)[:n]
		
	num_female = 0
	for node in top_nodes:
		if (node[0] in genderData):
			gender = str(genderData[node[0]]).strip()
			if (gender == 'F'):
				num_female += 1

	return float(float(num_female)/float(len(top_nodes)))

# Returns the 'fairness' score of the algorithm on the Instagram Data.
def getFairnesScore(predictionFile,genderFile,graph):

	genderData = getGenderData(genderFile)
	
	nodeWealthData = getNodeWealthData(predictionFile,graph)

	f_1 = getFValue(nodeWealthData,genderData,False,int(0.01*len(nodeWealthData)))
	f_1_prime = getFValue(nodeWealthData,genderData,True,int(0.01*len(nodeWealthData)))

	f_10 = getFValue(nodeWealthData,genderData,False,int(0.1*len(nodeWealthData)))
	f_10_prime = getFValue(nodeWealthData,genderData,True,int(0.1*len(nodeWealthData)))

	f_25 = getFValue(nodeWealthData,genderData,False,int(0.25*len(nodeWealthData)))
	f_25_prime = getFValue(nodeWealthData,genderData,True,int(0.25*len(nodeWealthData)))
	
	fairness = abs(f_1_prime-f_1) + abs(f_10_prime-f_10) + abs(f_25_prime-f_25)
	return fairness

# Returns tuples of (node,wealth,r_wealth) for the blog data.
def getBlogNodeWealthData(predictionFile,graph):

	nodeWealthData = []
	r_wealthValues = getRWealthValues(predictionFile)

	for node in list(graph.nodes):

		# In-Degree instead of all neighbors 
		wealth = graph.in_degree(node)

		if (node in r_wealthValues):
			r_wealth = r_wealthValues[node]
		else:
			r_wealth = 0

		# Does not include nodes that have a wealth or r_wealth of 0 
		if wealth > 0 and r_wealth > 0:
			nodeData = (node,wealth,r_wealth)
			nodeWealthData.append(nodeData)

	return nodeWealthData

# Calculates the F score for the data given the % of data being looked at.
# Calculates the fairness for blogs that are considered "liberal" (value 0).
def getBlogFValue(nodeWealthData,politicalData,is_prime,n):

	# Sorts based on wealth if not prime 
	if (is_prime == False):
		top_nodes = sorted(nodeWealthData,key=lambda x: x[1],reverse=True)[:n]
		
	# Sorts based on r_wealth if prime 
	else:
		top_nodes = sorted(nodeWealthData,key=lambda x: x[2],reverse=True)[:n]
		
	# Looks at the fairness for 'liberal' blogs.
	num_liberal = 0
	for node in top_nodes:
		if (node[0] in politicalData):
			affiliation = str(politicalData[node[0]]).strip()
			if (affiliation == '0'):
				num_liberal += 1

	return float(float(num_liberal)/float(len(top_nodes)))

# Returns the 'fairness' score for the blog data regarding political affiliation (liberal).
def getBlogFairnesScore(predictionFile,politicalData,graph):
	
	nodeWealthData = getBlogNodeWealthData(predictionFile,graph)

	f_1 = getBlogFValue(nodeWealthData,politicalData,False,int(0.01*len(nodeWealthData)))
	f_1_prime = getBlogFValue(nodeWealthData,politicalData,True,int(0.01*len(nodeWealthData)))

	f_10 = getBlogFValue(nodeWealthData,politicalData,False,int(0.1*len(nodeWealthData)))
	f_10_prime = getBlogFValue(nodeWealthData,politicalData,True,int(0.1*len(nodeWealthData)))

	f_25 = getBlogFValue(nodeWealthData,politicalData,False,int(0.25*len(nodeWealthData)))
	f_25_prime = getBlogFValue(nodeWealthData,politicalData,True,int(0.25*len(nodeWealthData)))
	
	fairness = abs(f_1_prime-f_1) + abs(f_10_prime-f_10) + abs(f_25_prime-f_25)
	return fairness

# Writes a (node,prediction) pair to the given output file 
def writeToFile(file,node,prediction):

	outputFile = open(file,'a')
	outputFile.write(str(node) + "," + str(prediction) + "\n")

# Returns the recommended node for a certain node 
def getPredictedNode(node,graph,alreadyPredicted,predictionDict):

	# Doesn't redo predictions 
	if (node in alreadyPredicted):
		return predictionDict[node]

	# Return random nodes for nodes that don't exist in the graph
	if (graph.has_node(node) == False):
		graphNodes = list(graph.nodes)
		randomNode = graphNodes[random.randint(0,len(graphNodes)-1)]
		bestNode = randomNode

		alreadyPredicted.add(node)
		predictionDict[node] = bestNode

		return bestNode

	curNeighbors = set(graph.neighbors(node))

	degree2Neighbors = set()

	# Create a list of viable nodes that are degree 2 away 
	for neighbor in curNeighbors:
		degree2 = set(graph.neighbors(neighbor))

		for node2 in degree2:
			if (node2 not in degree2Neighbors) and (node2 not in curNeighbors) and (node2 != node):
				degree2Neighbors.add(node2)

	bestNode = ''
	bestValue = float("-inf")

	# Find the best node that is degree 2 away 
	for node2 in degree2Neighbors:
		
		node2Neighbors = set(graph.neighbors(node2))
		intersection = curNeighbors.intersection(node2Neighbors)
		score = 0.0

		for intersect in intersection:
			score += 1 / math.log(len(set(graph.neighbors(intersect))))

		# The recommended node has the highest 'score' 
		if score > bestValue:
			bestValue = score
			bestNode = node2

	# If a node has no neighbors assign it randomly 
	if bestNode == '':
		graphNodes = list(graph.nodes)
		randomNode = graphNodes[random.randint(0,len(graphNodes)-1)]
		bestNode = randomNode

	alreadyPredicted.add(node)
	predictionDict[node] = bestNode

	# Returns the highest scoring node that is degree 2 away 
	return bestNode

# My Algorithm to predict the Instagram data (using common neighbors + Adamic-Adar).
def myAlgGetPredictedNode(node,graph,alreadyPredicted,predictionDict):

	# Doesn't redo predictions 
	if (node in alreadyPredicted):
		return predictionDict[node]

	# Return random nodes for nodes that don't exist in the graph
	if (graph.has_node(node) == False):
		graphNodes = list(graph.nodes)
		randomNode = graphNodes[random.randint(0,len(graphNodes)-1)]
		bestNode = randomNode

		alreadyPredicted.add(node)
		predictionDict[node] = bestNode

		return bestNode

	curNeighbors = set(graph.neighbors(node))

	degree2Neighbors = set()

	# Create a list of viable nodes that are degree 2 away 
	for neighbor in curNeighbors:
		degree2 = set(graph.neighbors(neighbor))

		for node2 in degree2:
			if (node2 not in degree2Neighbors) and (node2 not in curNeighbors) and (node2 != node):
				degree2Neighbors.add(node2)

	bestNode = ''
	bestValue = float("-inf")

	# Find the best node that is degree 2 away 
	for node2 in degree2Neighbors:
		
		node2Neighbors = set(graph.neighbors(node2))
		intersection = curNeighbors.intersection(node2Neighbors)
		score = 0.0

		for intersect in intersection:
			score += 1 / math.log(len(set(graph.neighbors(intersect))))

		numIntersection = len(list(intersection))
		score = score * math.log(numIntersection+1,10)

		if score > bestValue:
			bestValue = score
			bestNode = node2

	# If a node has no neighbors assign it randomly 
	if bestNode == '':
		graphNodes = list(graph.nodes)
		randomNode = graphNodes[random.randint(0,len(graphNodes)-1)]
		bestNode = randomNode

	alreadyPredicted.add(node)
	predictionDict[node] = bestNode

	# Returns the highest scoring node that is degree 2 away 
	return bestNode
		
# Predicts a node for each node in "nodes_predict.csv" and writes the output
def predictNodes(file,outputFile,graph,alreadyPredicted,predictionDict):

	predictionNodes = open(file,'r')
	lines = predictionNodes.readlines()

	for line in lines:
		node = int(line.strip())
		predictedNode = getPredictedNode(node,graph,alreadyPredicted,predictionDict)
		writeToFile(outputFile,node,predictedNode)

# Predicts each node using my algorithm on the Instagram data.
def myAlgPredictNodes(file,outputFile,graph,alreadyPredicted,predictionDict):

	predictionNodes = open(file,'r')
	lines = predictionNodes.readlines()

	for line in lines:
		node = int(line.strip())
		predictedNode = myAlgGetPredictedNode(node,graph,alreadyPredicted,predictionDict)
		writeToFile(outputFile,node,predictedNode)

# Predicts the Blog data using Adamic-Adar.
def getPredictedBlogNode(node,graph,alreadyPredicted,predictionDict):

	# Doesn't redo predictions 
	if (node in alreadyPredicted):
		return predictionDict[node]

	# Return random nodes for nodes that don't exist in the graph
	if (graph.has_node(node) == False):
		graphNodes = list(graph.nodes)
		randomNode = graphNodes[random.randint(0,len(graphNodes)-1)]
		bestNode = randomNode

		alreadyPredicted.add(node)
		predictionDict[node] = bestNode

		return bestNode

	curNeighbors = set(nx.all_neighbors(graph,node))

	degree2Neighbors = set()

	# Create a list of viable nodes that are degree 2 away 
	for neighbor in curNeighbors:
		degree2 = set(nx.all_neighbors(graph,neighbor))

		for node2 in degree2:
			if (node2 not in degree2Neighbors) and (node2 not in curNeighbors) and (node2 != node):
				degree2Neighbors.add(node2)

	bestNode = ''
	bestValue = float("-inf")

	# Find the best node that is degree 2 away 
	for node2 in degree2Neighbors:
		
		node2Neighbors = set(nx.all_neighbors(graph,node2))
		intersection = curNeighbors.intersection(node2Neighbors)
		score = 0.0

		for intersect in intersection:
			score += 1 / math.log(len(set(nx.all_neighbors(graph,intersect))))

		# The recommended node has the highest 'score' 
		if score > bestValue:
			bestValue = score
			bestNode = node2

	# If a node has no neighbors assign it randomly 
	if bestNode == '':
		graphNodes = list(graph.nodes)
		randomNode = graphNodes[random.randint(0,len(graphNodes)-1)]
		bestNode = randomNode

	alreadyPredicted.add(node)
	predictionDict[node] = bestNode

	# Returns the highest scoring node that is degree 2 away 
	return bestNode
		

# Predicts the Blog data using my algorithm (Common Neighbors + Adamic-Adar).
def myAlgGetPredictedBlogNode(node,graph,alreadyPredicted,predictionDict):

	# Doesn't redo predictions 
	if (node in alreadyPredicted):
		return predictionDict[node]

	# Return random nodes for nodes that don't exist in the graph
	if (graph.has_node(node) == False):
		graphNodes = list(graph.nodes)
		randomNode = graphNodes[random.randint(0,len(graphNodes)-1)]
		bestNode = randomNode

		alreadyPredicted.add(node)
		predictionDict[node] = bestNode

		return bestNode

	curNeighbors = set(nx.all_neighbors(graph,node))

	degree2Neighbors = set()

	# Create a list of viable nodes that are degree 2 away 
	for neighbor in curNeighbors:
		degree2 = set(nx.all_neighbors(graph,neighbor))

		for node2 in degree2:
			if (node2 not in degree2Neighbors) and (node2 not in curNeighbors) and (node2 != node):
				degree2Neighbors.add(node2)

	bestNode = ''
	bestValue = float("-inf")

	# Find the best node that is degree 2 away 
	for node2 in degree2Neighbors:
		
		node2Neighbors = set(nx.all_neighbors(graph,node2))
		intersection = curNeighbors.intersection(node2Neighbors)
		score = 0.0

		for intersect in intersection:
			score += 1 / math.log(len(set(nx.all_neighbors(graph,intersect))))

		numIntersection = len(list(intersection))
		score = score * math.log(numIntersection+1,10)

		# The recommended node has the highest 'score' 
		if score > bestValue:
			bestValue = score
			bestNode = node2

	# If a node has no neighbors assign it randomly 
	if bestNode == '':
		graphNodes = list(graph.nodes)
		randomNode = graphNodes[random.randint(0,len(graphNodes)-1)]
		bestNode = randomNode

	alreadyPredicted.add(node)
	predictionDict[node] = bestNode

	# Returns the highest scoring node that is degree 2 away 
	return bestNode

# Predicts the blog data using Adamic-Adar.
def predictBlogNodes(file,outputFile,graph,alreadyPredicted,predictionDict):

	predictionNodes = open(file,'r')
	lines = predictionNodes.readlines()

	for line in lines:
		node = int(line.strip())
		predictedNode = getPredictedBlogNode(node,graph,alreadyPredicted,predictionDict)
		writeToFile(outputFile,node,predictedNode)

# Predicts the blog data using my algorithm. 
def myAlgPredictBlogNodes(file,outputFile,graph,alreadyPredicted,predictionDict):

	predictionNodes = open(file,'r')
	lines = predictionNodes.readlines()

	for line in lines:
		node = int(line.strip())
		predictedNode = myAlgGetPredictedBlogNode(node,graph,alreadyPredicted,predictionDict)
		writeToFile(outputFile,node,predictedNode)

# Calculates the accuracy of a prediction file compared to the correct data.
def calcCorrectness(predictionFile,correctFile):
	
	predictFile = open(predictionFile,'r')
	correctFile = open(correctFile,'r')
	predictData = predictFile.readlines()
	correctData = correctFile.readlines()

	num_lines = len(predictData)
    
	numCorrect = 0

	for i in range(0,num_lines):
		myPrediction = predictData[i]
		correctAnswer = correctData[i]

		predictParts = myPrediction.strip().replace('\t',",").split(",")
		correctParts = correctAnswer.strip().replace('\t',",").split(",")

		fromPredict = predictParts[0]
		toPredict = predictParts[1]

		fromCorrect = correctParts[0]
		toCorrect = correctParts[1]

		if fromPredict == fromCorrect and toPredict == toCorrect:
			numCorrect += 1

	return float(float(numCorrect)/float(num_lines))

def main():

	# The blog data graph
	newGraph = createNewGraph()

	# Dict of [blog]:affiliation (0-liberal, 1-conservative)
	affiliations = nx.get_node_attributes(newGraph,'value')

	# The blog nodes to predict.
	myDataPredictionNodeFile = "blog_nodes_predict.csv"

	# The correct blog edges.
	myDataCorrectNodeFile = "blog_correct_node_data.csv"

	# The Instagram data to predict.
	updatedIGDataPredictionNodeFile = "new_ig_nodes_predict.csv"

	# The correct Instagram edges.
	IGDataCorrectNodeFile = "IG_correct_node_data.csv"

	# My predictions on the blog data using Adamic-Adar.
	AdamicPredictionsOnNewDataFile = 'adamic_blog_predictions.csv'

	# My predictions on the Instagram data using Adamic-Adar.
	AdamicPredictionsOnIGDataFile = 'adamic_ig_predictions.csv'

	# My predictions on the blog data using my algorithm.
	MyAlgPredictionsOnNewDataFile = 'myalg_blog_predictions.csv'

	# My predictions on the Instagram data using my algorithm.
	MyAlgPredictionsOnIGDataFile = 'myalg_ig_predictions.csv'

	# The Instagram graph
	graph = createGraph()

	# Already ran - prints the edges to remove for the blog data.
	# pruneAndPrint(myDataPredictionNodeFile,myDataCorrectNodeFile,newGraph)

	# The blog data with edges removed.
	prunedBlogGraph = getPrunedGraph(myDataCorrectNodeFile,newGraph)
	
	# Already ran - prints the edges to remove for the Instagram data.
	# pruneIGAndPrint(updatedIGDataPredictionNodeFile,IGDataCorrectNodeFile,graph)

	# The Instagram data with edges removed. 
	prunedIGGraph = getPrunedIGGraph(IGDataCorrectNodeFile,graph)

	# The four sets of code below have already been run to predict IG/blog data for both my algorithm and Adamic-Adar.

	#Predict (pruned) IG Data - Adamic-Adar
	# alreadyPredicted = set()
	# predictionDict = {}
	# predictNodes(updatedIGDataPredictionNodeFile,AdamicPredictionsOnIGDataFile,prunedIGGraph,alreadyPredicted,predictionDict)

	# Predict (pruned) blog data - Adamic-Adar
	# alreadyPredicted = set()
	# predictionDict = {}
	# predictBlogNodes(myDataPredictionNodeFile,AdamicPredictionsOnNewDataFile,prunedBlogGraph,alreadyPredicted,predictionDict)

	# Predict (pruned) IG Data - My Algorithm
	# alreadyPredicted = set()
	# predictionDict = {}
	# myAlgPredictNodes(updatedIGDataPredictionNodeFile,MyAlgPredictionsOnIGDataFile,prunedIGGraph,alreadyPredicted,predictionDict)

	# Predict (pruned) blog data - My Algorithm
	# alreadyPredicted = set()
	# predictionDict = {}
	# myAlgPredictBlogNodes(myDataPredictionNodeFile,MyAlgPredictionsOnNewDataFile,prunedBlogGraph,alreadyPredicted,predictionDict)

	genderFile = "graph_gender.csv"

	# Prints out the fairness scores.
	print "Fairness(IG Data_Adamic-Adar): " + str(getFairnesScore(AdamicPredictionsOnIGDataFile,genderFile,prunedIGGraph))
	print "Fairness(IG Data_My Algorithm): " + str(getFairnesScore(MyAlgPredictionsOnIGDataFile,genderFile,prunedIGGraph))

	print 

	print "Fairness(Blog Data_Adamic-Adar): " + str(getBlogFairnesScore(AdamicPredictionsOnNewDataFile,affiliations,prunedBlogGraph))
	print "Fairness(Blog Data_My Algorithm): " + str(getBlogFairnesScore(MyAlgPredictionsOnNewDataFile,affiliations,prunedBlogGraph))

	print 

	# Prints out the accuracy scores. 
	print "Accuracy(IG Data_Adamic-Adar): " + str(calcCorrectness(AdamicPredictionsOnIGDataFile,IGDataCorrectNodeFile))
	print "Accuracy(IG Data_My Algorithm): " + str(calcCorrectness(MyAlgPredictionsOnIGDataFile,IGDataCorrectNodeFile))

	print

	print "Accuracy(Blog Data_Adamic-Adar): " + str(calcCorrectness(AdamicPredictionsOnNewDataFile,myDataCorrectNodeFile))
	print "Accuracy(Blog Data_My Algorithm): " + str(calcCorrectness(MyAlgPredictionsOnNewDataFile,myDataCorrectNodeFile))

if __name__ == '__main__':	
	main()