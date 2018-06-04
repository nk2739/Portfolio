'''
Author: Neil Kumar(nk2739)
Assignment 1 (Artificial Intelligence)
2/15/18
'''

import sys
import collections
import copy
import Queue
import math
import heapq
import operator
import resource

'''
Implements the BFS algorithm.
'''
def bfs(initialBoard):

	usage = resource.getrusage(resource.RUSAGE_SELF)

	initialBoard = initialBoard.replace(",","")

	# Uses a deque to maintain FIFO order
	frontier = collections.deque([initialBoard]) 

	path_record = {}
	explored = set()
	numNodesExpanded = 0
	maxDepth = 0
	depths = {} 
	depths[initialBoard] = 0
	discovered = set()

	while not len(frontier) == 0:
		curState = frontier.pop() 

		explored.add(curState)

		# Reaches goal state then prints to output file
		if reachedGoalState(curState):
			path_to_goal = []
			
			while (curState != initialBoard):
				path_to_goal.insert(0,path_record[curState][1])
				curState = path_record[curState][0]

			runTime = format(getattr(usage,'ru_stime'), '.8f')
			
			if sys.platform == 'darwin':
				ramUsed = format(getattr(usage,'ru_maxrss')/ 1048576.0, '.8f') 

			else:
				ramUsed = format(getattr(usage,'ru_maxrss'), '.8f')

			printDataToFile(path_to_goal,numNodesExpanded,len(path_to_goal),maxDepth,runTime,ramUsed)
			return True 

		neighbors = getNeighbors(curState)
		numNodesExpanded += 1

		# Looks at the children of the current state
		for neighbor in neighbors:
			if neighbor[1] not in discovered:
				if neighbor[1] not in explored:
					path_record[neighbor[1]] = (curState,neighbor[0])
					discovered.add(neighbor[1])
					frontier.appendleft(neighbor[1])
					curDepth = findNodeDepth(neighbor[1],curState,path_record,depths)
					if curDepth > maxDepth:
						maxDepth = curDepth
					
	return False 

'''
Implements the DFS algorithm.
'''
def dfs(initialBoard):

	usage = resource.getrusage(resource.RUSAGE_SELF)
	
	initialBoard = initialBoard.replace(",","")

	# Uses a default list which is LIFO.
	frontier = []
	frontier.append(initialBoard)

	path_record = {}
	explored = set()

	numNodesExpanded = 0
	maxDepth = 0

	depths = {}
	depths[initialBoard] = 0

	discovered = set()

	while not len(frontier) == 0:
		curState = frontier.pop() #pops item at end 

		explored.add(curState)

		if reachedGoalState(curState):
			path_to_goal = []
			
			while (curState != initialBoard):
				path_to_goal.insert(0,path_record[curState][1])
				curState = path_record[curState][0]

			runTime = format(getattr(usage,'ru_stime'), '.8f')
			
			if sys.platform == 'darwin':
				ramUsed = format(getattr(usage,'ru_maxrss')/ 1048576.0, '.8f') 

			else:
				ramUsed = format(getattr(usage,'ru_maxrss'), '.8f')

			printDataToFile(path_to_goal,numNodesExpanded,len(path_to_goal),maxDepth,runTime,ramUsed)

			return True 

		# Reverse the neighbors so the order seen is correct.
		neighbors = getNeighbors(curState)
		neighbors = neighbors[::-1]
		numNodesExpanded += 1

		for neighbor in neighbors:
			if neighbor[1] not in discovered:
				if neighbor[1] not in explored:
					path_record[neighbor[1]] = (curState,neighbor[0])
					frontier.append(neighbor[1])
					discovered.add(neighbor[1])
					curDepth = findNodeDepth(neighbor[1],curState,path_record,depths)
					if curDepth > maxDepth:
						maxDepth = curDepth
					
	return False 

'''
Implements the A* algorithm.
'''
def ast(initialBoard):

	usage = resource.getrusage(resource.RUSAGE_SELF)
	
	initialBoard = initialBoard.replace(",","")
	
	# Manage the frontier using a heap.
	frontier = [] 
	heapq.heappush(frontier, (findManhattanDistance(convertStringToState(initialBoard)),convertDirectionToValue("Up"),initialBoard,0))

	path_record = {}
	explored = set()

	numNodesExpanded = 0
	maxDepth = 0

	depths = {}
	depths[initialBoard] = 0 

	discovered = set()

	while frontier:

		# Sort the frontier and get the minimum.
		curNode = sorted(frontier)[0]
		curState = curNode[2]
		frontier.remove(curNode)

		explored.add(curState)

		if reachedGoalState(curState):
			path_to_goal = []
			
			while (curState != initialBoard):
				path_to_goal.insert(0,path_record[curState][1])
				curState = path_record[curState][0]

			runTime = format(getattr(usage,'ru_stime'), '.8f')

			if sys.platform == 'darwin':
				ramUsed = format(getattr(usage,'ru_maxrss')/ 1048576.0, '.8f') 

			else:
				ramUsed = format(getattr(usage,'ru_maxrss'), '.8f')

			printDataToFile(path_to_goal,numNodesExpanded,len(path_to_goal),maxDepth,runTime,ramUsed)

			return True

		neighbors = getNeighbors(curState)
		numNodesExpanded += 1 

		for neighbor in neighbors:

			# If it has not been seen add it to the frontier.
			if neighbor[1] not in discovered:
				if neighbor[1] not in explored:
					path_record[neighbor[1]] = (curState,neighbor[0])
					discovered.add(neighbor[1])
					manhattanDist = findManhattanDistance(convertStringToState(neighbor[1])) #H(N)
					movesAway = findNodeDepth(neighbor[1],curState,path_record,depths)
					totalDist = movesAway + manhattanDist #F(N)
					heapq.heappush(frontier,(totalDist,convertDirectionToValue(neighbor[0]),neighbor[1],movesAway))

					if movesAway > maxDepth:
						maxDepth = movesAway

			# Has been seen then update its distance if possible. 
			elif neighbor[1] in discovered:
				for i in range(0,len(frontier)):
					if frontier[i][2] == neighbor[1]:
						manhattanDist = findManhattanDistance(convertStringToState(neighbor[1])) #H(N)
						movesAway = depths[curState] + 1
						newTotal = movesAway + manhattanDist

						if movesAway > maxDepth:
							maxDepth = movesAway

						oldTotal = frontier[i][0]

						if newTotal < oldTotal:

							path_record[neighbor[1]] = (curState,neighbor[0])
							decreaseKey(frontier,i,newTotal,movesAway)
							depths[neighbor[1]] = movesAway

	return False

'''
Find the depth of 'node' by finding its parent's depth and adding the distance to it.
'''
def findNodeDepth(node, parent, path, depths):
	depth = 0
	parentDepth = depths[parent]
	initNode = node
			
	while (node != parent):
		node = path[node][0]
		depth += 1

	# Sets the node's depth.
	depths[initNode] = depth + parentDepth
	return depth + parentDepth

'''
Finds the Manhattan distance of the state to the goal.
'''
def findManhattanDistance(curState):
	goalState = [[0,1,2],[3,4,5],[6,7,8]]

	distance = 0

	# Finds the distance of each integer from its position to its goal state position.
	for i in range(1,9):
		iLocation = findValue(curState,i)
		goalLocation = findValue(goalState,i)

		moves = math.fabs(iLocation[0]-goalLocation[0]) + math.fabs(iLocation[1]-goalLocation[1])
		distance += moves

	return int(distance)

'''
Changes the given frontier's data to the new total and new number of moves away.
'''
def decreaseKey(frontier,index,newTotal,newMoves):
	frontier[index] = (newTotal,frontier[index][1],frontier[index][2],newMoves)

'''
Converts a given direction as a string to an integer.
'''
def convertDirectionToValue(direction):

	if direction == "Up":
		return 4

	elif direction == "Down":
		return 3

	elif direction == "Left":
		return 2

	elif direction == "Right":
		return 1

	else:
		return 0

'''
Converts a 2D array to its string form.
'''
def convertStateToString(curState):
	numRows = len(curState)
	stringBoard = ""

	for i in range(numRows):
		for j in range(numRows):
			stringBoard += str(curState[i][j])

	return stringBoard

'''
Converts a 9 digit string to its 2D array representation.
'''
def convertStringToState(curString):
	m = 3
	board = [[0] * m for i in range(m)]
	index = 0
	for i in range(m):
		for j in range(m):
			board[i][j] = int(curString[index])
			index += 1

	return board

'''
Finds the integer 'value' in the 2D array and returns its (X,Y) position.
'''
def findValue(curState, value):

	numRows = len(curState)
	location = [-1,-1]

	for i in range(numRows):
		for j in range(numRows):
			if curState[i][j] == value:
				location[0]=i
				location[1]=j

	return location

'''
Returns the 2D Array of 0 swapped with another integer in a new position.
'''
def swap(curState, oldLocation, newLocation):
	newBoard = curState[:]
	newVal = newBoard[newLocation[0]][newLocation[1]]
	oldVal = newBoard[oldLocation[0]][oldLocation[1]]

	newBoard[oldLocation[0]][oldLocation[1]] = newVal
	newBoard[newLocation[0]][newLocation[1]] = oldVal

	return newBoard

'''
Returns the neighbors of the current state as strings.
'''
def getNeighbors(curStateAsString):
	board = convertStringToState(curStateAsString)
	zeroLocation = findValue(board,0)
	numRows = len(board)

	x = zeroLocation[0]
	y = zeroLocation[1]

	neighbors = []

	#Up
	if x - 1 >= 0 and x - 1 < numRows:
		boardCopy = convertStringToState(curStateAsString)
		newBoard = swap(boardCopy,[x,y],[x-1,y])
		neighbors.append(("Up",convertStateToString(newBoard)))

	#Down
	if x + 1 >= 0 and x + 1 < numRows:
		boardCopy = convertStringToState(curStateAsString)
		newBoard = swap(boardCopy,[x,y],[x+1,y])
		neighbors.append(("Down",convertStateToString(newBoard)))

	#Left
	if y - 1 >= 0 and y - 1 < numRows:
		boardCopy = convertStringToState(curStateAsString)
		newBoard = swap(boardCopy,[x,y],[x,y-1])
		neighbors.append(("Left",convertStateToString(newBoard)))

	#Right
	if y + 1 >= 0 and y + 1 < numRows:
		boardCopy = convertStringToState(curStateAsString)
		newBoard = swap(boardCopy,[x,y],[x,y+1])
		neighbors.append(("Right",convertStateToString(newBoard)))

	return neighbors

'''
Checks to see if the current state is the goal.
'''
def reachedGoalState(curState):
	goalState = '012345678'
	if curState == goalState:
		return True
	return False

'''
Writes the requested information to 'output.txt'.
'''
def printDataToFile(pathToGoal, numNodesExpanded, goalSearchDepth, maxSearchDepth,runTime,maxRAM):
	
	outputFile = open("output.txt","w")
	outputFile.write("path_to_goal: " + str(pathToGoal) + '\n')
	outputFile.write("cost_of_path: " + str(len(pathToGoal)) + '\n')
	outputFile.write("nodes_expanded: " + str(numNodesExpanded) + '\n')
	outputFile.write("search_depth: " + str(goalSearchDepth) + '\n')
	outputFile.write("max_search_depth: " + str(maxSearchDepth) + '\n')
	outputFile.write("running_time: " + str(runTime) + '\n')
	outputFile.write("max_ram_usage: " + str(maxRAM))
	outputFile.close()

# Main entry point
def main(arg1, arg2):

	if arg1 == "bfs":
		bfs(arg2)

	if arg1 == "dfs":
		dfs(arg2)

	if arg1 == "ast":
		ast(arg2)

if __name__ == '__main__':	
	main(sys.argv[1], sys.argv[2])