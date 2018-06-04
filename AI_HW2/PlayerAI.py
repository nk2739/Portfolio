from random import randint
from BaseAI import BaseAI
import time
import random 
import math
'''
Neil Kumar (nk2739)
AI Homework 2
3/20/18
'''

class PlayerAI(BaseAI):

	'''
	Returns the optimal move for the maximizing agent.
	'''
	def getMove(self, grid):

		moves = grid.getAvailableMoves()
		optimalMove = -1

		# Allots each move a fraction of the total move time 
		timeLimit = 0.2 / float(len(moves))
		optimalMoveVal = -1*float("inf")

		# Goes through the available moves and runs minimax on each
		for move in moves:

			gridCopy = grid.clone()
			gridCopy.move(move)

			# Makes sure each move runs within a specific time limit 
			initTime = time.clock()
			curTime = initTime
			alpha = -1*float("inf")
			beta = float("inf")
			depth_limit = 3

			(state,score) = self.maximize(gridCopy,alpha,beta,depth_limit,initTime,curTime,timeLimit)

			# Selects the max-scored move amongst all moves 
			if score > optimalMoveVal:
				optimalMoveVal = score
				optimalMove = move

		return optimalMove

	'''
	Evaluates a board based on a number of heuristics 
	'''
	def evaluateGrid(self,grid):
		numAvailableCells = len(grid.getAvailableCells()) 
		smoothness = self.calcSmoothness(grid) 
		monotonicity = self.calcMonotonicity(grid) 
		totalGridValue = self.getGridTotalValue(grid)
		maxTileValue = grid.getMaxTile()
		distanceToMaxValue = self.getDistanceToMax(grid)
		averageTileValue = self.getGridAverageTileValue(grid)

		normSmoothness = float(math.log(smoothness)/math.log(2))
		normNumAvailable = math.log(numAvailableCells)/math.log(2)  if numAvailableCells != 0 else 0
		
		availableCellWeight = 13.5
		monotonicityWeight = 5
		smoothnessWeight = 6
		maxTileWeight = 1.5
		distanceToMaxValueWeight = 25
		averageTileValueWeight = 0.00002

		return (availableCellWeight*normNumAvailable + maxTileWeight*maxTileValue) \
		- (monotonicityWeight*monotonicity + smoothnessWeight*normSmoothness + distanceToMaxValueWeight*distanceToMaxValue)

	'''
	The 'Maximize' function used in Expectiminimax.
	Used to maximize the player's score.
	'''
	def maximize(self,grid,alpha,beta,depth,initTime,curTime,timeLimit):
		
		moves = grid.getAvailableMoves()
		maxChild = None
		maxUtil = -1*float('inf')

		# Terminal Test - done when no more moves, depth limit reached, or time is up
		if (len(moves) == 0 or depth == 0 or time.clock() - initTime >= timeLimit):
			return (maxChild, self.evaluateGrid(grid))

		# Runs through each "child" or state where the next move is played
		for move in moves:
			gridCopy = grid.clone()
			gridCopy.move(move)

			# Runs minimize from maximize
			newTime = time.clock()
			(newChild,newUtil) = self.minimize(gridCopy,alpha,beta,depth,initTime,newTime,timeLimit)

			if newUtil > maxUtil:
				maxChild = gridCopy
				maxUtil = newUtil

			# Prunes if necessary 
			if maxUtil >= beta:
				break

			if maxUtil > alpha:
				alpha = maxUtil

		return (maxChild,maxUtil)

	'''
	The 'Minimize' function used in Expectiminimax. 
	Used to minimize the player's score. 
	'''
	def minimize(self,grid,alpha,beta,depth,initTime,curTime,timeLimit):
		
		openCells = grid.getAvailableCells()
		minChild = None
		minUtil = float('inf')

		# Terminal Test - done when no free tiles, depth limit is reached, or time is up 
		if (len(openCells) == 0 or depth == 0 or time.clock() - initTime >= timeLimit):
			return (minChild, self.evaluateGrid(grid))

		# Goes through each "child" or way to place a 2 or 4 tile
		for cell in openCells:
		
			# Gets a probability in the range [0,1)
			p = random.random() 

			# Chooses a tile to insert 
			toInsert = 4 if p >= 0.9 else 2 

			gridCopy = grid.clone()
			gridCopy.insertTile(cell,toInsert)

			# Runs maximize
			newTime = time.clock()
			(newChild,newUtil) = self.maximize(gridCopy,alpha,beta,depth-1,initTime,newTime,timeLimit)

			if newUtil < minUtil:
				minChild = gridCopy
				minUtil = newUtil

			# Prunes if necessary 
			if minUtil <= alpha:
				break

			if minUtil < beta:
				beta = minUtil

		return (minChild,minUtil)

	'''
	Returns the distance from the max element to the nearest corner.
	Aims to keep the max element near the corners.
	'''
	def getDistanceToMax(self,grid):
		maxTileValue = grid.getMaxTile()
		minDistance = float("inf")

		corners = [(0,0),(0,grid.size-1),(grid.size-1,0),(grid.size-1,grid.size-1)]

		for x in xrange(grid.size):
			for y in xrange(grid.size):

				if (grid.map[x][y]==maxTileValue):

					for corner in corners:
						dist = abs(corner[0]-x) + abs(corner[1]-y)

						# Returns the smallest distance to the nearest corner
						if dist < minDistance:
							minDistance = dist

		# Returns the distance of the max tile to the nearest corner 
		return minDistance

	'''
	Calculates the 'Monotonicity' of the grid or the number of times values
	change between neighbors regarding growth in size. 
	'''
	def calcMonotonicity(self,grid):

		totalMisplaced = 0
		row1 = [(0,0),(0,1),(0,2),(0,3)]
		row2 = [(1,0),(1,1),(1,2),(1,3)]
		row3 = [(2,0),(2,1),(2,2),(2,3)]
		row4 = [(3,0),(3,1),(3,2),(3,3)]
		rows = [row1,row2,row3,row4]

		# Calculates the number of changes in the Left/Right direction
		for row in rows:
			totalMisplaced += self.checkNumMisplaced(row,grid,True)

		# Calculates the number of changes in the Up/Down direction.
		col1 = [(0,0),(1,0),(2,0),(3,0)]
		col2 = [(0,1),(1,1),(2,1),(3,1)]
		col3 = [(0,2),(1,2),(2,2),(3,2)]
		col4 = [(0,3),(1,3),(2,3),(3,3)]

		cols = [col1,col2,col3,col4]

		for col in cols:
			totalMisplaced += self.checkNumMisplaced(col,grid,False)

		# Returns the total number of times values vary
		return totalMisplaced

	'''
	Used to calculate monotonicity. For each direction calculates the number of times
	the values don't consistently grow or shrink.
	'''
	def checkNumMisplaced(self,vals,grid,leftRight):

		numMisplaced = 0

		for i in xrange(len(vals)):
			curVal = grid.map[vals[i][0]][vals[i][1]]

			if not (curVal == 0):

				if i+1 < grid.size:
					nextVal = grid.map[vals[i+1][0]][vals[i+1][1]]

					# For each value sees if its neighbor is correctly smaller or larger
					if not (nextVal == 0):

						# Increases going right in the Left/Right direction
						if leftRight:

							if nextVal < curVal:
								numMisplaced += 1

						# Decreases going down in the Up/Down direction
						else:

							if nextVal > curVal:
								numMisplaced += 1

		# Returns the number of inconsistencies for that specific direction
		return numMisplaced

	'''
	Calculates the 'Smoothness' or difference between adjacent cells.
	'''
	def calcSmoothness(self,grid):

		totalVal = 0

		for x in xrange(grid.size):
			for y in xrange(grid.size):
				curVal = grid.map[x][y]

				# Only cares about non-zero values
				if (curVal > 0):

					# Looks at the value of each of its neighbors 
					neighbors = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

					for neighbor in neighbors:
						if not grid.crossBound(neighbor):

							# Only cares about non-zero neighbors as well
							if (neighbor > 0):

								diff = abs(curVal-grid.map[neighbor[0]][neighbor[1]])
								totalVal += diff

		# Returns the total differences for each node combined 
		return totalVal

	'''
	Calculates the total of all the grid cell values.
	'''
	def getGridTotalValue(self,grid):
		total = 0
		
		for x in xrange(grid.size):
			for y in xrange(grid.size):
				total += grid.map[x][y]

		# Returns the total for all grid cells 
		return total

	'''
	Returns the average tile value by getting the total and dividing by the # of cells.
	'''
	def getGridAverageTileValue(self,grid):

		return float(self.getGridTotalValue(grid))/float(pow((grid.size),2))