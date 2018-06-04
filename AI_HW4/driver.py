#!/usr/bin/env python
# coding:utf-8

"""
Usage:
$ python3 driver.py <81-digit-board>
$ python3 driver.py   => this assumes a 'sudokus_start.txt'

Saves output to output.txt
"""
'''
Neil Kumar
AI HW4
5/1/18
'''
import sys

ROW = "ABCDEFGHI"
COL = "123456789"
TIME_LIMIT = 1.  # max seconds per board
out_filename = 'output.txt'
src_filename = 'sudokus_start.txt'

# Sets for each 3x3 "Box" on the Sudoku board 
box1 = set(['A1','A2','A3','B1','B2','B3','C1','C2','C3'])
box2 = set(['A4','A5','A6','B4','B5','B6','C4','C5','C6'])
box3 = set(['A7','A8','A9','B7','B8','B9','C7','C8','C9'])
box4 = set(['D1','D2','D3','E1','E2','E3','F1','F2','F3'])
box5 = set(['D4','D5','D6','E4','E5','E6','F4','F5','F6'])
box6 = set(['D7','D8','D9','E7','E8','E9','F7','F8','F9'])
box7 = set(['G1','G2','G3','H1','H2','H3','I1','I2','I3'])
box8 = set(['G4','G5','G6','H4','H5','H6','I4','I5','I6'])
box9 = set(['G7','G8','G9','H7','H8','H9','I7','I8','I9'])

# A list of all boxes 
boxes = [box1,box2,box3,box4,box5,box6,box7,box8,box9]

def print_board(board):
    """Helper function to print board in a square."""
    print "-----------------"
    for i in ROW:
        row = ''
        for j in COL:
            row += (str(board[i + j]) + " ")
        print row


def string_to_board(s):
    """
        Helper function to convert a string to board dictionary.
        Scans board L to R, Up to Down.
    """
    return {ROW[r] + COL[c]: int(s[9 * r + c])
            for r in range(9) for c in range(9)}


def board_to_string(board):
    """Helper function to convert board dictionary to string for writing."""
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return ''.join(ordered_vals)


def write_solved(board, f_name=out_filename, mode='w+'):
    """
        Solve board and write to desired file, overwriting by default.
        Specify mode='a+' to append.
    """
    result = backtracking(board)
    # print result  # TODO: Comment out prints when timing runs.
    # print

    # Write board to file
    outfile = open(f_name, mode)
    outfile.write(result)
    outfile.write('\n')
    outfile.close()

    return result

# Returns a set of all unassigned variables on the board (value = 0). 
def getUnassignedVariables(board):

    unassigned = set()

    for position in board:
        if board[position] == 0:
            unassigned.add(position)

    return unassigned

# Returns a set of all the "neighbors" of a position - positions in the same box, row, and column.
def getNeighbors(board,place):

    placeRow = place[0]
    placeCol = place[1]

    neighbors = set()

    for position in board:

        # Does not include the current position
        if position != place:
            row = position[0]
            col = position[1]

            sameBox = False

            for box in boxes:
                if (position in box and place in box):
                    sameBox = True

            if (row == placeRow or col == placeCol or sameBox == True):
                neighbors.add(position)

    return neighbors

# Returns a dictionary of "inferences", or assignments for unassigned variables.
def getInferences(board,var,val,unassignedVars):

    inferences = {}

    for var in unassignedVars:

        valuesAvailable = getAvailableValues(board,var)

        if len(valuesAvailable) == 1:
            inferences[var] = list(valuesAvailable)[0]

    return inferences

# Returns the current domain for a position based on its neighbors. 
def getAvailableValues(board,place):

    neighbors = getNeighbors(board,place)

    allVals = set([1,2,3,4,5,6,7,8,9])

    for neighbor in neighbors:
        curVal = board[neighbor]

        if (curVal in allVals):
            allVals.remove(curVal)

    return allVals

# Sees if the board is consistent if a value is placed in this position.
def checkConsistency(board,place,value):

    neighbors = getNeighbors(board,place)

    for neighbor in neighbors:
        curVal = board[neighbor]

        if (curVal == value):
            return False

    return True

# Used to compare my results with the correct Sudoku boards.
def compareResults():

    outputFile = open(out_filename,'r')
    correctFile = open('sudokus_finish.txt','r')
    outData = outputFile.readlines()
    correctData = correctFile.readlines()

    num_lines = len(outData)
    
    numCorrect = 0

    for i in range(0,num_lines):
        myPrediction = outData[i]
        correctAnswer = correctData[i]

        if myPrediction == correctAnswer:
            numCorrect += 1

    print numCorrect


# Returns the unassigned variable with the least number of possible values (using MRV).
def getBestUnassignedVariable(board,unassignedVars):

    bestVar = ''
    bestVarOptions = ''
    numRemainingValues = float("inf")

    for var in unassignedVars:
        allVals = set([1,2,3,4,5,6,7,8,9])

        neighbors = getNeighbors(board,var)

        for neighbor in neighbors: 
            curVal = board[neighbor]

            if (curVal in allVals):
                allVals.remove(curVal)

        if len(allVals) < numRemainingValues:
            bestVar = var
            bestVarOptions = allVals
            numRemainingValues = len(allVals)

    return (bestVar,bestVarOptions)

# Checks to see if a board is complete. 
def checkBoardComplete(board):

    for position in board:
        
        if (board[position] == 0 or checkConsistency(board,position,board[position])==False):
            return False

    return True

def backtracking(board):
    """Takes a board and returns solved board."""
    # TODO: implement this
    
    boardComplete = checkBoardComplete(board)
    if boardComplete: return board_to_string(board)

    unassignedVars = getUnassignedVariables(board)
    bestUnassignedVarVals = getBestUnassignedVariable(board,unassignedVars)
    
    # Unassigned variable with the fewest remaining values 
    unassignedVar = bestUnassignedVarVals[0]

    # The domain of this unassigned variable
    unassignedVarValues = bestUnassignedVarVals[1]

    for val in unassignedVarValues:
        if (checkConsistency(board,unassignedVar,val) == True):
            
            # Adds (var = value) to assignment
            board[unassignedVar] = val 
            
            newUnassigned = getUnassignedVariables(board)
            inferences = getInferences(board,unassignedVar,val, newUnassigned)
           
            inferenceFailure = False

            # Inferences fail if at least one is inconsistent 
            for inference in inferences:
                if (checkConsistency(board,inference,inferences[inference]) == False):
                    inferenceFailure = True

            # Assigns inferences
            if inferenceFailure == False:
                for inference in inferences:
                    board[inference] = inferences[inference]

                result = backtracking(board)
                if not (result == "failure"):
                    return result      

            # Resets values if a mistake is made 
            board[unassignedVar] = 0
            for inference in inferences:
                board[inference] = 0

    return "failure"

if __name__ == '__main__':

    # compareResults()
    if len(sys.argv) > 1:  # Run a single board, as done during grading
        board = string_to_board(sys.argv[1])
        write_solved(board)

    else:
        print "Running all from sudokus_start"

        #  Read boards from source.
        try:
            srcfile = open(src_filename, "r")
            sudoku_list = srcfile.read()
        except:
            print "Error reading the sudoku file %s" % src_filename
            exit()

        # Solve each board using backtracking
        for line in sudoku_list.split("\n"):

            if len(line) < 9:
                continue

            # Parse boards to dict representation
            board = string_to_board(line)
            # print_board(board)  # TODO: Comment this out when timing runs.

            # Append solved board to output.txt
            write_solved(board, mode='a+')

        print "Finished all boards in file."