import CubeLib
from random import randint
import numpy as np
import tensorflow as tf
import os
import sys
from scipy.sparse import coo_matrix
import collections
import math
import gc
from CubeModel import buildModel, compileModel
from tensorflow.keras.models import load_model
import constants

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

def reward(cube):
    return 1 if CubeLib.isSolved(cube, True) else -1

def solveSingleCubeGreedy(model, cube, maxMoves):
    for move_num in range(maxMoves):
        if CubeLib.isSolved(cube):
            return True, move_num, cube  # Now returning the cube state as well
        state = CubeLib.getState(cube).flatten()[None, :]
        _, policy = model.predict(state)
        best_move = np.argmax(policy[0])
        cube = CubeLib.doAlgStr(cube, moves[best_move])
    return False, maxMoves, cube  # Include the cube in the return statement


def solveSingleCubeVanillaMCTS(model, cube, maxMoves, maxDepth):
    numMovesTaken = 0
    q = {}
    counts = {}
    while numMovesTaken <= maxMoves:
        if CubeLib.isSolved(cube, convert=True):
            return True, numMovesTaken, cube
        bestMove = selectActionVanillaMCTS(model, cube, maxDepth, q, counts)
        if bestMove == -1:
            print("something went wrong when selecting best move")
            break
        cube = CubeLib.doAlgStr(cube, moves[bestMove])
        numMovesTaken += 1
    return False, maxMoves+1, cube

def selectActionVanillaMCTS(model, state, depth, q, counts):
    stateStr = str(state)
    #q = {}
    #counts = {}
    seenStates = set()
    for i in range(constants.kMCTSSimulateIterations):
        simulateVanillaMCTS(model, state, depth, q, counts, seenStates, stateStr)
    allVals = np.zeros(len(moves))
    for i in range(len(moves)):
        allVals[i] = q[stateStr][moves[i]]
    return allVals.argmax()

def simulateVanillaMCTS(model, state, depth, q, counts, seenStates, stateStr):
    if depth == 0:
        return 0
    if stateStr not in seenStates:
        q[stateStr] = {}
        counts[stateStr] = {}
        for move in moves:
            nextState = CubeLib.doAlgStr(state, move)
            nextStateArray = np.array([CubeLib.getState(nextState).flatten()])
            value, _ = model.predict(nextStateArray)
            q[stateStr][move] = value + reward(nextState)
            counts[stateStr][move] = 1
        seenStates.add(stateStr)
        return rolloutVanillaMCTS(model, state, depth)
    totalStateCounts = 0
    for move in moves:
        totalStateCounts += counts[stateStr][move]
    allQuantities = np.zeros(len(moves))
    for i in range(len(moves)):
        allQuantities[i] = q[stateStr][moves[i]] + constants.kMCTSExploration * math.sqrt(math.log(totalStateCounts)/counts[stateStr][moves[i]])
    bestActionIndex = allQuantities.argmax()
    bestMove = moves[bestActionIndex]
    nextState = CubeLib.doAlgStr(state, bestMove)
    r = reward(nextState)
    newQ = r + constants.kDiscountFactor * simulateVanillaMCTS(model, nextState, depth - 1, q, counts, seenStates, str(nextState))
    counts[stateStr][bestMove] += 1
    q[stateStr][bestMove] += (newQ - q[stateStr][bestMove])/counts[stateStr][bestMove] 
    return newQ

def rolloutVanillaMCTS(model, cube, depth):
    if depth == 0:
        return 0
    state = np.array([CubeLib.getState(cube).flatten()])
    _, policies = model.predict(state)
    actionIndex = selectActionSoftmax(policies)
    nextState = CubeLib.doAlgStr(cube, moves[actionIndex])
    r = reward(nextState)
    return r + constants.kDiscountFactor * rolloutVanillaMCTS(model, nextState, depth - 1)

def selectActionSoftmax(probabilities):
    probabilities = probabilities[0]
    weights = np.zeros(len(probabilities))
    for i in range(len(probabilities)):
        weights[i] = math.exp(constants.kLambda * probabilities[i])
    return weighted_choice(weights)

#this code stolen off stackoverflow (thank you kind stranger)
def weighted_choice(weights):
    totals = np.cumsum(weights)
    norm = totals[-1]
    throw = np.random.rand()*norm
    return np.searchsorted(totals, throw)



def initStateVals(stateStr, counts, maxVals, priorProbabilities, virtualLosses, probs):
    counts[stateStr] = {}
    maxVals[stateStr] = {}
    priorProbabilities[stateStr] = {}
    virtualLosses[stateStr] = {}
    for i, move in enumerate(moves):
        counts[stateStr][move] = 0
        maxVals[stateStr][move] = 0
        virtualLosses[stateStr][move] = 0
        priorProbabilities[stateStr][move] = probs[i]       

def simulateCubeSolvingGreedy(model, numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = CubeLib.createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeGreedy(model, scrambledCube, 6 * currentSolveDistance + 1)
            print(numMoves, numMoves != 6*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)

def simulateCubeSolvingVanillaMCTS(model, numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    solveLengths = []
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = CubeLib.createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeVanillaMCTS(model, scrambledCube, 6 * currentSolveDistance + 1, 1)
            print(numMoves, numMoves != 6*currentSolveDistance + 2)
            if result:
                solveLengths.append(numMoves)
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)
    solveLengths.sort()
    print(solveLengths[len(solveLengths)//2])

