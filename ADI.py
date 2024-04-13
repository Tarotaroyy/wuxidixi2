import py222
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
import MCTS
import constants
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

def getRandomMove():
    return moves[randint(0, len(moves) - 1)]

# TODO: Add the loss weight to each sample?
def generateSamples(k, l, batch_size):
    total_samples = k * l
    all_samples = []
    all_states = []
    
    for _ in range(batch_size):
        currentCube = py222.initState()
        for _ in range(total_samples):
            move = getRandomMove()
            scrambledCube = py222.doAlgStr(currentCube, move)
            state = py222.getState(scrambledCube).flatten()
            all_samples.append(scrambledCube)
            all_states.append(state)
            currentCube = scrambledCube

    samples = np.array(all_samples)
    states = np.array(all_states)
    return samples, coo_matrix(states)


def reward(cube):
    return 1 if py222.isSolved(cube, True) else -1

def doADI(k, l, M, batch_size):
    model = buildModel(constants.kNumStickers * constants.kNumCubes)
    compileModel(model, constants.kLearningRate)

    for iterNum in range(M):
        samples, _ = generateSamples(k, l, batch_size)
        states = np.vstack([py222.getState(sample).flatten() for sample in samples])
        optimalVals = []
        optimalPolicies = []

        for sample in samples:
            values = []
            for move in moves:
                child = py222.doAlgStr(sample, move)
                childState = py222.getState(child).flatten()[None, :]
                value, _ = model.predict(childState)
                value = value[0][0] + reward(child)
                values.append(value)
            
            optimalVals.append(np.max(values))
            optimalPolicies.append(np.argmax(values))

        model.fit(states, {"PolicyOutput": optimalPolicies, "ValueOutput": optimalVals}, 
                  batch_size=batch_size, epochs=constants.kNumMaxEpochs, verbose=False)

        gc.collect()
        print(f"Iteration {iterNum} completed")

    return model



if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("Invalid number of arguments. Must specify model source (-newmodel or -restoremodel) followed by model prefix (can enter 'default' for default prefix) and search strategy (-greedy, -vanillamcts, -fullmcts)")
    else:
        model_prefix = sys.argv[2]
        if model_prefix == "default":
            model_prefix = constants.kModelPath
        if sys.argv[1].lower() == "-newmodel":
            model = doADI(k=20,l=1,M=100)
            model.save("{}.h5".format(model_prefix))
            print("Model saved in path: {}.h5".format(model_prefix))
        elif sys.argv[1].lower() == "-restoremodel":
            model = load_model("{}.h5".format(model_prefix))
            print("Model restored from " + model_prefix)
        else:
            print("Invalid first argument: must be -newmodel or -restoremodel")

        #only simulate cubes upon restoring model for now. can be removed later
        if sys.argv[1].lower() == "-restoremodel":
            if sys.argv[3].lower() == "-greedy":
                MCTS.simulateCubeSolvingGreedy(model, numCubes=50, maxSolveDistance=7)
            if sys.argv[3].lower() == "-vanillamcts":
                MCTS.simulateCubeSolvingVanillaMCTS(model, numCubes=50, maxSolveDistance=7)
            if sys.argv[3].lower() == "-fullmcts":
                MCTS.simulateCubeSolvingFullMCTS(model, numCubes=50, maxSolveDistance=7)




