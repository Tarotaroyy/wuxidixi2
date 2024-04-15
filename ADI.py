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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split



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

    # Early Stopping and Checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(f"{constants.kModelPath}.h5", monitor='val_loss', save_best_only=True, verbose=1)

    for iterNum in range(M):
        samples, states = generateSamples(k, l, batch_size)
        optimalVals = np.empty((len(samples), 1))
        optimalPolicies = np.empty(len(samples), dtype=np.int32)
        
        for i, sample in enumerate(samples):
            values = []
            for move in moves:
                child = py222.doAlgStr(sample, move)
                childState = py222.getState(child).flatten()[None, :]
                value, _ = model.predict(childState)
                values.append(value[0][0] + reward(child))
            
            optimalVals[i] = np.max(values)
            optimalPolicies[i] = np.argmax(values)
        
        # Splitting data into training and validation sets
        X_train, X_val, Y_train_policy, Y_val_policy, Y_train_value, Y_val_value = train_test_split(
            states,
            optimalPolicies,
            optimalVals,
            test_size=0.2,  # 20% for validation
            random_state=42
        )

        # Fit the model
        model.fit(
            X_train, {'PolicyOutput': Y_train_policy, 'ValueOutput': Y_train_value},
            validation_data=(X_val, {'PolicyOutput': Y_val_policy, 'ValueOutput': Y_val_value}),
            batch_size=batch_size,
            epochs=constants.kNumMaxEpochs,
            verbose=True,
            callbacks=[early_stopping, model_checkpoint]
        )
        print(f"Iteration {iterNum+1} completed")

    print("Model training completed. Checkpoint saved at:", constants.kModelPath)



if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ADI.py -newmodel|-restoremodel <model_prefix> -greedy|-vanillamcts|-fullmcts")
    else:
        model_prefix = sys.argv[2]
        if model_prefix == "default":
            model_prefix = constants.kModelPath
        model_path = f"{model_prefix}.h5"

        if sys.argv[1].lower() == "-newmodel":
            model = doADI(k=20, l=1, M=100, batch_size=30)
            print("Model saved at:", model_path)
        elif sys.argv[1].lower() == "-restoremodel":
            model = load_model(model_path)
            print("Model restored from", model_path)
            strategy = sys.argv[3].lower()
            if strategy == "-greedy":
                MCTS.simulateCubeSolvingGreedy(model, numCubes=50, maxSolveDistance=7)
            elif strategy == "-vanillamcts":
                MCTS.simulateCubeSolvingVanillaMCTS(model, numCubes=50, maxSolveDistance=7)
            elif strategy == "-fullmcts":
                MCTS.simulateCubeSolvingFullMCTS(model, numCubes=50, maxSolveDistance=7)
            else:
                print("Invalid search strategy:", strategy)
        else:
            print("Invalid operation mode:", sys.argv[1].lower())





