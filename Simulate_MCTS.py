# Import necessary libraries
from tensorflow.keras.models import load_model  # To load the trained model
import constants
import CubeLib
import MCTS
import numpy as np



# Define the findMove function
def findMove(prev_state, curr_state):
    # Define a list of possible moves
    moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

    # Loop through each move
    for move in moves:
        # Apply the move to the previous state
        next_state = CubeLib.doAlgStr(prev_state, move)

        # If the next state matches the current state, return the move
        if np.array_equal(next_state, curr_state):
            return move

    # If no move matches the transition, return None
    return None

# Load the trained model using the correct filename
model_path = "{}.h5".format(constants.kModelPath)
model = load_model(model_path)

# Set the parameters
scrambleDepth = 3  # Adjust as needed
maxMoves = 1000      # Adjust as needed

# Create a scrambled cube
scrambledCube = CubeLib.createScrambledCube(scrambleDepth)

# Display the initial cube
print("Initial Cube:")
CubeLib.printCube(CubeLib.getNumerical(scrambledCube))

# Solve the cube using the MCTS algorithm
result, numMoves, path = MCTS.solveSingleCubeFullMCTS(model, scrambledCube, maxMoves)

# Apply each move in the path to the scrambledCube
for i in range(1, len(path)):
    prev_state = path[i - 1]
    curr_state = path[i]
    move = findMove(prev_state, curr_state)
    scrambledCube = CubeLib.doAlgStr(scrambledCube, move)

# Check if the cube was solved
if result:
    print("Cube solved in", numMoves, "moves.")
else:
    print("Cube not solved within the maximum number of moves.")

# Display the solved cube
print("Solved Cube:")
CubeLib.printCube(CubeLib.getNumerical(scrambledCube))
