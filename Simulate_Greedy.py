# Import necessary libraries
from tensorflow.keras.models import load_model  # To load the trained model
import constants
import py222
from MCTS

# Load the trained model using the correct filename
model_path = "{}.h5".format(constants.kModelPath)
model = load_model(model_path)

# Set parameters for the cube solving
scrambleDepth = 10  # Depth of cube scrambling
maxMoves = 100      # Maximum number of moves allowed for solving

# Create a scrambled cube with the specified depth
scrambledCube = py222.createScrambledCube(scrambleDepth)

# Solve the scrambled cube using the greedy algorithm
result, numMoves = MCTS.solveSingleCubeGreedy(model, scrambledCube, maxMoves)

# Check if the cube was solved within the maximum number of moves
if result:
    print("Cube solved in", numMoves, "moves.")
else:
    print("Cube not solved within the maximum number of moves.")

