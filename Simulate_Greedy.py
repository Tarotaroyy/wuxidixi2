# Import necessary libraries
from tensorflow.keras.models import load_model  # To load the trained model
import constants
import py222
import MCTS

# Load the trained model using the correct filename
model_path = "{}.h5".format(constants.kModelPath)
model = load_model(model_path)

# Set parameters for the cube solving
scrambleDepth = 2  # Depth of cube scrambling
maxMoves = 100     # Maximum number of moves allowed for solving

# Create a scrambled cube with the specified depth
scrambledCube = py222.createScrambledCube(scrambleDepth)

# Display the initial cube
print("Initial Cube:")
py222.printCube(py222.getNumerical(scrambledCube))

# Solve the scrambled cube using the greedy algorithm
result, numMoves, scrambledCube = MCTS.solveSingleCubeGreedy(model, scrambledCube, maxMoves)

# Check if the cube was solved within the maximum number of moves
if result:
    print("Cube solved in", numMoves, "moves.")
else:
    print("Cube not solved within the maximum number of moves.")


# Display the solved cube
print("Solved Cube:")
py222.printCube(py222.getNumerical(scrambledCube))
