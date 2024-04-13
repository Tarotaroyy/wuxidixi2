# Import necessary libraries
from tensorflow.keras.models import load_model  # To load the trained model
import constants
import py222
import MCTS

# Load the trained model using the correct filename
model_path = "{}.h5".format(constants.kModelPath)
model = load_model(model_path)

# Set the parameters
scrambleDepth = 3  # Adjust as needed
maxMoves = 100      # Adjust as needed

# Create a scrambled cube
scrambledCube = py222.createScrambledCube(scrambleDepth)

# Display the initial cube
print("Initial Cube:")
py222.printCube(py222.getNumerical(scrambledCube))

# Solve the cube using the vanilla MCTS algorithm
result, numMoves, path = MCTS.solveSingleCubeFullMCTS(model, scrambledCube, maxMoves)

# Update the scrambled cube using the path
for move in path:
    scrambledCube = py222.doAlgStr(scrambledCube, move)

# Check if the cube was solved
if result:
    print("Cube solved in", numMoves, "moves.")
else:
    print("Cube not solved within the maximum number of moves.")

# Display the solved cube
print("Solved Cube:")
py222.printCube(py222.getNumerical(scrambledCube))
