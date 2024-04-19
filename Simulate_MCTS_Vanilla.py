# Import necessary libraries
from tensorflow.keras.models import load_model  # To load the trained model
import constants
import CubeLib
import MCTS

# Load the trained model using the correct filename
model_path = "{}.h5".format(constants.kModelPath)
model = load_model(model_path)

# Set the parameters
scrambleDepth = 3  # Adjust as needed
maxMoves = 100      # Adjust as needed
maxDepth = 10       # Adjust as needed

# Create a scrambled cube
scrambledCube = CubeLib.createScrambledCube(scrambleDepth)

# Display the initial cube
print("Initial Cube:")
CubeLib.printCube(CubeLib.getNumerical(scrambledCube))

# Solve the cube using the vanilla MCTS algorithm
result, numMoves, scrambledCube = MCTS.solveSingleCubeVanillaMCTS(model, scrambledCube, maxMoves, maxDepth)

# Check if the cube was solved
if result:
    print("Cube solved in", numMoves, "moves.")
else:
    print("Cube not solved within the maximum number of moves.")

# Display the solved cube
print("Solved Cube:")
CubeLib.printCube(CubeLib.getNumerical(scrambledCube))
