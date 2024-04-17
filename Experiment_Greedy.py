# Import necessary libraries
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from tensorflow.keras.models import load_model  # To load the trained model
import constants
import py222
import MCTS

# Set matplotlib to display plots inline
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

# Load the trained model using the correct filename
model_path = "{}.h5".format(constants.kModelPath)
model = load_model(model_path)

# Set parameters for the cube solving
scrambleDepth = 3  # Depth of cube scrambling
maxMoves = 100     # Maximum number of moves allowed for solving
num_trials = 100    # Number of trials to run

# Lists to store results
success_list = []
num_moves_list = []
time_taken_list = []

# Run the experiment for the specified number of trials
for i in range(num_trials):
    # Create a scrambled cube with the specified depth
    scrambledCube = py222.createScrambledCube(scrambleDepth)

    # Solve the scrambled cube using the greedy algorithm
    result, numMoves, _ = MCTS.solveSingleCubeGreedy(model, scrambledCube, maxMoves)

    # Append results to the lists
    success_list.append(result)
    num_moves_list.append(numMoves)

# Calculate success rate
success_rate = sum(success_list) / num_trials * 100

# Display overall results
print("Overall Results:")
print("Success Rate:", success_rate, "%")
print("Average Number of Moves:", sum(num_moves_list) / num_trials)
