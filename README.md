# Single Player Ping Pong Game
This game allows a player to play a single player ping pong game controlled with their finger.
This game uses historgram of oriented gradients (HOG) along with SVM to perform finger detection.
Two models can be used: either the manual hand/finger input model or the 11k hands dataset model. Refer to changing models section below on how to change the model to use.

- [Project presentation](docs/project_overview.pdf)

## Usage:
To play the ping pong game, run main.py on your computer

## Changing Model in Config.py
There are two training models used in this project. One involved manual hand/finger inputs, and another used an 11k hands dataset from the internet. In order to change the training model used when playing the ping pong game, enter the config.py
file and modify the SVM model path as follows:

**Manual hand/finger input model:** SVM_MODEL_PATH = "models/hog_finger_svm.joblib"  
**11K Hands Dataset Model:** SVM_MODEL_PATH = "models/hog_11k_svm.joblib"