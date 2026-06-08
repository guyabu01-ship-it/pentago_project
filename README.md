


# Pentago AI & Game Engine 

# Project Goals
The main objective of this project is to build a complete environment for the complex strategy game Pentago, and to develop an intelligent AI opponent using Deep Learning.
The project bridges traditional algorithmic game logic with modern Convolutional Neural Networks (CNN) to evaluate board states, recognize spatial patterns, and make strategic decisions.

# Data Representation: The Base-3 Approach
A unique and crucial aspect of this project's architecture is the way the board data is structured for the neural networks, utilizing a **Base-3 (Ternary)** representation. 
Since every single spot on the Pentago board can exist in exactly one of three states—Empty, Player 1, or Player 2—encoding the board states in base-3 provides the most efficient, mathematically natural, and compressed way to represent the game. This approach significantly optimizes the data processing and helps the CNN models learn winning patterns more effectively.

# Tools & Technologies
* Python 3: The core programming language for all logic and scripting.
* TensorFlow / Keras: Used for designing, compiling, and training the neural networks (CNNs and ANNs).
* Kivy: Chosen for building a responsive, cross-platform graphical user interface (GUI).
* NumPy & Pandas: Essential for matrix manipulation, mathematical operations, and preparing the datasets.

# Project Components
The project is modular and divided into several distinct parts:
1. Game Engine (`pentago.py`): The core logic handling board state, matrix rotations, and rule enforcement. Note: As per official Pentago rules, a win is evaluated only after a rotation move is made.
2. AI Models (`cnn2.py`, `cnn3.py`, `cnn4.py`, `ann_model.py`): Various neural network architectures built to predict the best moves based on the base-3 board states.
3. Data Pipeline (`prepare_data.py`, `analyze_dict.py`): Scripts responsible for cleaning, transforming, and structuring the raw game data into usable datasets for training.
4. Graphical Interface (`pentago_graphics.py`): The visual frontend allowing a human player to interact and play against the AI opponent.

# Data 
Due to GitHub's file size limits, the large dataset is hosted externally.

[Click here to download the Datasets and Models from Google Drive]
https://drive.google.com/drive/folders/1hifMleRkeUxOP4O8XjK_TASQAwiPqMLP?usp=sharing

# How to Use 
1. Clone this repository to your local machine:

   git clone [https://github.com/guyabu01-ship-it/pentago_project.git](https://github.com/guyabu01-ship-it/pentago_project.git)


2. Navigate into the project folder and install the required dependencies:


   pip install tensorflow keras kivy numpy pandas


3. Ensure you have downloaded the Data files from the Drive link and placed them directly in the main project directory.
4. Launch the game interface:
 python pentago_graphics.py


# Contributing

If you find a mistake in the code, encounter bugs, or want to suggest improvements, feel free to open an issue or submit a pull request!

