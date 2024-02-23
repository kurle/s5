# PyTorch Project for Image Classification

This project implements a convolutional neural network (CNN) for image classification using the MNIST dataset.

## Project Structure

* **S5.ipynb:**  Main Jupyter Notebook script containing model training, evaluation, and potentially visualization code.
* **model.py:** Contains the definition of the `Net` class, representing the CNN architecture.
* **utils.py** Holds utility functions, including:
    * `get_device`: Determines the appropriate device for training (CPU or GPU).
    * `get_data_loaders`: Loads and prepares the MNIST training and testing data.
    * Other potential visualization or data processing utilities.

## How to Run

**1.  Dependencies**

*   Python 3.x
*   PyTorch (refer to the official PyTorch website for installation instructions: https://pytorch.org)
*   torchvision
*   NumPy
*   matplotlib (Optional, for visualizations)
*   tqdm (Optional, for progress bars)

**2. Running the Code**

1.  Clone or download this repository.
2.  Open `S5.ipynb` in a Jupyter Notebook environment.
3.  Execute the cells in order to train the model and evaluate its performance.

## Code Overview

**S5.ipynb:**

*   Import necessary libraries.
*   Set up the device for training (CPU or GPU).
*   Load and prepare the MNIST dataset.
*   Define the CNN model (`Net` class from `model.py`).
*   Instantiate the model, optimizer, and loss function.
*   Training loop (iterates over epochs,  forward pass, backpropagation, etc.).
*   Evaluation loop.
*   (Optional) Visualize training/testing results.

**model.py:**

*   Defines the `Net` class inheriting from `nn.Module`.
*   `__init__`: Initializes convolutional layers and fully connected layers.
*   `forward`: Defines the forward pass through the network.

**utils.py:**

*   `get_device`, `get_data_loaders`, and other helper functions.

## Contact

For any questions or feedback, feel free to open an issue on the repository.
