# Feedforward Network

This project is part of the implements a feedforward neural network using PyTorch to com

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/amir0135/Feedforward-Network.git
    cd Feedforward-Network
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Generate data:**
    ```bash
    python scripts/generate_data.py
    ```

4. **Train the model:**
    ```bash
    python scripts/train.py
    ```

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy

## Description

### Models

- `feedforward_ensemble.py`: Contains the definition of the `FeedforwardEnsembleNetwork` class, which implements the feedforward neural network.

### Utils

- `data_utils.py`: Contains utility functions for reading data from CSV files and saving tensors to CSV files.

### Scripts

- `train.py`: Script to train the feedforward neural network model.
- `generate_data.py`: Script to generate random data and save it to CSV files needed for training and testing.

### Data

- The `data` directory will contain the generated CSV files used for training and testing the model.

## Running the Project

1. **Generate Data:**

   Run the `generate_data.py` script to create the necessary CSV files in the `data` directory.

    ```bash
    python scripts/generate_data.py
    ```

2. **Train the Model:**

   Run the `train.py` script to train the feedforward neural network model.

    ```bash
    python scripts/train.py
    ```
