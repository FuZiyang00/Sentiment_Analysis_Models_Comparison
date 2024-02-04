#!/bin/bash

# Activate your virtual environment if needed
# source activate your_virtual_env

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Launch the first Jupyter notebook in the background
jupyter notebook classification.ipynb &

# Launch the second Jupyter notebook in the background
jupyter notebook deep_models.ipynb &

