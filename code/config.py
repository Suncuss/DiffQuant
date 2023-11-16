import torch
from datetime import datetime

# Training parameters
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# Model save path
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_SAVE_PATH = f"trained_model_{current_time}.pth"

# Transform parameters
CROP_SIZE = 1000

# Evaluation parameters
EVAL_BATCH_SIZE = 32

if torch.cuda.is_available():
    DEVICE ='cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
