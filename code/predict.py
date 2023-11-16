import torch
import config
import utils
from PIL import Image
from model import ResNetRegression

# Set up hyperparameters and load the trained model
device = torch.device(config.DEVICE)
model = ResNetRegression().to(device)

def predict_images(predict_dir, model_weight_path):
    image_paths = utils.load_prediction_dir(predict_dir)
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    predictions = []
    with torch.no_grad():
        for image_path in image_paths:
            # Load image and apply the same preprocessing used during training
            image = Image.open(image_path)
            image = utils.resize224_and_to_tensor_transforms(image)
            image = image.unsqueeze(0)
            image = image.to(device)

            # Make a prediction
            output = model(image)
            prediciton = output.squeeze().item()
            predictions.append((image_path,prediciton))  
    
    for image_path, prediction in predictions:
        print(f"Image: {image_path}, Predicted Day: {prediction}")
    
    return predictions

