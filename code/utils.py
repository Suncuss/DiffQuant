# Transforms and helper functions for the project
import torchvision.transforms as transforms

resize224_and_to_tensor_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15)
])


# File Utils
import config
import csv
import os
from PIL import Image
from torchvision import transforms

def load_folder(path):
    image_paths = []
    for file in sorted(os.listdir(path)):
        if file.lower().endswith(".tif") or file.lower().endswith(".jpg"):
            image_paths.append(os.path.join(path, file))
    return image_paths


def load_prediction_dir(predict_dir):
    image_paths = []
    for files in os.listdir(predict_dir):
        if files.lower().endswith(".tif") or files.lower().endswith(".jpg"):
            image_paths.append(os.path.join(predict_dir, files))
    image_paths = sorted(image_paths)

    return image_paths

def crop_into_five(image_paths, augmented_root_dir):
     for image_path in image_paths:
          image = Image.open(image_path)
          crops = transforms.FiveCrop(size=config.CROP_SIZE)(image)
          for i, crop in enumerate(crops):
               new_file_name = os.path.split(os.path.splitext(image_path)[0])[1]+"_"+str(i)+os.path.splitext(image_path)[1]
               new_path = os.path.join(augmented_root_dir, *new_file_name.split("/"))
               os.makedirs(os.path.dirname(new_path), exist_ok=True)
               print(new_path)
               crop.save(new_path)
          
def vhflip(image_paths, augmented_root_dir):
     for image_path in image_paths:
        image = Image.open(image_path)
        image = transforms.RandomVerticalFlip(p=1)(image)
        image = transforms.RandomHorizontalFlip(p=1)(image)
        new_file_name = os.path.split(os.path.splitext(image_path)[0])[1]+"_vh_flipped"+os.path.splitext(image_path)[1]
        new_path = os.path.join(augmented_root_dir, *new_file_name.split("/"))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        print(new_path)
        image.save(new_path)

def generate_csv(data_dir, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename', 'label'])

        for day in range(5):
            day_dir = os.path.join(data_dir, f'd{day}')
            for img_file in os.listdir(day_dir):
                img_path = os.path.join(day_dir, img_file)
                csv_writer.writerow([img_path, float(day)])


def write_results_to_csv(predictions, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename', 'prediction'])

        for img_path, prediction in predictions:
            csv_writer.writerow([img_path, float(prediction)])

# Other Utils

import matplotlib.pyplot as plt

def visualize_prediction(image, true_label, predicted_label):
    """
    Displays an image with its true label and predicted label.

    Args:
        image (PIL.Image): The image to display.
        true_label (int): The true label (day) of the image.
        predicted_label (int): The predicted label (day) of the image.
    """
    plt.imshow(image)
    plt.title(f"True Day: {true_label}, Predicted Day: {predicted_label}")
    plt.axis("off")
    plt.show()
