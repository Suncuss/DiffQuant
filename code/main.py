import argparse
import train
import utils
import os
from predict import predict_images

def main(args):
    if args.task == "train":
        train.train_model(train_csv_path=args.train_csv_path, eval_csv_path=args.eval_csv_path,model_weight_path=args.model_weight_path)
    elif args.task == "predict":        
        predictions = predict_images(args.predict_dir, args.model_weight_path)
        utils.write_results_to_csv(predictions, args.output_csv)      
    elif args.task == "preprocess":
        image_pathes = utils.load_folder(args.preprocess_dir)
        augmented_root_dir = os.path.join(args.preprocess_dir, "augmented")
        utils.crop_into_five(image_pathes, augmented_root_dir)
    else:
        raise ValueError(f"Invalid task: {args.task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Morphology Prediction")
    parser.add_argument("task", choices=["train", "evaluate", "predict","preprocess"],
                        help="Task to perform: train, evaluate, or predict")
    parser.add_argument("--predict_dir", type=str, default="",
                        help="Comma-separated list of image paths for prediction")
    parser.add_argument("--output_csv", type=str, default="", help="Path to output CSV file")
    parser.add_argument("--preprocess_dir", type=str, default="",help="Path to image files that has not been preprocessed")
    parser.add_argument("--model_weight_path", type=str, default="",help="Path to model weight file for prediction")
    parser.add_argument("--train_csv_path", type=str, default="MyoQuant_data/train_labels.csv", help="Path to training CSV file")
    parser.add_argument("--eval_csv_path", type=str, default="MyoQuant_data/eval_labels.csv", help="Path to evaluation CSV file")
    args = parser.parse_args()
    main(args)
