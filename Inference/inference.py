import argparse
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt

class_thresholds = {
    'class' :0.6,
    'amphibia': 0.1,
    'aves': 0.5,
    'mammalia': 0.8,
    'serpentes': 0.9
}

labels = {
    'class' : ['amphibia', 'aves', 'invertebrates', 'lacertilia', 'mammalia', 'serpentes', 'testudines'],
    'serpentes': ["Butler's Gartersnake", "Dekay's Brownsnake", 'Eastern Gartersnake', 'Eastern Hog-nosed snake', 'Eastern Massasauga', 'Eastern Milksnake', 'Eastern Racer Snake', 'Eastern Ribbonsnake', 'Gray Ratsnake', "Kirtland's Snake", 'Northern Watersnake', 'Plains Gartersnake', 'Red-bellied Snake', 'Smooth Greensnake'],
    'mammalia': ['American Mink', 'Brown Rat', 'Eastern Chipmunk', 'Eastern Cottontail', 'Long-tailed Weasel', 'Masked Shrew', 'Meadow Jumping Mouse', 'Meadow Vole', 'N. Short-tailed Shrew', 'Raccoon', 'Star-nosed mole', 'Striped Skunk', 'Virginia Opossum', 'White-footed Mouse', 'Woodchuck', 'Woodland Jumping Mouse'],
    'aves': ['Common Yellowthroat', 'Gray Catbird', 'Indigo Bunting', 'Northern House Wren', 'Song Sparrow', 'Sora'],
    'amphibia': ['American Bullfrog', 'American Toad', 'Green Frog', 'Northern Leopard Frog']
}

hierarchical_models = {label: load_model(f"Models/inceptionv3_{label}.h5") for label in labels}

# Move images from subfolder to main folder
def move_images(main_folder):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            source = os.path.join(root, file)
            dest = os.path.join(main_folder, file)
            shutil.move(source, dest)

# Load and preprocess images
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Copy images based on confidence threshold
def copy_images(output_dir, image_path, label, confidence, threshold = 0.6):
    if confidence < threshold:
        human_review_dir = os.path.join(output_dir, 'human', label)
        os.makedirs(human_review_dir, exist_ok=True)
        shutil.copy(image_path, os.path.join(human_review_dir, os.path.basename(image_path)))
    else:
        output_path = os.path.join(output_dir, label)
        os.makedirs(output_path, exist_ok=True)
        shutil.copy(image_path, os.path.join(output_path, os.path.basename(image_path)))

# Process images for inference
def process_images(input_dir, level, output_dir, threshold):
    for fname in os.listdir(input_dir):
        image_path = os.path.join(input_dir, fname)
        images = load_and_preprocess_image(image_path)
        prediction = hierarchical_models["class"].predict(images)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = labels["class"][predicted_class_index]
        confidence = prediction[predicted_class_index]
        print(predicted_class_label)
        copy_images(output_dir, image_path, predicted_class_label, confidence, threshold)
        if level == "species" and predicted_class_label in ['serpentes', 'mammalia', 'aves', 'amphibia']:
            class_output_dir = os.path.join(output_dir, predicted_class_label)
            if os.path.exists(class_output_dir):
                prediction = hierarchical_models[predicted_class_label].predict(images)[0]
                predicted_species_index = np.argmax(prediction)
                predicted_species_label = labels[predicted_class_label][predicted_species_index]
                confidence = prediction[predicted_species_index]
                print(predicted_species_index,predicted_species_label)
                copy_images(class_output_dir, image_path, predicted_species_label, confidence, class_thresholds[predicted_class_label])

# Main function to parse arguments and run the program
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Image Inference with Hierarchical Model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory for input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed images')
    parser.add_argument('--move_images', action='store_true', help='Moves the images within subfolders to main folder in the input directory')
    parser.add_argument('--set_threshold', type=float, default=0.6, help='Confidence threshold for classification')
    parser.add_argument('--class_species', type=str, default='class', help='Class or species for inference')
    
    args = parser.parse_args()
    
    if not os.listdir(args.input_dir):
        print(f"No images found in {args.input_dir}.")
        return

    # Move images if the flag is set
    if args.move_images:
        move_images(args.input_dir)
    
    # Perform inference
    process_images(args.input_dir, args.class_species, args.output_dir, args.set_threshold)

if __name__ == "__main__":
    main()
