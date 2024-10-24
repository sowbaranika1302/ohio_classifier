
# Ohio Classifier

This repository contains a machine learning model for classifying species from images using the `ohio_classifier`. 
The model is built with TensorFlow and leverages a threshold-based classification system to filter results based on confidence scores.

## Table of Contents
- [Ohio Classifier](#ohio-classifier)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Examples](#examples)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Features
- Image classification using a species detection model.
- Threshold-based filtering to manage classification accuracy.
- Input and output directories can be mounted using Docker volumes for easy data management.
- Easily adjustable parameters, such as threshold levels and classification categories.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sowbaranika1302/ohio_classifier.git
   cd ohio_classifier/Inference
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t sowbaranika1302/ohio_classifier:1.0 .
   ```

## Usage

You can use the `ohio_classifier` by running the Docker image and mounting your local directories for input and output. For example:

```bash
docker run -v /path/to/local/images:/input \
           -v /path/to/local/output:/output \
           sowbaranika1302/ohio_classifier:1.0 \
           --input_dir /input \
           --output_dir /output \
           --set_threshold 0.6 \
           --class_species species
```

This command will run the classifier on the images in the specified directory, classify the species, and save the results to the output directory.

## Arguments

- `--input_dir`: Directory containing input images.
- `--output_dir`: Directory where results will be stored.
- `--set_threshold`: (Optional) Confidence threshold for classification. Default is `0.6`.
- `--class_species`: (Optional) Defines the classification category. In this case, it’s set to `species`.

## Examples

### Basic Classification Example

```bash
docker run -v /Users/yourusername/images:/input \
           -v /Users/yourusername/output:/output \
           ohio_classifier:1.0 \
           --input_dir /input \
           --output_dir /output
```

### Set a Custom Threshold

```bash
docker run -v /path/to/images:/input \
           -v /path/to/output:/output \
           ohio_classifier:1.0 \
           --input_dir /input \
           --output_dir /output \
           --set_threshold 0.8
```

## Requirements

- Docker installed on your machine.
- Input images stored in a local directory.
- Sufficient memory and CPU resources to run the container effectively.

## Dataset

The dataset used for this project is available at [LILA Ohio Small Animals](https://lila.science/datasets/ohio-small-animals/).

## Citation

Please cite the following if you use this repository or the dataset for your research:

Balasubramaniam S. Optimized Classification in Camera Trap Images: An Approach with Smart Camera Traps, Machine Learning, and Human Inference. Master’s thesis, The Ohio State University. 2024.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new features.

1. Fork the repository
2. Create a feature branch (`git checkout -b new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin new-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
