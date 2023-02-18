# YOLO Image Augmentation

This project provides a set of image augmentation techniques that can be used to improve the performance of YOLO object detection models. The techniques include random cropping, rotation, flipping, and more. This project assumes that you already have a dataset labeled in YOLO format.

## Getting Started

To use this project, you will need to have the labeled image dataset in the format expected by your YOLO model.

Once you have your dataset, you can clone this repository using the following command:

```bash
git clone https://github.com/RastinS/YOLO-Image-Augmentation.git
```

You will also need to install the required Python packages by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the image augmentation techniques provided by this project, you can run the following command:

```bash
python augment.py --count <number_of_images_to_generate>
```

This will generate a set of augmented images based on the input images and their corresponding labels, and save them to the output directory. The augmentation techniques used can be changed in the code. List of available augmentations can be found [Albumentation](https://albumentations.ai) specified using the command line options, and the default values will be used if not specified.

The input directory should contain the original images and their corresponding label files, and the names of these files should start from 1 and go up. For example:

```text
input/
    1.jpg
    1.txt
    2.jpg
    2.txt
    ...
```

The label files should be in the YOLO format, with each line containing the object class and the normalized coordinates of the object's bounding box.

The output directory will be created if it doesn't already exist, and the augmented images will be saved in the following format:

```text
output/
    1.jpg
    1.txt
    2.jpg
    2.txt
    ...
```
    
## Contributing

Contributions to this project are welcome! If you find a bug or have an idea for a new feature, please open an issue on the GitHub repository.

If you would like to contribute code, please fork the repository and submit a pull request. Please make sure to follow the existing code style and include tests for any new functionality.
