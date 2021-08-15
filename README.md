# YOLO-Image-Augmentation
Image augmentation based on yolo labeled images

Images and labels must be in the same directory. Code reads both files and uses Albumentations different methods to generate new data. In the current file, 16 methods are used for generating augmented data and they are saved in seperate folder named 'DataOut'. After generating new data, original image and label are moved to the appropriate folder in 'DataOut'.
