import albumentations as A
from matplotlib import pyplot as plt
import random
import cv2
import numpy as np
from tqdm import tqdm


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def writeًImgAndLabels(num, img, bboxes, category_ids):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./DataOut/images/%d.jpg' % (num), image)
    
    f = open('./DataOut/labels/%d.txt' % num, 'w')
    for i in range(len(category_ids)):
        f.write('%d %f %f %f %f\n' % (int(category_ids[i]), bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]))
    f.close()


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_center_norm, y_center_norm, w_norm, h_norm = bbox
    imageY, imageX, _ = img.shape

    x_center = x_center_norm * imageX
    y_center = y_center_norm * imageY
    bboxW = w_norm * imageX
    bboxH = h_norm * imageY

    x_min, x_max, y_min, y_max = int(x_center - bboxW/2), int(x_center + bboxW/2), int(y_center - bboxH/2), int(y_center + bboxH/2)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


def makeTransforms():
    transforms = []

    transforms.append(A.Compose([
        A.Blur(p=0.5, always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.RandomScale(scale_limit=0.5, always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.Perspective(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.CLAHE(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.Downscale(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.GaussNoise(var_limit=(100.0, 110.0), always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.GaussianBlur(blur_limit= (11, 15), always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.GlassBlur(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.MotionBlur(blur_limit=(15, 20), always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.RandomFog(fog_coef_upper= 0.4, always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    transforms.append(A.Compose([
        A.RandomRain(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))
    
    transforms.append(A.Compose([
        A.RandomShadow(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))
    
    transforms.append(A.Compose([
        A.RandomSunFlare(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))
    
    transforms.append(A.Compose([
        A.RandomToneCurve(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))
    
    transforms.append(A.Compose([
        A.Superpixels(always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])))

    return transforms

def openClassList():
    classes = {}
    file = open('./images/classes.txt', 'r')
    count = 0

    while True:
        name = file.readline()
        if not name:
            break
        classes[count] = name.strip()
        count += 1
    file.close()
    return classes

if __name__ == '__main__':
    
    classes = openClassList()

    transforms = makeTransforms()

    writeImgNum = 1232
    for i in tqdm(range(1232)):
        img = cv2.imread('./images/%d.jpg' % (i))

        if img is None: continue

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = np.loadtxt('./images/%d.txt' % (i))

        bboxes = np.roll(bboxes, -1, axis=1)

        bboxes, category_ids = bboxes[:, :-1], bboxes[:, -1]

        for T in transforms:
            transformed = T(image=image, bboxes=bboxes, category_ids=category_ids)

            writeًImgAndLabels(writeImgNum, transformed['image'], transformed['bboxes'], transformed['category_ids'])

            writeImgNum += 1
        
        writeًImgAndLabels(i, image, bboxes, category_ids)
