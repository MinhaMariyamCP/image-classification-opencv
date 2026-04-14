import cv2
import numpy as np

def preprocess(images):
    processed = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (32, 32))
        normalized = resized / 255.0
        processed.append(normalized.flatten())
    return np.array(processed)


def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray, enhanced, blurred