import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_and_crop_grid(image_path, output_size=(900, 900)):
    """
    Reads an image, finds the largest rectangular contour (the grid),
    aligns it, and crops it to a consistent rectangle.
    Returns the aligned and cropped grid image.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.bitwise_not(thresh)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon and get the bounding rectangle
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) != 4:
        raise ValueError("Could not find a rectangular grid in the image.")

    # Order the points in consistent order: top-left, top-right, bottom-right, bottom-left
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    # Compute the perspective transform matrix and apply it
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, output_size)

    return warped

def crop_center(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

def plot_original_and_aligned(original_image, aligned_image):

    # Convert images from BGR to RGB for correct display
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    warped_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(warped_rgb)

    return
