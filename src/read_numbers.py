import cv2
import numpy as np
import pytesseract
from keras.models import load_model

def preprocess_cell(cell_img):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_cells(grid_img, grid_size=(9, 9)):
    h, w = grid_img.shape[:2]
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    cells = []
    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            cell = grid_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            row.append(cell)
        cells.append(row)
    return cells

def recognize_number(cell_img, ocr_type='auto', model=None):
    # Try Tesseract first for printed, fallback to model for handwritten
    config = '--psm 10 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(cell_img, config=config)
    text = text.strip()
    if text.isdigit():
        return int(text)
    elif model is not None:
        # Preprocess for MNIST model
        img = cv2.resize(cell_img, (28, 28))
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 28, 28, 1)
        pred = model.predict(img)
        return int(np.argmax(pred))
    else:
        return None

def read_grid_numbers(
        image: np.ndarray, 
        grid_size=(9, 9), 
        model_path=None
    ):
    # img = cv2.imread(image_path)
    cells = extract_cells(image, grid_size)
    model = load_model(model_path) if model_path else None
    numbers = []
    for row in cells:
        num_row = []
        for cell in row:
            proc = preprocess_cell(cell)
            num = recognize_number(proc, model=model)
            num_row.append(num)
        numbers.append(num_row)
    return numbers

