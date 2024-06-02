import cv2
import numpy as np
import os

# Chuyển ảnh sang mức xám
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Nhị phân hóa ảnh
def binarize(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

# Chiếu lược đồ sáng dọc
def vertical_projection(binary_image):
    return np.sum(binary_image, axis=0)

# Phân chia ký tự theo lược đồ sáng
def find_character_boundaries(vertical_projection, threshold):
    boundaries = []
    in_character = False
    start = 0

    for i, total in enumerate(vertical_projection):
        if not in_character and total > threshold:
            in_character = True
            start = i
        elif in_character and total <= threshold:
            in_character = False
            boundaries.append((start, i - 1))
    
    if in_character:
        boundaries.append((start, len(vertical_projection) - 1))

    return boundaries

# Lưu các ký tự tách rời
def save_characters(image, boundaries, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (start, end) in enumerate(boundaries):
        character = image[:, start:end + 1]
        if character.shape[1] > 0:  # Ensure there is a non-zero width
            character_resized = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(output_dir, f'character_{i}.png'), character_resized)

# Tiền xử lý ảnh
def preprocess_image(image_path, output_dir):
    image = cv2.imread(image_path)
    gray_image = grayscale(image)
    binary_image = binarize(gray_image)
    vertical_profile = vertical_projection(binary_image)

    threshold_mean = np.mean(vertical_profile) / 2
    threshold_otsu, _ = cv2.threshold(vertical_profile.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(threshold_mean, threshold_otsu)

    boundaries = find_character_boundaries(vertical_profile, threshold)
    save_characters(binary_image, boundaries, output_dir)

if __name__ == "__main__":
    root_path = os.getcwd()
    input_image_path = os.path.join(root_path, "data", "raw", "DataSamples1", "1.jpg")
    output_directory = os.path.join(root_path, "data", "processed", "DataSamples1")

    try:
        preprocess_image(input_image_path, output_directory)
        print("Tiền xử lý ảnh thành công!")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
