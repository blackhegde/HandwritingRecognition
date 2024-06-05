import cv2
import os
import shutil

# Nhị phân hó a ảnh
def binarize(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

# Tiền xử lý ảnh
def preprocess_image(input_dir, output_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"{input_dir} does not exist.")
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir)if f.endswith(('.png', '.jpg', '.jepg'))]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    # Xu ly tung anh
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        # Đọc ảnh đầu vào
        image = cv2.imread(image_path)

        # Chuyển sang ảnh mức xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Nhị phân hóa ảnh
        binary_image = binarize(gray_image)

        # Lưu ảnh đã xử lý vào thư mục đầu ra
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        cv2.imwrite(output_path, binary_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
    # Sao chép labels.json vào thư mục output_dir
    labels_file = os.path.join(input_dir, 'labels.json')
    if os.path.exists(labels_file):
        output_labels_file = os.path.join(output_dir, 'labels.json')
        shutil.copy(labels_file, output_labels_file)
    else:
        print(f"{labels_file} không tồn tại.")