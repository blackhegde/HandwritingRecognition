import cv2
import os
import shutil
import numpy as np
from sklearn.cluster import KMeans

# Nhị phân hó a ảnh
def binarize(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

# Tiền xử lý ảnh
# Write a function to preprocess img to black and white img, reduce noise or remove background  
def img_to_blackwhite(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #blur to vector
    vectorized = blur.reshape((-1, 1))
    #Kmeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(vectorized)
    #Get the label of each pixel
    labels = kmeans.labels_
    #Reshape the labels to the original image
    labels = labels.reshape((gray.shape))
    # Compare the number of pixels in each cluster
    if np.sum(labels == 0) < np.sum(labels == 1):
        labels = np.where(labels, 0, 1)

    #Get the foreground and background
    foreground = (labels == 1).astype(np.uint8) * 255
    #binarize the image
    binary_image = binarize(foreground)
    return binary_image


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

        # Chuyển ảnh màu sang ảnh đen trắng
        binary_image = img_to_blackwhite(image)

        # Lưu ảnh đã xử lý vào thư mục đầu ra
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        cv2.imwrite(output_path, binary_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
    # # Sao chép labels.json vào thư mục output_dir
    # labels_file = os.path.join(input_dir, 'labels.json')
    # if os.path.exists(labels_file):
    #     output_labels_file = os.path.join(output_dir, 'labels.json')
    #     shutil.copy(labels_file, output_labels_file)
    # else:
    #     print(f"{labels_file} không tồn tại.")

# Test on input_dir is 'data/raw/DataSamples1' and output_dir is 'data/preprocess/DataSamples1'
input_dir = 'data/raw/PrivateTest'
output_dir = 'data/preprocess/test'
preprocess_image(input_dir, output_dir)