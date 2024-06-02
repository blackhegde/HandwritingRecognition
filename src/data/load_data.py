import os
import json
import pandas as pd
import cv2

def load_images_and_labels(directory):
    # Kiểm tra sự tồn tại của thư mục
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"{directory} does not exist.")
    
    # Đọc tệp label.json
    label_file = os.path.join(directory, 'labels.json')
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"{label_file} does not exist.")
    
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    # Đọc tất cả các tệp ảnh trong thư mục
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {directory}.")
    
    # Tạo danh sách chứa thông tin ảnh và label
    data = []
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Unable to load image {image_path}")

        # Lấy label tương ứng từ tệp JSON
        image_label = labels.get(image_file)
        
        data.append({
            'file_name': image_file,
            'image': image,
            'label': image_label
        })
    
    # Chuyển dữ liệu thành DataFrame của Pandas
    df = pd.DataFrame(data)
    return df

# Sử dụng hàm load_images_and_labels
root_path = os.getcwd()
directory = os.path.join(root_path, "data/raw/DataSamples1")

try:
    data_df = load_images_and_labels(directory)
    print("Dữ liệu đã được tải thành công:")
    print(data_df.head())
except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)