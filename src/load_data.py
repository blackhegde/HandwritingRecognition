import os
import json
import cv2
import tensorflow as tf
import numpy as np

# Danh sách các ký tự được hỗ trợ
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂĂÀÁẢẠÃẤẦẨẬẪẮẰẲẶẴÊẸÈÉẺẼẾỀỂỆỄỊÌÍỈĨÔỌÒÓỎÕỐỒỔỘỖỜỚỞỢỠỤÙÚỦŨỨỪỬỰỮÝỲỴỶỸĐàáảạãấầẩậẫắằẳặẵẹêẻèéẽếềểễệỉìíĩịơọòóỏõốồổộỗờớởợỡừùúủũụứừửựữỵỳỹýỷỹđâăôịưự,./-#'() "
CHAR_DICT = len(characters) + 1
MAX_LEN = 70
# Tạo từ điển ánh xạ ký tự thành số nguyên
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}


# Load dữ liệu trong thư mục
def load_images_and_labels(directory):
    # Kiểm tra sự tồn tại của thư mục
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"{directory} does not exist.")
    
    # Đọc tệp label.json
    label_file = os.path.join(directory, 'labels.json')
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"{label_file} does not exist.")
    
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # Đọc tất cả các tệp ảnh trong thư mục
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {directory}.")
    
    # Tạo danh sách chứa thông tin ảnh và label
    data = []
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Unable to load image {image_path}")

        # Lấy label tương ứng từ tệp JSON
        image_label = labels.get(image_file)
        
        data.append({
            'file_name': image_file,
            'image': image,
            'label': image_label
        })
    
    return data

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, characters, batch_size=32, input_shape=(32, 128, 1)):
        self.data = data
        self.characters = characters
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.indexes = np.arange(len(data))

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in indexes]
        X, y, input_lengths, label_lengths = self.prepare_data(batch_data)
        return (X, y, input_lengths, label_lengths)  # Trả về tuple (inputs, targets)

    def prepare_data(self, batch_data):
        X = np.zeros((len(batch_data), *self.input_shape), dtype=np.float32)
        y = []
        input_lengths = []
        label_lengths = []
        for i, row in enumerate(batch_data):
            image = row['image']
            text = row['label']
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            image = np.expand_dims(image, axis=-1)
            image = image.astype(np.float32) / 255.0
            X[i] = image
            encoded_text = []
            for c in text:
                if c in self.characters:
                    encoded_text.append(self.characters.index(c))
                else:
                    print(f"Character '{c}' not found in characters list")
            y.append(encoded_text)
            input_lengths.append(self.input_shape[1] // 4)
            label_lengths.append(len(text))
        y = tf.keras.preprocessing.sequence.pad_sequences(y, padding='post')
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.characters) + 1)
        return X, y, np.array(input_lengths), np.array(label_lengths)
