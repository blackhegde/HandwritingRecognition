import os
from preprocess import preprocess_image
from load_data import load_images_and_labels, DataGenerator
from crnn import build_model
from keras.optimizers import Adam

    
# Định nghĩa đường dẫn và các siêu tham số
root_path = os.getcwd()
train_data_dir = os.path.join(root_path, "data", "raw", "PrivateTest")
val_data_dir = os.path.join(root_path, "data", "raw", "DataSamples1")
processed_val_data_dir = os.path.join(root_path, "data", "processed", "DataSamples1")
processed_train_data_dir = os.path.join(root_path, "data", "processed", "PrivateTest")
input_shape = (128, 64, 1)  # Kích thước đầu vào ảnh
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂĂÀÁẢẠÃẤẦẨẬẪẮẰẲẶẴÊẸÈÉẺẼẾỀỂỆỄỊÌÍỈĨÔỌÒÓỎÕỐỒỔỘỖỜỚỞỢỠỤÙÚỦŨỨỪỬỰỮÝỲỴỶỸĐàáảạãấầẩậẫắằẳặẵẹêẻèéẽếềểễệỉìíĩịơọòóỏõốồổộỗờớởợỡừùúủũụứừửựữỵỳỹýỷỹđâăôịưự,./-#'() "
num_classes = len(characters)

# Tiền xử lý ảnh
preprocess_image(train_data_dir, processed_train_data_dir)
preprocess_image(val_data_dir, processed_val_data_dir)

# Tải dữ liệu
train_data = load_images_and_labels(processed_train_data_dir)
val_data = load_images_and_labels(processed_val_data_dir)
img_size = (128, 64, 1)
train_generator = DataGenerator(train_data, characters, batch_size=32, input_shape=input_shape)
val_generator = DataGenerator(val_data, characters, batch_size=32, input_shape=input_shape)


# Xây dựng và huấn luyện mô hình CRNN
input_shape = (img_size[0], img_size[1], 1)  # (64, 128, 1)
num_classes = len(characters) + 1
model = build_model(input_shape, num_classes)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam())

# Hiển thị cấu trúc mô hình
model.summary()
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Lưu mô hình sau khi huấn luyện
model.save('crnn_model.h5')