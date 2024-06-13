# HandwritingRecognition
Handwriting Recognition using Machine Learning

Hướng dẫn cài đặt

1. Các gói cài đặt cần thiết cho dự án
pip install -r requirements.txt

2. Tiền xử lý hình ảnh
Đặt biến input_dir cho thư mục chứa ảnh cần xử lý
Đặt biến output_dir cho thư mục đầu ra
python3 src/preprocess.py

3. Huấn luyện mô hình
python3 src/scrnn.py --train {data-folder}  --label {nhãn.json}

4. Để predict
python3 src/predict.py --model {thư mục model} --data {thư mục chứa ảnh cần predict}
