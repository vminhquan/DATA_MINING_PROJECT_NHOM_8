DATA MINING PROJECT - ĐỀ TÀI 11: PHÂN TÍCH ĐÁNH GIÁ KHÁCH SẠN & CHỦ ĐỀ DỊCH VỤ

Dự án này thực hiện quy trình khai phá dữ liệu toàn diện trên tập dữ liệu đánh giá khách sạn (Hotel Reviews) từ Datafiniti. Mục tiêu là trích xuất thông tin hữu ích về khía cạnh dịch vụ, phân cụm chủ đề, và xây dựng các mô hình dự đoán sắc thái cũng như điểm số đánh giá.

1. Tổng quan Kiến trúc Dự án (Project Structure)

Dự án được tổ chức theo chuẩn module hóa, phân tách rõ ràng giữa dữ liệu, mã nguồn xử lý, và báo cáo kết quả .

configs/: Chứa các file cấu hình tham số.

data/:

raw/: Thư mục chứa dữ liệu gốc (không commit lên Git).

processed/: Thư mục chứa dữ liệu sau khi tiền xử lý và ma trận đặc trưng.

notebooks/: Chứa các file Jupyter Notebook trình bày báo cáo phân tích từ 01 đến 05 .

src/: Thư mục chứa mã nguồn cốt lõi (Core logic).

data/: Các module đọc và làm sạch dữ liệu văn bản .

features/: Module xây dựng đặc trưng (TF-IDF) .

mining/: Module thuật toán Khai phá phân cụm (Clustering) và Luật kết hợp (Association Rules) .

models/: Module huấn luyện mô hình Phân lớp, Bán giám sát và Hồi quy .

scripts/: Chứa các file script chạy tự động toàn bộ luồng pipeline .

outputs/: Nơi lưu trữ biểu đồ, bảng kết quả và mô hình đã huấn luyện .

2. Các Bài toán Khai phá được Giải quyết

Dự án giải quyết 5 bài toán cốt lõi bám sát yêu cầu đề tài 11:

Phân cụm (Clustering): Sử dụng K-means trên ma trận TF-IDF để nhóm các bài đánh giá thành các cụm chủ đề (topic clusters), đặt tên cụm và trích xuất review đại diện.

Luật kết hợp (Association Rules): Sử dụng thuật toán Apriori. Rời rạc hóa các từ khóa theo khía cạnh dịch vụ (Aspect) để tìm top các luật dịch vụ đi kèm với lời khen hoặc phàn nàn.

Phân lớp (Supervised Classification): Dự đoán cảm xúc (Sentiment/Aspect) bằng các mô hình Linear. Đánh giá bằng metric F1-macro và tập trung phân tích lỗi ở các review đa chủ đề.

Bán giám sát (Semi-supervised Learning): Giả lập thiếu nhãn (chỉ dùng 5-30% nhãn). Sử dụng Self-training, vẽ learning curve và phân tích các trường hợp gán nhãn giả sai (pseudo-label sai) ở review ngắn.

Hồi quy (Regression): Dự đoán điểm rating của khách hàng. Xây dựng baseline và mô hình Random Forest, đánh giá bằng MAE và RMSE.

3. Cài đặt và Yêu cầu Môi trường

Để tái lập (reproduce) dự án này trên máy tính của bạn, vui lòng thực hiện các bước sau trong Terminal:

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

4. Hướng dẫn Chạy Dự án (How to Run)
Bước 1: Chuẩn bị Dữ liệu
Tải tập dữ liệu Datafiniti_Hotel_Reviews.csv từ Kaggle.
Đặt file dữ liệu đã tải (ví dụ: 7282_1.csv) vào thư mục data/raw/ .
+1

Bước 2: Chạy Tự động Pipeline
Bạn có thể chạy toàn bộ quy trình từ làm sạch dữ liệu đến huấn luyện mô hình chỉ với một lệnh duy nhất từ thư mục gốc:

python scripts/run_pipeline.py

Kết quả in ra Terminal sẽ bao gồm các thông số MAE, RMSE và F1-Macro của toàn bộ các mô hình.

Bước 3: Xem Báo cáo Phân tích
Mở các file trong thư mục notebooks/ để xem trực quan hóa dữ liệu (EDA), biểu đồ Learning Curve, Confusion Matrix và các insight phân tích lỗi chuyên sâu:

jupyter notebook

5. Insight Nổi bật (Actionable Insights)

Mô hình phân lớp dễ dự đoán sai đối với các đánh giá có độ dài lớn do người dùng thường đan xen cả khen và chê trong cùng một câu.

Review quá ngắn (dưới 50 ký tự) không phù hợp cho thuật toán Self-training vì thiếu ngữ cảnh trầm trọng, dẫn đến lỗi gán nhãn giả.
