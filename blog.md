# 🚀 [DevLog] ĐỌC VỊ KHÁCH HÀNG: CÁCH CHÚNG TÔI DÙNG AI & NLP ĐỂ "BÓC TÁCH" HÀNG NGÀN ĐÁNH GIÁ KHÁCH SẠN

**Tác giả:** Nhóm Data Mining 
**Team:** Kiều Quang Trường (Leader), Võ Minh Quân, Tô Vi Đức

---

## 1. Mở đầu: Lời nguyền của những "Cơn mưa sao"

Bạn đã bao giờ tự hỏi: Tại sao một khách sạn hạng sang 4 sao lại bất ngờ nhận bão 1 sao chỉ vì một chiếc vòi hoa sen hỏng? Hay có những người chấm trọn vẹn 5 sao, nhưng dòng đầu tiên trong bình luận lại là lời chê trách gay gắt thái độ của nhân viên lễ tân?

Đó chính là góc khuất của điểm số. **Điểm tổng (Rating) là một con số vô hồn và không bao giờ nói lên toàn bộ câu chuyện.**

Sau khi đã quen thuộc với các phân tích giỏ hàng hay dữ liệu dạng bảng truyền thống, nhóm chúng tôi quyết định nâng độ khó lên một bậc: Khai phá dữ liệu phi cấu trúc (*Unstructured Text Data*). Với bộ dữ liệu **Datafiniti Hotel Reviews** chứa hàng chục ngàn đánh giá thực tế, mục tiêu của chúng tôi là: *"Dạy máy tính cách thấu hiểu ngôn ngữ con người, tự động bóc tách khía cạnh khen/chê và dự đoán điểm số."*

## 2. Thử thách kỹ thuật: Xử lý Big Data trên "Cỗ máy thời gian"

Hành trình xử lý ngôn ngữ tự nhiên (NLP) chưa bao giờ là dễ dàng. Ngay khi bắt tay vào vector hóa văn bản, chúng tôi đã va phải bức tường lớn: Kích thước ma trận từ vựng phình to ra hàng chục ngàn cột, ngốn sạch RAM và làm chiếc MacBook Pro đời 2014 của nhóm gần như "đình công" hoàn toàn.

Thay vì bỏ cuộc hay thuê server Cloud đắt đỏ, team kỹ thuật đã cấu trúc lại toàn bộ Pipeline để tối ưu hóa tài nguyên:

* **Deep Text Cleaning:** Thay vì chỉ xóa dấu câu, chúng tôi can thiệp bằng `WordNetLemmatizer` để đưa toàn bộ từ vựng về dạng nguyên thể (ví dụ: *running -> run*), đồng thời dọn sạch các từ vô nghĩa (stopwords).
* **Ma trận thưa (Sparse Matrix) & Giới hạn TF-IDF:** Bằng cách giới hạn `max_features=3000` kết hợp với chuẩn n-gram (1, 2) và nén dữ liệu dưới dạng ma trận thưa (`scipy.sparse`), dung lượng bộ nhớ được giải phóng đến 80% mà vẫn giữ lại được những đặc trưng cốt lõi nhất.
* **Kiến trúc Module hóa:** Chấm dứt tình trạng "Jupyter Notebook Spaghetti". Toàn bộ logic được chia nhỏ thành các class (`TextCleaner`, `FeatureBuilder`, `SentimentClassifier`) trong thư mục `src/`, giúp việc debug và tái lập (reproduce) trở nên dễ dàng chỉ với một dòng lệnh duy nhất.

## 3. Bức tranh toàn cảnh: Tự động gom cụm chủ đề bằng AI

Không sử dụng các bộ từ điển gán sẵn tốn thời gian, chúng tôi để AI tự lên tiếng. Thông qua thuật toán **K-Means Clustering** chạy trên không gian vector TF-IDF, thuật toán đã tự động nhóm hàng ngàn bài đánh giá thành các "Cụm chủ đề" (*Topic Clusters*) riêng biệt:

* 🛏️ **Cụm "Giấc ngủ vương giả" (Room & Comfort):** Nơi hội tụ các từ khóa về nệm êm (*bed, mattress*), phòng sạch (*clean, spotless*), không gian yên tĩnh.
* 🛎️ **Cụm "Điểm chạm đầu tiên" (Service & Reception):** Xoay quanh thái độ nhân viên (*staff, friendly*), tốc độ nhận phòng (*check-in, desk*).
* 🏖️ **Cụm "Tiện ích ngoại khu" (Location):** Tập trung vào trải nghiệm đi bộ, vị trí gần biển hay trung tâm thương mại.

## 4. Insight đắt giá: "Bắt bài" tâm lý qua Toán học

Bằng việc kết hợp thuật toán **Apriori** (Luật kết hợp) và **Supervised Classification** (Phân lớp), bạn Võ Minh Quân đã bóc tách được 2 quy luật tâm lý cực kỳ thú vị được ẩn giấu trong dữ liệu:

### a. Hiệu ứng "Giận cá chém thớt" (Dựa trên chỉ số Lift & Confidence)
Thuật toán Apriori chỉ ra một liên kết mạnh mẽ: **(Khía cạnh: Lễ tân) → (Cảm xúc: Tiêu cực)**. 
> **Giải mã:** Khi khách hàng phàn nàn về thái độ lễ tân khi check-in, họ thường có xu hướng "soi" rất kỹ và chê luôn cả vấn đề vệ sinh phòng tắm hoặc bữa sáng. Đây là hiệu ứng tâm lý lây lan (*Halo Effect ngược*) – một trải nghiệm xấu ngay lúc bước vào cửa sẽ phá hỏng toàn bộ cái nhìn về dịch vụ phía sau.

### b. Cạm bẫy "Review Đa Chủ Đề"
Khi soi vào *Confusion Matrix* của mô hình phân lớp mạnh nhất (Linear SVC với F1-Macro xấp xỉ 0.8), chúng tôi phát hiện AI thường xuyên bị "lú" ở một kịch bản: Khách hàng viết một bài review dài 500 chữ, mở đầu khen nức nở vị trí khách sạn (Positive), nhưng đoạn cuối lại chê bai gay gắt đồ ăn (Negative). Máy tính bị nhiễu bởi các mảng từ vựng trái ngược nhau, dẫn đến phân loại sai lệch.

## 5. Lời giải cho bài toán chi phí: Học Bán Giám Sát (Semi-supervised Learning)

Trong thực tế doanh nghiệp, việc thuê người ngồi đọc và gán nhãn hàng triệu bình luận là bất khả thi về mặt chi phí. Chúng tôi đã giả lập bài toán này bằng cách **chỉ giữ lại 10% dữ liệu có nhãn** và áp dụng kỹ thuật Self-Training.

* 📈 **Kết quả bất ngờ:** Đường cong học tập (*Learning Curve*) chứng minh rằng, chỉ với 10% dữ liệu mồi, mô hình vẫn tự học và duy trì được mức F1-Score trên 0.7 – một con số quá hời so với chi phí bỏ ra.
* ⚠️ **Phân tích rủi ro:** Tuy nhiên, phương pháp này có một "tử huyệt" là các review siêu ngắn (dưới 50 ký tự). Những bình luận kiểu *"Not bad"*, *"OK"* do thiếu ngữ cảnh nên thường bị gán nhãn giả (*pseudo-label*) sai hoàn toàn, làm nhiễu tập huấn luyện.

## 6. Từ Dữ liệu đến Chiến lược kinh doanh (Actionable Insights)

Dữ liệu sẽ chỉ là rác nếu không tạo ra giá trị kinh doanh. Nhóm đề xuất 3 chiến lược hành động ngay lập tức cho các nhà quản lý:

1. 🥇 **Chiến lược "First-Touch":** Đầu tư tối đa cho khâu đào tạo Lễ tân. Lễ tân thân thiện và check-in nhanh là tấm khiên vững chắc nhất để khách hàng "chín bỏ làm mười" cho các lỗi nhỏ lặt vặt trong phòng, dập tắt hiệu ứng lây lan tiêu cực.
2. 🛡️ **Lọc nhiễu Hệ thống Feedback:** Cập nhật UI/UX trên App đặt phòng, thiết lập yêu cầu nhập tối thiểu 50 ký tự khi đánh giá. Điều này giúp loại bỏ các review vô nghĩa, tiết kiệm chi phí làm sạch dữ liệu và tăng độ chính xác cho hệ thống AI phân tích sau này.
3. 🎯 **Quản trị KPI theo khía cạnh (Aspect-based Management):** Từ bỏ việc đánh giá hiệu quả chi nhánh dựa trên điểm trung bình (Rating). Hệ thống AI của chúng tôi cho phép bóc tách: Phân hệ buồng phòng nhận điểm bao nhiêu, nhà hàng điểm bao nhiêu. Từ đó thưởng/phạt đúng người, tối ưu hóa dòng tiền hoạt động.

## 7. Kết luận

Đằng sau hàng triệu dòng chữ đánh giá lộn xộn là những quy luật vận hành và tâm lý học cực kỳ chặt chẽ. Xây dựng một Pipeline Data Mining không chỉ là việc gọi các hàm trong thư viện `scikit-learn`, mà là nghệ thuật kết hợp giữa tối ưu hệ thống máy tính và tư duy nhạy bén của người làm kinh doanh.

---
