[BLOG] ĐỌC VỊ KHÁCH HÀNG: CÁCH CHÚNG TÔI DÙNG AI & NLP ĐỂ "BÓC TÁCH" HÀNG NGÀN ĐÁNH GIÁ KHÁCH SẠN

Tác giả: Nhóm 8 Data Mining 

1. Mở đầu: Không chỉ là chuyện "Đánh giá mấy sao"
Bạn đã bao giờ tự hỏi: Tại sao một khách sạn hạng sang 4 sao lại bất ngờ nhận bão 1 sao chỉ vì một chiếc vòi hoa sen hỏng? Hay tại sao người ta có thể chấm 5 sao nhưng trong phần bình luận lại phàn nàn chê trách thái độ của lễ tân?

Đó chính là góc khuất của điểm số. Điểm tổng (Rating) không bao giờ nói lên toàn bộ câu chuyện.

Trong dự án kết thúc môn học lần này, nhóm chúng tôi đã quyết định thử sức "đào sâu" vào bộ dữ liệu Datafiniti Hotel Reviews (với hàng chục ngàn đánh giá thực tế từ khách du lịch) để tìm câu trả lời cho bài toán: "Khách hàng thực sự đang khen/chê điều gì, và làm sao để máy tính tự động hiểu được cảm xúc của họ?"

2. Thử thách kỹ thuật: Xử lý ngôn ngữ tự nhiên trên "phần cứng có hạn"
Hành trình nào cũng có chông gai. Ngay khi bắt tay vào xử lý dữ liệu văn bản (Text Data), team chúng tôi đã va phải một bức tường lớn: Kích thước ma trận từ vựng quá khủng khiếp khiến quá trình huấn luyện mô hình NLP trở nên ì ạch, thậm chí treo máy trên chiếc máy tính Mac dùng chip Intel.

Tuy nhiên, dưới sự dẫn dắt của team kỹ thuật, nhóm đã tìm ra giải pháp tối ưu đường ống (pipeline):

Làm sạch sâu (Deep Cleaning): Can thiệp bằng NLTK Lemmatizer để đưa từ về nguyên thể, dọn dẹp triệt để các từ vô nghĩa (stopwords).

Tối ưu hóa TF-IDF: Giới hạn max_features=3000 để giữ lại những đặc trưng cốt lõi nhất, tiết kiệm hàng chục GB RAM.

Vượt rào "Thiếu nhãn": Thay vì huấn luyện mô hình tốn kém, chúng tôi áp dụng Học bán giám sát (Semi-supervised Learning - Self Training). Chứng minh được rằng: Chỉ cần 10% dữ liệu có nhãn, máy vẫn có thể học và dự đoán tốt với điểm F1-Macro xấp xỉ 0.8!

3. Bức tranh toàn cảnh: Bản đồ chủ đề dịch vụ
Sau khi thuật toán Phân cụm (K-means) chạy xong, điều thú vị nhất đã hiện ra. Thay vì một đống chữ lộn xộn, chúng tôi đã gom nhóm thành công các "Cụm chủ đề" (Topic Clusters) tách biệt nhau:

Cụm "Giấc ngủ vương giả" (Room & Bed): Nơi hội tụ các từ khóa về nệm êm, phòng sạch, không gian yên tĩnh.

Cụm "Ấn tượng đầu tiên" (Service & Reception): Xoay quanh thái độ nhân viên, tốc độ check-in.

Cụm "Bữa sáng vội vã" (Food): Phân tích riêng về trải nghiệm ẩm thực tại khách sạn.

(Hình 1: Đám mây từ vựng - WordCloud cho từng cụm chủ đề khách sạn)

4. Insight đắt giá: "Bắt bài" cảm xúc qua Luật kết hợp và Phân tích lỗi
Dựa vào số liệu từ thuật toán Apriori và Classification, chúng tôi đã tìm ra 2 quy luật tâm lý cực kỳ thú vị:

a. Hiệu ứng "Kéo theo" (Dựa trên Luật kết hợp Apriori)
Chúng tôi phát hiện ra cặp liên kết: (Khía cạnh: Lễ tân) & (Khía cạnh: Phòng tắm) -> Cảm xúc: Tiêu cực (Negative).

Giải mã: Khi khách hàng phàn nàn về thái độ lễ tân, họ thường có xu hướng "soi" rất kỹ và phàn nàn luôn cả vấn đề vệ sinh phòng tắm. Đây là hiệu ứng tâm lý lây lan, một trải nghiệm xấu lúc check-in sẽ phá hỏng toàn bộ cái nhìn về căn phòng.

b. Cái bẫy "Review Đa Chủ Đề" và "Review Ngắn"
Khi soi vào Confusion Matrix (Ma trận nhầm lẫn), chúng tôi nhận thấy AI thường dự đoán sai ở 2 trường hợp:

Khách hàng "quay xe": Những đoạn review siêu dài, mở đầu thì khen nức nở vị trí khách sạn (Positive), nhưng chốt lại bằng việc chê bai gay gắt đồ ăn (Negative). Máy tính bị "lú" vì đan xen quá nhiều từ ngữ trái ngược.

Cạm bẫy Pseudo-label: Ở mô hình bán giám sát, những review quá ngắn (dưới 50 ký tự) như "ok", "not bad" thường bị gán nhãn giả sai lệch vì thiếu ngữ cảnh trầm trọng.

5. Từ Dữ liệu đến Chiến lược kinh doanh
Không để dữ liệu nằm trên giấy, nhóm đề xuất 3 chiến lược hành động ngay lập tức (Actionable Insights) cho các nhà quản lý khách sạn:

Chiến lược 1: "Điểm chạm đầu tiên" (First-Touch Optimization)

Hành động: Tối ưu hóa tối đa quy trình Check-in và dọn dẹp vệ sinh buồng phòng.

Lợi ích: Dập tắt ngay hiệu ứng "Kéo theo" đã phân tích ở trên. Lễ tân thân thiện sẽ làm khách hàng dễ tính hơn với các lỗi nhỏ trong phòng.

Chiến lược 2: "Lọc nhiễu" Hệ thống Feedback

Hành động: Trên App/Website đặt phòng, thiết lập ngưỡng yêu cầu nhập tối thiểu 50 ký tự khi đánh giá để tránh các review vô nghĩa (như "tốt", "ok"), giúp hệ thống AI phân loại tự động chính xác hơn.

Chiến lược 3: Quản trị theo khía cạnh (Aspect-based Management)

Hành động: Thay vì trả lương/thưởng dựa trên số sao (rating) tổng, hãy dùng AI bóc tách: Phân hệ buồng phòng nhận điểm bao nhiêu, phân hệ nhà hàng nhận điểm bao nhiêu. Từ đó thưởng/phạt đúng người, đúng tội.

6. Kết luận
Dự án này đã chứng minh rằng: Đằng sau hàng triệu dòng chữ đánh giá lộn xộn là những quy luật vận hành cực kỳ chặt chẽ.

Việc ứng dụng Khai phá dữ liệu văn bản (Text Mining) và NLP không chỉ giúp chúng ta hiểu khách hàng đang nghĩ gì, mà còn là chìa khóa để tối ưu hóa nguồn lực, cá nhân hóa dịch vụ và nâng tầm trải nghiệm trong ngành công nghiệp không khói.

Thành viên thực hiện:
👨‍💻 [Kiều Quang Trường]: Thiết kế kiến trúc Pipeline, Tối ưu thuật toán & Tiền xử lý NLP.

📊 [Võ Minh Quân]: Phân tích luật kết hợp (Association Rules), Insight & Chiến lược kinh doanh.

✍️ [Tô Vi Đức]: Huấn luyện mô hình Bán giám sát (Semi-supervised) & Trực quan hóa dữ liệu.
