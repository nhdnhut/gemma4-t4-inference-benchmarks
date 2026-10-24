# Kiểm chứng gói bàn giao

Ngày: 13/10/2026.

## Đã kiểm tra

- 15 kiểm thử CPU/mock: PASS. Gồm thống kê stream, nanosecond, stream thiếu done, thiếu metrics, phân biệt content/thinking, chấm JSON, chặn prompt quá dài, kiểm soát draft, chia dữ liệu, bootstrap ghép cặp, không đưa test vào train, train/lưu predictor bằng dữ liệu mock, lọc quá ngân sách, khóa digest, lưu lỗi trong ma trận và loại cặp thiếu độ dài.
- Ba notebook: cú pháp Python từng ô PASS, cell ID duy nhất, outputs rỗng và execution_count=None.
- Mã nhúng trong cả ba notebook khớp chính xác thư mục src tại thời điểm đóng gói.
- PDF 26 trang đã render; kiểm tra tổng quan toàn bộ trang và xem kỹ trang bìa, bảng benchmark, checklist và nguồn. Không còn trang tràn một dòng.
- Notebook đầu vào được giữ nguyên byte trong original/.

## Môi trường kiểm thử logic

Python 3.12.14; scikit-learn 1.8.0; NumPy 2.3.5; pandas 2.2.3; joblib 1.5.3. Kiểm thử dùng unittest tiêu chuẩn.

## Chưa kiểm chứng

Không chạy Ollama/Gemma thật trên T4 tại môi trường bàn giao. Chưa chạy cài đặt Colab, chưa xác nhận mọi option có hiệu lực trên CUDA/backend cụ thể, chưa train predictor bằng dữ liệu T4, chưa có tốc độ/VRAM/chất lượng thực tế. Các kiểm thử mock không xác nhận khả năng tăng tốc của model.

Trong Colab cần kiểm tra model_show, logs, raw, số token, VRAM, điểm test và so với best-fixed. Chỉ báo hiệu quả trong phạm vi dữ liệu đã đo. Gói không hứa trước mức tăng tốc hoặc mức giảm bộ nhớ.
