# Phân tích notebook đầu vào

Nguồn: `original/Ollama_Gemma4_12B_original.ipynb`, bản sao nguyên byte của tệp người dùng gửi. Ô đánh số từ 0.

- Model được gọi thật trong mã: `gemma4:12b`; không thay bằng Qwen/Gemma 3.
- Metadata Colab và output nvidia-smi cho thấy T4; output lưu cũ không chứng minh runtime hiện tại.
- Ô 13 chứa `ollama pull gemma4:12b` trong Markdown, không chạy khi Run all.
- Ô 3 cài OpenAI SDK nhưng các lời gọi dùng Ollama SDK, nên dependency đó không cần cho pipeline này.
- Ô 8 bind mọi interface, không kiểm tra sẵn sàng bằng health polling; stdout/stderr PIPE không được đọc liên tục. Bản mới ghi log ra tệp, bind loopback và quản lý process riêng.
- Ô 19 comment lệnh tải ảnh; ô 21 vẫn đọc tệp. Runtime mới thiếu ảnh sẽ lỗi.
- Ô 22 đưa raw PCM float32 đã base64 hóa vào `content`. Đây là văn bản trong lời gọi đó, không phải payload audio được xác nhận đúng.
- Ô 23 lưu content rỗng, 4095 prompt token, 1 generated token, done_reason=length. Không được dùng làm bằng chứng transcription đúng hay sai của model.
- Không có train, bộ dữ liệu chia tập, bộ đo lặp, theo dõi đỉnh tài nguyên hoặc đối chứng có kiểm soát trong notebook gốc.

## Điều được giữ

Model gemma4:12b, engine Ollama, Python trên Colab và T4 là thiết bị mục tiêu. Bản gốc giữ riêng để đối chiếu, không sửa đè.

## Điều được bổ sung

Tải/xác minh model; khóa phiên bản/digest; native API để lấy thống kê; streaming; benchmark có kiểm soát; sampler bên ngoài Ollama; predictor độ trễ; validation/test; logs và báo cáo.

## Điều không được tự nhận

Không gọi bản tag Ollama là FP16; không gọi train predictor là train Gemma; không gọi việc bật option là chứng minh backend đã thực thi; không giả số đo GPU hoặc tỷ lệ acceptance.
