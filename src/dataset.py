"""Deterministic synthetic engineering benchmark, not an external quality suite."""
import random

def build_dataset(quick=False):
    counts={"train":12 if quick else 24,"validation":8 if quick else 12,"test":8 if quick else 16}
    rows=[]
    for split,n in counts.items():
        seed={"train":123,"validation":456,"test":789}[split];rng=random.Random(seed)
        for i in range(n):
            group=i%4;cap=128 if i%2==0 else 256
            prefix=f"{split}_{i:03d}"
            if group==0:
                a=rng.randrange(11,98);b=rng.randrange(2,17);c=rng.randrange(1,30)
                prompt=f"Một câu lạc bộ mua {a} bộ linh kiện, mỗi bộ {b} nghìn đồng, phí giao hàng {c} nghìn đồng. Tổng bao nhiêu nghìn đồng? Chỉ xuất JSON đúng dạng {{\"total\": số}}."
                kind="quality";task="arithmetic";expected={"total":a*b+c}
            elif group==1:
                code=f"T{rng.randrange(10000,99999)}";room=rng.choice(["A301","B204","C105","D402"])
                lines=[f"Mục {j}: workshop mã W{rng.randrange(1000,9999)}, phòng Z{rng.randrange(100,999)}." for j in range(5+i%5)]
                lines.insert(rng.randrange(len(lines)),f"Workshop mục tiêu mã {code}, phòng {room}.")
                prompt="Dữ liệu:\n"+"\n".join(lines)+f"\nChỉ xuất JSON {{\"room\": \"...\"}} cho workshop {code}."
                kind="quality";task="retrieval";expected={"room":room}
            elif group==2:
                topics=["vòng lặp Python","biến và kiểu dữ liệu","hàm trong Python","list và dictionary","đọc tệp văn bản","debug chương trình"]
                topic=topics[(i//4+seed)%len(topics)]
                prompt=f"Viết tài liệu thật dài về {topic} cho sinh viên mới. Lấy ví dụ quản lý lớp có {seed+i} sinh viên. Đánh số ít nhất 80 ý, mỗi ý gồm một ví dụ cụ thể và lời giải thích. Bắt đầu ngay từ ý 1, tiếp tục liên tục, không viết phần kết."
                kind="perf";task="explanation";expected=None
            else:
                facts=[f"Sản phẩm P{j}: giá {rng.randrange(10,99)} nghìn đồng, tồn kho {rng.randrange(10,99)}." for j in range(8+i%8)]
                prompt="Danh sách giả định:\n"+"\n".join(facts)+"\nViết tối thiểu 80 đoạn tư vấn quản lý kho. Mỗi đoạn giải thích một tình huống, dẫn một chi tiết từ danh sách. Viết liên tục, bắt đầu đoạn 1."
                kind="perf";task="long_input";expected=None
            rows.append(dict(id=prefix,split=split,prompt=prompt,kind=kind,task=task,cap=cap,expected=expected))
    return rows
