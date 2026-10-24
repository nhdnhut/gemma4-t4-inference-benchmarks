"""Rebuild self-contained Colab notebooks from the reviewed shared source."""
from pathlib import Path
import json,hashlib,base64,zlib,textwrap

ROOT=Path(__file__).resolve().parent
def md(s):return dict(cell_type="markdown",metadata={},source=textwrap.dedent(s).strip()+"\n")
def code(s,title=None):
    meta={"cellView":"form"} if title else {}
    return dict(cell_type="code",metadata=meta,source=(f"# @title {title}\n" if title else "")+textwrap.dedent(s).strip()+"\n",outputs=[],execution_count=None)

sources={p.name:p.read_text() for p in (ROOT/"src").glob("*.py")}
packed=zlib.compress(json.dumps(sources).encode(),9)
encoded=base64.b64encode(packed).decode()
bootstrap=code(f'''
import base64, zlib, json, hashlib, sys
from pathlib import Path
_compressed = base64.b64decode({encoded!r})
assert hashlib.sha256(_compressed).hexdigest() == {hashlib.sha256(packed).hexdigest()!r}
CODE_DIR = Path("/content/gemma4_t4_code")
CODE_DIR.mkdir(parents=True, exist_ok=True)
for name, source in json.loads(zlib.decompress(_compressed)).items():
    (CODE_DIR / name).write_text(source, encoding="utf-8")
sys.path.insert(0, str(CODE_DIR))
for name in ["gemma_lab", "controller", "dataset"]:
    sys.modules.pop(name, None)
from gemma_lab import *
from dataset import build_dataset
print("Đã nạp mã dùng chung. Bản dễ đọc nằm trong src/ của gói ZIP.")
''',"Nạp mã dùng chung (tự chứa trong notebook)")

storage=code('''
from google.colab import drive
drive.mount('/content/drive')
EXPERIMENT_ID = "gemma4_t4_v1" # Giữ nguyên ở cả 3 notebook
QUICK = True                 # True: kiểm tra quy trình; False: đợt đo đầy đủ hơn
BUDGET_MIB = 13824            # 13.5 GiB; là ngân sách thí nghiệm, không phải số đã đo
ROOT = Path('/content/drive/MyDrive/Gemma4_T4_Lab') / EXPERIMENT_ID
ROOT.mkdir(parents=True, exist_ok=True)
SETTINGS = dict(quick=QUICK, budget_mib=BUDGET_MIB, model=MODEL, context=4096)
if (ROOT / 'settings.json').exists():
    assert load_json(ROOT / 'settings.json') == SETTINGS, 'Cấu hình đã đổi: tạo EXPERIMENT_ID mới rồi chạy từ 01.'
else:
    save_json(ROOT / 'settings.json', SETTINGS)
if (ROOT / 'dataset.json').exists():
    DATA = load_json(ROOT / 'dataset.json')
else:
    DATA = build_dataset(quick=QUICK)
    save_json(ROOT / 'dataset.json', DATA)
print(ROOT, '| chế độ', 'KIỂM TRA QUY TRÌNH' if QUICK else 'ĐO ĐẦY ĐỦ')
''',"Chọn nơi lưu và quy mô thí nghiệm")

install=code('''
import os, subprocess, urllib.request, shutil
locked = load_json(ROOT / 'experiment_lock.json') if (ROOT / 'experiment_lock.json').exists() else None
INSTALL_OLLAMA = not shutil.which('ollama')
if INSTALL_OLLAMA:
    subprocess.run(['apt-get','update','-qq'], check=True)
    subprocess.run(['apt-get','install','-y','zstd','pciutils'], check=True)
    installer = Path('/content/install_ollama.sh')
    urllib.request.urlretrieve('https://ollama.com/install.sh', installer)
    install_env = os.environ.copy()
    if locked:
        install_env['OLLAMA_VERSION'] = locked['ollama_version']
    subprocess.run(['sh',str(installer)],env=install_env,check=True)
print(subprocess.run(['ollama','--version'],capture_output=True,text=True).stdout)
HW = hardware()
print(json.dumps(HW, ensure_ascii=False, indent=2))
assert 'T4' in HW['gpu_name'], 'Đổi sang T4 để đúng phạm vi đồ án.'
assert HW['gpu_mib'] > BUDGET_MIB, 'Ngân sách phải nhỏ hơn VRAM thực; tạo thí nghiệm khác nếu đổi ngân sách.'
save_json(ROOT / 'hardware_latest.json', HW)
# Không tải trọng số nhiều GB vào Google Drive mặc định: Drive lưu kết quả, /content lưu model.
if 'manager' in globals():
    manager.stop()
manager = Server(ROOT / 'logs', model_dir='/content/ollama_models')
manager.start(BASELINE)
''',"Cài Ollama và kiểm tra GPU")

def common(number):
    return [md(f'''
    # {number} · Gemma 4 12B trên Colab T4
    **Giữ nguyên model `gemma4:12b` của notebook bạn gửi.**
    Chọn **Runtime → Change runtime type → T4 GPU**. Chạy các ô theo thứ tự.
    Cả ba notebook tự chứa cùng mã nguồn; không cần upload thêm thư mục `src`.
    `QUICK=True` chỉ thử quy trình. Dùng `False` và một EXPERIMENT_ID mới cho báo cáo.
    Không có kết quả T4 được điền sẵn. Khi Colab ngắt, dữ liệu đo đã ghi nằm trên Drive.
    ''') ,bootstrap,storage,md('''
    ## Chuẩn bị engine
    Notebook chạy Ollama ở `127.0.0.1:11435` trong máy Colab. Đây không phải máy cá nhân của bạn.
    Lần đầu tải engine và model cần Internet. Các lượt sinh sau đó dùng GPU trong runtime.
    Lưu phiên bản/digest để tránh hai notebook vô tình dùng hai mô hình khác nhau.
    ''') ,install,
    code(f'''LOCK = establish_lock(manager.api, ROOT, create={number.startswith('01')})
print(json.dumps(LOCK, ensure_ascii=False, indent=2))
print("Khả năng model:", load_json(ROOT / 'model_show.json').get('capabilities'))
''',"Xác minh và khóa phiên bản mô hình")]

n1=common("01 · Baseline")+[
md('''
## A. Đối chiếu mã gốc
Mã gốc chỉ cài engine và hỏi một câu. `ollama pull gemma4:12b` trong file gốc nằm ở ô Markdown,
nên Run all không thực hiện việc tải. Ô xác minh phía trên đã bổ sung thao tác tải có kiểm tra.
Thử nghiệm gốc dưới đây không giới hạn generation và có thể mất nhiều thời gian; mặc định bỏ qua.
Đây chỉ là demo tính năng, không dùng để công bố mức tăng tốc.
'''),code('''
RUN_ORIGINAL_DEMO = False
if RUN_ORIGINAL_DEMO:
    original = manager.api.json('/api/chat', dict(model=MODEL,
        messages=[dict(role='user',content='Why is the sky blue?')], stream=False))
    print(original['message']['content'])
'''),md('''
## B. Vì sao phải có baseline kiểm soát?
Giữ trọng số của tag gốc, Flash Attention tự động, cache f16, draft theo mặc định của model.
Đặt cùng context 4096, temperature 0, seed 42 và think=False cho mọi đối chứng.
Đây là **baseline cấu hình ban đầu có kiểm soát**, không phải Gemma FP16 chưa nén.
Kiểm tra `/api/show`, `/api/ps` và log; phiên bản mới có thể đã tối ưu sẵn.
'''),code('''
manager.start(BASELINE)
example = dict(id='cold_demo',split='demo',kind='quality',task='arithmetic',cap=128,
    prompt='Tính 12 nhân 3. Chỉ xuất JSON {"total": số}.', expected={'total':36})
cold = benchmark_one(manager, BASELINE, example, 'cold')
warm = benchmark_one(manager, BASELINE, example, 'warm_different')
save_json(ROOT / 'cold_warm_demo.json', {'cold':cold,'warm':warm,'ps':manager.api.json('/api/ps')})
for label,r in [('cold',cold),('warm',warm)]:
    print(label, {k:r.get(k) for k in ['content','load_s','wall_s','ttft_content_s','decode_tps_engine','peak_gpu_mib']})
''',"Đo ví dụ lần nạp đầu và khi model đã ở bộ nhớ"),md('''
## C. Đo baseline trên tập phát triển
Phép đo này dùng train/validation; test được dành cho notebook 03.
Mỗi câu trả lời lưu cùng số token thực, lý do dừng, thời gian và tài nguyên.
Bài tốc độ được yêu cầu viết dài; chỉ so độ dài cố định khi thực tế sinh đủ cap.
Các bài có đáp án JSON đo chất lượng với dừng tự nhiên, không ép viết cho đủ token.
'''),code('''
dev = [x for x in DATA if x['split'] in {'train','validation'}]
rows, run = run_matrix(manager,[BASELINE],dev,ROOT,'baseline',repeats=1 if QUICK else 2)
print(json.dumps(summarize(rows),ensure_ascii=False,indent=2))
save_json(run / 'summary.json',summarize(rows))
''',"Chạy baseline và lưu số liệu"),md('''
## Đã xong notebook 01
Đọc `model_show.json`, `experiment_lock.json`, `cold_warm_demo.json`, thư mục `baseline_.../`.
Nếu có OOM hoặc offload CPU, giữ nguyên bằng chứng; đừng gọi một lần chạy thất bại là baseline nhanh.
Tiếp tục notebook 02 với cùng EXPERIMENT_ID và QUICK. Model đã Q4 không có nghĩa cache cũng Q4.
'''),code("manager.stop()","Dừng server của notebook này")]

n2=common("02 · Train & Optimize")+[
md('''
## Train cái gì?
Ta huấn luyện **Random Forest dự đoán độ trễ** từ số đo T4. Trọng số Gemma giữ nguyên.
Input: độ dài prompt theo byte, nhóm tác vụ, giới hạn token đầu ra và cấu hình.
Target: log(thời gian trả lời khi model đã nạp). Không dùng số token đầu ra thực làm feature,
vì lúc người dùng gửi câu hỏi ta chưa biết đại lượng đó.
Fine-tuning Gemma là bài toán khác, không được thay bằng `ollama pull` hoặc đổi prompt.
'''),code('''
import subprocess, sys
packages=['scikit-learn==1.8.0','joblib==1.5.3']
subprocess.run([sys.executable,'-m','pip','install','-q',*packages],check=True)
save_json(ROOT / 'python_packages_train.json',dict(packages=packages))
(ROOT / 'pip_freeze_train.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)
from controller import train_selector
''',"Cài thư viện cho bộ dự đoán nhỏ"),md('''
## Các cấu hình ứng viên
B0 giữ mặc định engine/model nhưng kiểm soát yêu cầu sinh.
C1 thử Flash Attention; C2 thêm cache Q8; C3 thay batch xử lý prompt;
C4 tắt draft; C5 thử tối đa 6 token nháp. `draft_num_predict` là giới hạn, không phải tỷ lệ chấp nhận.
Backend có thể từ chối hoặc bỏ qua một tùy chọn. Đọc log để xác nhận trước khi quy lợi ích cho kỹ thuật đó.
T4 không cần cài gói Python `flash-attn`; ta sử dụng kernel có sẵn của engine Ollama.
'''),code('''
print(json.dumps([asdict(c) for c in CONFIGS],indent=2))
dev = [x for x in DATA if x['split'] in {'train','validation'}]
REPEATS = 1 if QUICK else 2
print('Số lượt generation đo:',len(CONFIGS)*len(dev)*REPEATS,'; chưa tính warm-up.')
rows, run = run_matrix(manager,CONFIGS,dev,ROOT,'tuning',repeats=REPEATS)
save_json(run / 'summary.json',summarize(rows))
print(json.dumps(summarize(rows),ensure_ascii=False,indent=2))
''',"Thu dữ liệu thật trên T4"),md('''
## Huấn luyện, lọc cấu hình, khóa quyết định
Train chỉ dùng split=train; validation chọn độ sâu cây và kiểm tra chất lượng/bộ nhớ.
Không trộn các lần lặp của cùng prompt vào hai tập. Dữ liệu mặc định là bài kiểm thử kỹ thuật tổng hợp,
chưa đại diện cho tất cả câu hỏi học thuật hay tiếng Việt.
Bộ chọn so với best-fixed đo trên validation. Nếu dự đoán chọn kém, dùng best-fixed làm fallback.
Không có cấu hình đạt ràng buộc thì dừng và lưu lý do; không tự giảm context hoặc nới chất lượng.
'''),code('''
selection, audit = train_selector(rows,CONFIGS,ROOT / 'optimizer',budget_mib=BUDGET_MIB,quality_drop=0.05)
selection['experiment_lock_sha256'] = digest(LOCK)
selection['hardware_gpu_name'] = HW['gpu_name']
save_json(ROOT / 'optimizer' / 'selection.json',selection)
print(json.dumps(selection,ensure_ascii=False,indent=2))
print(json.dumps(audit,ensure_ascii=False,indent=2))
''',"Train bộ dự đoán và chọn cấu hình"),md('''
## Đọc kết quả
`validation_mae_s`: sai số dự đoán tính bằng giây, càng thấp càng tốt trong cùng bộ dữ liệu.
`selected_config`: cấu hình cố định cho phiên demo. `best_fixed_config`: đối chứng quan trọng.
`latency_predictor.joblib`: mô hình nhỏ đã train; **không phải checkpoint Gemma**.
Không đổi cache hay restart engine giữa mỗi token; chi phí nạp được đo riêng và cần khấu hao trên số câu hỏi.
Nếu QUICK=True, chạy cả quy trình để hiểu trước; chuyển sang thí nghiệm mới QUICK=False trước khi báo cáo.
'''),code("manager.stop()","Dừng server của notebook này")]

n3=common("03 · Evaluate & Demo")+[
md('''
## Khóa cấu hình trước khi mở test
Chạy lại B0, best-fixed và cấu hình hệ thống chọn trong **cùng runtime T4**.
Nếu hai cấu hình trùng nhau thì chỉ đo một lần. Không tune lại theo điểm test.
Nếu không cải thiện, kết luận không cải thiện; không chọn một vài prompt thuận lợi để báo cáo.
'''),code('''
selection=load_json(ROOT / 'optimizer' / 'selection.json')
assert selection['experiment_lock_sha256']==digest(LOCK),'Model/engine đã khác lúc train.'
assert selection['hardware_gpu_name']==HW['gpu_name'],'GPU đã đổi; cần hiệu chỉnh lại.'
selected=Config(**selection['selected_config'])
best_fixed=Config(**selection['best_fixed_config'])
EVAL_CONFIGS=list({c.name:c for c in [BASELINE,best_fixed,selected]}.values())
save_json(ROOT / 'evaluation_policy_frozen.json',dict(selection=selection,configs=[asdict(c) for c in EVAL_CONFIGS]))
test=[x for x in DATA if x['split']=='test']
test_rows, test_run=run_matrix(manager,EVAL_CONFIGS,test,ROOT,'test',repeats=2 if QUICK else 3)
report=summarize(test_rows)
comparisons=[paired_comparison(test_rows,BASELINE.name,c.name) for c in EVAL_CONFIGS if c.name!=BASELINE.name]
if selected.name!=best_fixed.name:
    comparisons.append(paired_comparison(test_rows,best_fixed.name,selected.name))
save_json(test_run / 'summary.json',report)
export_csv(test_run / 'comparison.csv',report)
grouped=[dict(group=kind,**row) for kind in ['quality','perf'] for row in summarize([r for r in test_rows if r['kind']==kind])]
export_csv(test_run / 'comparison_by_kind.csv',grouped)
save_json(test_run / 'paired_speedup.json',comparisons)
print(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(comparisons,ensure_ascii=False,indent=2))
''',"Đánh giá test cùng phiên T4"),md('''
## Báo cáo và biểu đồ
Biểu đồ chỉ vẽ số đo vừa thu. Giữ cả cấu hình lỗi trong dữ liệu thô.
Decode tokens/s do engine cung cấp khác throughput từ đầu đến cuối và khác TTFT.
Chỉ các prompt tốc độ đủ cùng token cap mới vào thống kê độ dài cố định.
'''),code('''
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','matplotlib>=3.8,<4','pandas>=2.2,<3'],check=True)
import matplotlib.pyplot as plt
import pandas as pd
display(pd.DataFrame(report))
fig,axes=plt.subplots(1,2,figsize=(12,4))
names=[x['config'] for x in report]
for ax,key,title in [(axes[0],'fixed_decode_tps_median','Decode: fixed-length rows (token/s)'),
                     (axes[1],'peak_gpu_mib_median','Median of sampled per-request VRAM peaks (MiB)')]:
    vals=[x.get(key) for x in report]
    ax.barh(names,[v if v is not None else 0 for v in vals]);ax.set_title(title)
    for i,v in enumerate(vals):ax.text(v or 0,i,' N/A' if v is None else f' {v:.1f}',va='center')
fig.tight_layout();fig.savefig(test_run / 'speed_memory.png',dpi=180);plt.show()
print('Số đo p95 trên ít prompt chưa đáng tin như benchmark lớn. Bootstrap ghép cặp nằm trong paired_speedup.json.')
''',"Xem kết quả và xuất biểu đồ"),md('''
## Kiểm tra nghiệm thu tự động tối thiểu
Đây là kiểm tra kỹ thuật, không thay đánh giá thủ công câu trả lời thực tế.
Không đổi cấu hình đã công bố trong đánh giá test. Demo dùng B0 nếu bản được chọn vi phạm ràng buộc.
Trường hợp này phải ghi rõ bản tối ưu chưa đạt, không xóa test để chọn lại.
'''),code('''
chosen_rows=[r for r in test_rows if r['config']==selected.name]
base_rows=[r for r in test_rows if r['config']==BASELINE.name]
def qscore(rs):
    vals=[r.get('score',0) if r['status']=='ok' else 0 for r in rs if r['kind']=='quality']
    return sum(vals)/len(vals) if vals else 0
passed=bool(chosen_rows) and all(r['status']=='ok' and r.get('peak_gpu_mib') is not None and
    r['peak_gpu_mib']<=BUDGET_MIB for r in chosen_rows)
passed=passed and qscore(chosen_rows)>=qscore(base_rows)-0.05
DEMO_CONFIG=selected if passed else BASELINE
save_json(test_run / 'technical_gate.json',dict(passed=passed,demo_config=asdict(DEMO_CONFIG),
    selected_score=qscore(chosen_rows),baseline_score=qscore(base_rows)))
print('Đạt kiểm tra kỹ thuật:',passed,'| Demo dùng:',DEMO_CONFIG.name)
''',"Kiểm tra ngân sách và chất lượng"),md('''
## Sử dụng hệ thống: hỏi đáp văn bản có streaming
Lần đầu của phiên gồm thời gian khởi động/nạp model; không nhầm thành tốc độ decoding chậm.
Demo này là từng câu độc lập để quản lý context rõ ràng. Với hội thoại nhiều lượt, phải tính cả lịch sử.
Server ở bên trong Colab; không mở cổng public, không cần API key của OpenAI.
'''),code('''
manager.start(DEMO_CONFIG)
PROMPT = 'Giải thích vòng lặp for trong Python cho người mới, kèm một ví dụ ngắn.'
demo_item=dict(id='user_demo',split='demo',kind='perf',task='explanation',cap=256,prompt=PROMPT)
demo_result=benchmark_one(manager,DEMO_CONFIG,demo_item,on_text=lambda s:print(s,end='',flush=True))
print(chr(10)*2, {k:demo_result.get(k) for k in ['wall_s','load_s','ttft_content_s','eval_count','decode_tps_engine','peak_gpu_mib']})
save_json(ROOT / 'last_demo.json',demo_result)
''',"Nhập câu hỏi của bạn"),md('''
## Ảnh và âm thanh trong file gốc
Benchmark chính đã chọn phạm vi text. Ảnh cần upload tệp thật và gửi qua trường `images`;
phải đo riêng chi phí xử lý ảnh. Không dùng chuỗi base64 trong `content` làm đầu vào âm thanh.
Xem phần phân tích file gốc trong cẩm nang trước khi mở rộng bài đo đa phương thức.
'''),code('''
import shutil
archive=shutil.make_archive('/content/'+EXPERIMENT_ID+'_results','zip',ROOT)
print('Gói kết quả:',archive)
# Tải khi cần: from google.colab import files; files.download(archive)
''',"Đóng gói kết quả chạy thật"),code("manager.stop()","Dừng server khi dùng xong")]

for name,cells in [('01_Baseline_Gemma4_T4.ipynb',n1),('02_Train_Optimize_Gemma4_T4.ipynb',n2),('03_Evaluate_Demo_Gemma4_T4.ipynb',n3)]:
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(provenance=[],gpuType='T4'),
        accelerator='GPU',kernelspec=dict(display_name='Python 3',name='python3'),language_info=dict(name='python')),cells=cells)
    for i,c in enumerate(cells):c['id']=hashlib.sha256((name+str(i)).encode()).hexdigest()[:12]
    (ROOT/'notebooks'/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    print(name,len(cells),'cells')
