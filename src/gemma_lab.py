"""Gemma 4 12B / Ollama lab. Native API, measured selection, no weight training.

All performance numbers are collected on the user's runtime. This module never
invents missing metrics. See docs/CAM_NANG.md for units and experimental limits.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, platform, random, signal, socket
import statistics, subprocess, threading, time, urllib.error, urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

MODEL = "gemma4:12b"
SYSTEM = "Trả lời đúng yêu cầu. Không thêm lời dẫn nếu đề chỉ yêu cầu kết quả."
BASE_URL = "http://127.0.0.1:11435"

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def read_jsonl(path):
    if not Path(path).exists(): return []
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]

def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())

def export_csv(path, rows):
    if not rows: return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in r.items()})

class API:
    def __init__(self, base_url=BASE_URL): self.base_url = base_url
    def request(self, path, body=None, timeout=900):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type":"application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}") from e
    def json(self, path, body=None, timeout=900):
        with self.request(path, body, timeout) as r: return json.load(r)
    def stream(self, path, body):
        with self.request(path, body) as r:
            for line in r:
                if line.strip():
                    row=json.loads(line)
                    if "error" in row: raise RuntimeError(row["error"])
                    yield row

@dataclass(frozen=True)
class Config:
    name: str
    flash: str = "auto"
    kv: str = "f16"
    batch: int = 512
    draft: int | None = None
    ctx: int = 4096
    def options(self, cap):
        d = dict(num_ctx=self.ctx, num_batch=self.batch, num_predict=cap,
                 temperature=0, seed=42, top_k=1, top_p=1.0, repeat_penalty=1.0)
        if self.draft is not None: d["draft_num_predict"] = self.draft
        return d

BASELINE = Config("B0_original_controlled")
CONFIGS = [BASELINE,
    Config("C1_flash", flash="1"),
    Config("C2_flash_kv8", flash="1", kv="q8_0"),
    Config("C3_kv8_batch256", flash="1", kv="q8_0", batch=256),
    Config("C4_kv8_no_draft", flash="1", kv="q8_0", draft=0),
    Config("C5_kv8_draft6", flash="1", kv="q8_0", draft=6)]

class Server:
    """Only owns the process it starts. Never pkill other Ollama processes."""
    def __init__(self, directory, model_dir="/content/ollama_models", base_url=BASE_URL):
        self.directory=Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.model_dir=str(model_dir); self.api=API(base_url)
        self.proc=None; self.log=None; self.config=None; self.log_path=None
    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try: self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL); self.proc.wait(timeout=5)
        if self.log: self.log.close()
        self.proc=None; self.config=None
    def start(self, cfg):
        self.stop()
        port=int(self.api.base_url.rsplit(":",1)[1])
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1",port))==0:
                raise RuntimeError(f"Cổng {port} đang được dùng. Chạy ô manager.stop() ở notebook cũ hoặc ngắt runtime cũ.")
        env=os.environ.copy()
        for key in list(env):
            if key.startswith("OLLAMA_"): env.pop(key)
        env.update(OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS=self.model_dir,
                   OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1",
                   OLLAMA_KV_CACHE_TYPE=cfg.kv, OLLAMA_DEBUG="1", OLLAMA_NO_CLOUD="1")
        if cfg.flash != "auto": env["OLLAMA_FLASH_ATTENTION"]=cfg.flash
        self.log_path=self.directory / f"{time.time_ns()}_{cfg.name}.log"
        self.log=self.log_path.open("wb")
        self.proc=subprocess.Popen(["ollama","serve"], env=env, stdout=self.log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            if self.proc.poll() is not None:
                raise RuntimeError(self.log_path.read_text(errors="replace")[-3000:])
            try:
                self.api.json("/api/version",timeout=1); self.config=cfg; return
            except Exception: time.sleep(0.5)
        self.stop(); raise RuntimeError("Ollama chưa sẵn sàng sau 60 giây; xem log.")
    def log_text(self):
        return self.log_path.read_text(errors="replace") if self.log_path else ""

def hardware():
    p=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version,uuid",
                      "--format=csv,noheader,nounits"],text=True,capture_output=True)
    if p.returncode: raise RuntimeError("Chọn Runtime > Change runtime type > T4 GPU, rồi chạy lại.")
    if len(p.stdout.strip().splitlines()) != 1: raise RuntimeError("Bài đo này yêu cầu đúng một GPU.")
    parts=[x.strip() for x in p.stdout.strip().split(",")]
    return dict(gpu_name=parts[0],gpu_mib=float(parts[1]),driver=parts[2],gpu_uuid=parts[3],
                python=platform.python_version(),platform=platform.platform(),cpu_count=os.cpu_count())

def establish_lock(api, root, create=False):
    root=Path(root); path=root/"experiment_lock.json"
    tags=api.json("/api/tags")["models"]
    found=next((x for x in tags if x.get("name")==MODEL or x.get("model")==MODEL),None)
    if not found:
        print("Đang tải",MODEL,"(có thể mất vài phút)...",flush=True)
        last=""
        for row in api.stream("/api/pull",dict(model=MODEL,stream=True)):
            state=row.get("status","")
            if state!=last: print(state,flush=True);last=state
        found=next(x for x in api.json("/api/tags")["models"] if x.get("name")==MODEL or x.get("model")==MODEL)
    show=api.json("/api/show",dict(model=MODEL))
    if show.get("remote_host") or show.get("remote_model"):
        raise RuntimeError("Phải dùng mô hình local; không dùng biến thể cloud.")
    current=dict(model=MODEL,model_digest=found["digest"],ollama_version=api.json("/api/version")["version"],
                 details=show.get("details",{}),parameters=show.get("parameters",""),
                 manifests=show.get("manifests",[]),template_sha256=digest(show.get("template","")),
                 dataset_sha256=digest(load_json(root/"dataset.json")))
    if path.exists():
        old=load_json(path)
        for key in ["model_digest","ollama_version","template_sha256","dataset_sha256"]:
            if old[key]!=current[key]:
                raise RuntimeError(f"{key} đã đổi. Khôi phục đúng model/engine hoặc tạo EXPERIMENT_ID mới và đo lại từ 01; không trộn kết quả.")
    elif create: save_json(path,current)
    else: raise RuntimeError("Chưa có experiment_lock.json. Chạy notebook 01 trước.")
    save_json(root/"model_show.json",show)
    save_json(root/"hardware_latest.json",hardware())
    return current

class Monitor:
    """Device-wide sampled VRAM; RSS sum of server process tree; host RAM used.
    Sampling is approximate, not an allocator peak. nvidia-smi overhead is shared
    by all profiles. Missing sensors remain None and invalidate memory selection.
    """
    def __init__(self,pid,interval=0.25):
        self.pid=pid; self.interval=interval; self.rows=[]; self.event=threading.Event()
    def sample(self):
        row={"at":time.monotonic()}
        try:
            p=subprocess.run(["nvidia-smi","--query-gpu=memory.used,temperature.gpu,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3)
            if p.returncode==0:
                vals=p.stdout.strip().splitlines()[0].split(",")
                for k,v in zip(["gpu_mib","temperature_c","gpu_util","power_w"],vals):
                    try: row[k]=float(v.strip())
                    except ValueError: pass
        except Exception: pass
        try:
            entries={}
            for d in Path("/proc").glob("[0-9]*"):
                try:
                    fields={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in (d/"status").read_text().splitlines() if ":" in line}
                    entries[int(d.name)]=(int(fields["PPid"]),int(fields.get("VmRSS","0 kB").split()[0])/1024)
                except (OSError,KeyError,ValueError): pass
            family={self.pid}
            while True:
                nxt=family|{pid for pid,(parent,_) in entries.items() if parent in family}
                if nxt==family: break
                family=nxt
            row["server_rss_mib"]=sum(entries.get(pid,(0,0))[1] for pid in family)
            mem={l.split(":")[0]:int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()}
            row["host_used_mib"]=(mem["MemTotal"]-mem["MemAvailable"])/1024
        except Exception: pass
        self.rows.append(row)
    def loop(self):
        while not self.event.is_set():
            self.sample(); self.event.wait(self.interval)
    def __enter__(self):
        self.sample(); self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.event.set();self.thread.join(timeout=5);self.sample()
    def metrics(self):
        out={"resource_samples":len(self.rows),"sample_interval_s":self.interval}
        for k in ["gpu_mib","server_rss_mib","host_used_mib","temperature_c","power_w"]:
            vals=[r[k] for r in self.rows if k in r]
            out["peak_"+k]=max(vals) if vals else None
        return out

def conservative_fit(prompt,cap,ctx):
    # Conservative byte-based guard, not a substitute for a tokenizer.
    return len((SYSTEM+prompt).encode("utf-8"))+cap+256<=ctx

def payload(cfg,item,nonce=""):
    prompt=(f"[Mã lượt đo {nonce}; bỏ qua mã này.]\n" if nonce else "")+item["prompt"]
    cap=item.get("cap",128)
    if not conservative_fit(prompt,cap,cfg.ctx):
        raise ValueError("Prompt vượt bộ kiểm tra dung lượng bảo thủ. Không cắt tự động; dùng thí nghiệm context lớn hơn.")
    return dict(model=MODEL,messages=[dict(role="system",content=SYSTEM),dict(role="user",content=prompt)],
                options=cfg.options(cap),stream=True,think=False,keep_alive="30m")

def stream_measure(api,body,on_text=None):
    start=time.perf_counter();first_content=None;first_generated=None;final=None;parts=[];thinking_chars=0
    for chunk in api.stream("/api/chat",body):
        now=time.perf_counter();msg=chunk.get("message",{})
        content=msg.get("content","");thinking=msg.get("thinking","")
        if (content or thinking) and first_generated is None: first_generated=now-start
        if content:
            if first_content is None: first_content=now-start
            parts.append(content)
            if on_text: on_text(content)
        thinking_chars+=len(thinking)
        if chunk.get("done"): final=chunk
    wall=time.perf_counter()-start
    if final is None: raise RuntimeError("Stream bị ngắt trước done=true; không dùng làm kết quả benchmark.")
    result=dict(content="".join(parts),wall_s=wall,ttft_content_s=first_content,
                ttft_generated_s=first_generated,thinking_chars=thinking_chars,done_reason=final.get("done_reason"))
    for k in ["total_duration","load_duration","prompt_eval_duration","eval_duration"]:
        result[k.replace("_duration","_s")]=final[k]/1e9 if k in final else None
    for k in ["prompt_eval_count","prompt_eval_cached_count","eval_count"]: result[k]=final.get(k)
    ed=result["eval_s"]; ec=result["eval_count"]
    result["decode_tps_engine"]=(ec/ed) if ec is not None and ed and ed>0 else None
    pd=result["prompt_eval_s"];pc=result["prompt_eval_count"];cached=result["prompt_eval_cached_count"]
    result["prefill_uncached_tps"]=(max(0,pc-cached)/pd) if pc is not None and cached is not None and pd and pd>0 else None
    result["end_to_end_tps"]=ec/wall if ec is not None and wall>0 else None
    return result

def grade(item,result):
    if item["kind"]=="perf": return None
    if result.get("done_reason")=="length" or not result.get("content","").strip(): return 0.0
    s=result["content"].strip()
    try:
        if s.startswith("```json"): s=s[7:].rsplit("```",1)[0].strip()
        return float(json.loads(s)==item["expected"])
    except (ValueError,KeyError): return 0.0

def benchmark_one(server,cfg,item,nonce="",on_text=None):
    log_start=len(server.log_text());body=payload(cfg,item,nonce)
    with Monitor(server.proc.pid) as monitor:
        result=stream_measure(server.api,body,on_text)
    result.update(monitor.metrics())
    result.update(prompt_id=item["id"],kind=item["kind"],task=item["task"],split=item["split"],
                  cap=item["cap"],prompt_bytes=len(item["prompt"].encode()),config=cfg.name)
    result["score"]=grade(item,result)
    result["fixed_length_ok"]=(item["kind"]=="perf" and result.get("eval_count")==item["cap"] and result["thinking_chars"]==0)
    tail=server.log_text()[log_start:]
    result["runtime_warnings"]=[l[-500:] for l in tail.splitlines() if any(x in l.lower() for x in ["unsupported","invalid option","truncating input","failed"])][-10:]
    if any("truncating input" in x.lower() for x in result["runtime_warnings"]):
        raise RuntimeError("Engine báo cắt prompt; loại phép đo và tăng context trong thí nghiệm mới.")
    if any("invalid option" in x.lower() for x in result["runtime_warnings"]):
        raise RuntimeError("Phiên bản engine không nhận tùy chọn; xem log, không coi là đã áp dụng.")
    if not result["content"].strip(): raise RuntimeError("Câu trả lời rỗng; kiểm tra think/cap/backend.")
    return result

def run_matrix(server,configs,items,root,label,repeats=2,seed=42):
    """Randomized profile blocks per repetition. Fresh model warm-up per block.
    Each prompt gets the same measurement nonce in all profiles, unique across
    prompts/repeats. Template-prefix caching may remain; raw cache counts saved.
    A repeated invocation creates a NEW run, never silently duplicates old rows.
    """
    root=Path(root);run=root/f"{label}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()%1000000:06d}"
    run.mkdir(parents=True,exist_ok=False)
    meta=dict(label=label,configs=[asdict(c) for c in configs],items_sha256=digest(items),
              repeats=repeats,seed=seed,hardware=hardware(),status="running",protocol="warm_model_unique_prompt_v1")
    save_json(run/"manifest.json",meta); rows=[];rng=random.Random(seed)
    for repeat in range(repeats):
        order=list(configs);rng.shuffle(order)
        for cfg in order:
            print(f"[{label}] lần {repeat+1}/{repeats}: {cfg.name}",flush=True)
            setup=time.perf_counter()
            try:
                server.start(cfg)
                warm=dict(id="warm",prompt="Chỉ viết: sẵn sàng.",cap=16,kind="perf",task="warm",split="warm")
                warm_result=stream_measure(server.api,payload(cfg,warm))
                setup_s=time.perf_counter()-setup
                save_json(run/f"setup_{repeat}_{cfg.name}.json",dict(setup_s=setup_s,warm=warm_result,ps=server.api.json("/api/ps"),log=str(server.log_path)))
                block=list(items);random.Random(seed+repeat).shuffle(block)
                for item in block:
                    key=digest([label,repeat,item["id"]])[:12]
                    row=dict(config=cfg.name,prompt_id=item["id"],split=item["split"],kind=item["kind"],task=item["task"],
                             cap=item["cap"],prompt_bytes=len(item["prompt"].encode()),repeat=repeat,
                             setup_s=setup_s,log=str(server.log_path),status="ok")
                    try: row.update(benchmark_one(server,cfg,item,key))
                    except Exception as e: row.update(status="error",error=str(e))
                    rows.append(row);append_jsonl(run/"raw.jsonl",row)
            except Exception as e:
                for item in items:
                    row=dict(config=cfg.name,prompt_id=item["id"],split=item["split"],kind=item["kind"],task=item["task"],
                             cap=item["cap"],prompt_bytes=len(item["prompt"].encode()),repeat=repeat,status="error",error=str(e))
                    rows.append(row);append_jsonl(run/"raw.jsonl",row)
    server.stop(); meta["status"]="complete";save_json(run/"manifest.json",meta)
    export_csv(run/"raw.csv",rows);save_json(root/f"latest_{label}.json",dict(path=str(run)))
    print("Kết quả:",run);return rows,run

def quantile(values,q):
    values=sorted(values)
    if not values:return None
    pos=(len(values)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)

def summarize(rows):
    out=[]
    for name in sorted({r["config"] for r in rows}):
        allr=[r for r in rows if r["config"]==name];ok=[r for r in allr if r["status"]=="ok"]
        r=dict(config=name,n=len(allr),failures=len(allr)-len(ok))
        for key in ["wall_s","ttft_content_s","decode_tps_engine","peak_gpu_mib","peak_server_rss_mib"]:
            vals=[x[key] for x in ok if x.get(key) is not None]
            r[key+"_median"]=statistics.median(vals) if vals else None
            if key=="wall_s":r["wall_s_p95"]=quantile(vals,.95)
        scores=[x["score"] for x in ok if x.get("score") is not None]
        r["quality_exact_json"]=statistics.mean(scores) if scores else None
        fixed=[x for x in ok if x.get("fixed_length_ok")]
        r["fixed_rows"]=len(fixed)
        r["fixed_decode_tps_median"]=statistics.median(x["decode_tps_engine"] for x in fixed if x.get("decode_tps_engine")) if fixed and all(x.get("decode_tps_engine") for x in fixed) else None
        out.append(r)
    return out

def paired_comparison(rows,baseline,other,seed=17):
    """Prompt-level bootstrap, not independent resampling of repeat measurements."""
    a={};b={}
    for r in rows:
        if r["status"]!="ok":continue
        dest=a if r["config"]==baseline else b if r["config"]==other else None
        if dest is not None: dest.setdefault(r["prompt_id"],[]).append(r)
    pairs=[]
    for pid in sorted(set(a)&set(b)):
        ar=a[pid];br=b[pid]
        # Performance rows only enter fixed-length comparison if both conform.
        if ar[0]["kind"]=="perf" and not all(x.get("fixed_length_ok") for x in ar+br):continue
        pairs.append((statistics.median(x["wall_s"] for x in ar),statistics.median(x["wall_s"] for x in br)))
    if not pairs:return dict(baseline=baseline,other=other,n_pairs=0,speedup=None)
    def ratio(p):return sum(x[0] for x in p)/sum(x[1] for x in p)
    rng=random.Random(seed);boot=[ratio([rng.choice(pairs) for _ in pairs]) for _ in range(2000)]
    return dict(baseline=baseline,other=other,n_pairs=len(pairs),speedup=ratio(pairs),
                bootstrap_ci95=[quantile(boot,.025),quantile(boot,.975)],
                definition="ratio of summed per-prompt median wall times; perf pairs require equal token cap")
