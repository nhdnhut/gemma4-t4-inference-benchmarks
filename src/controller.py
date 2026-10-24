"""Train a small latency predictor; select one configuration per session.
No Gemma weights are trained. Test observations are forbidden in this module.
"""
from dataclasses import asdict
from pathlib import Path
import statistics
from gemma_lab import BASELINE,Config,digest,save_json,export_csv

def features(item,cfg):
    return dict(prompt_bytes=item.get("prompt_bytes",len(item.get("prompt","").encode())),
                output_cap=item["cap"],task=item["task"],config=cfg.name)

def train_selector(rows,configs,root,budget_mib=13824,quality_drop=0.05):
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    import joblib
    if not rows or any(r["split"] not in {"train","validation"} for r in rows):
        raise ValueError("Chỉ dùng train/validation. Tuyệt đối không đưa test vào selector.")
    train_ids={r["prompt_id"] for r in rows if r["split"]=="train"}
    val_ids={r["prompt_id"] for r in rows if r["split"]=="validation"}
    if train_ids & val_ids:raise ValueError("Prompt xuất hiện ở cả train và validation.")
    if len(train_ids)<8 or len(val_ids)<4:raise ValueError("Cần ít nhất 8 prompt train và 4 validation; thêm dữ liệu.")
    names={c.name:c for c in configs}
    baseline_val=[r for r in rows if r["config"]==BASELINE.name and r["split"]=="validation"]
    if not baseline_val or any(r["status"]!="ok" for r in baseline_val):raise ValueError("Baseline validation chưa chạy thành công đầy đủ.")
    expected={(r["prompt_id"],r["repeat"]) for r in baseline_val}
    base_score=statistics.mean(r["score"] for r in baseline_val if r.get("score") is not None)
    candidates=[];audit=[]
    for cfg in configs:
        allc=[r for r in rows if r["config"]==cfg.name]
        vr=[r for r in allc if r["split"]=="validation"]
        reasons=[]
        if {(r["prompt_id"],r["repeat"]) for r in vr}!=expected:reasons.append("missing_validation_pairs")
        if any(r["status"]!="ok" for r in allc):reasons.append("runtime_error")
        good=[r for r in allc if r["status"]=="ok"]
        if not good or any(r.get("peak_gpu_mib") is None for r in good):reasons.append("missing_memory")
        peaks=[r["peak_gpu_mib"] for r in good if r.get("peak_gpu_mib") is not None]
        if peaks and max(peaks)>budget_mib:reasons.append("over_budget")
        if any(r.get("thinking_chars",0)>0 for r in good):reasons.append("thinking_not_disabled")
        if any(r["kind"]=="perf" and not r.get("fixed_length_ok") for r in good):reasons.append("performance_length_mismatch")
        scores=[r["score"] for r in vr if r["status"]=="ok" and r.get("score") is not None]
        score=statistics.mean(scores) if scores else None
        if score is None or score<base_score-quality_drop:reasons.append("quality_drop")
        if not reasons:candidates.append(cfg.name)
        audit.append(dict(config=cfg.name,eligible=not reasons,reasons=reasons,val_score=score,peak_mib=max(peaks) if peaks else None))
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    save_json(root/"candidate_audit.json",audit)
    export_csv(root/"candidate_audit.csv",audit)
    if not candidates:raise ValueError("Không có cấu hình đạt ràng buộc. Xem candidate_audit/raw/log; không tự nới ngưỡng.")
    tr=[r for r in rows if r["split"]=="train" and r["status"]=="ok" and r["config"] in candidates]
    va=[r for r in rows if r["split"]=="validation" and r["status"]=="ok" and r["config"] in candidates]
    # Aggregate repeats before training. Features are available BEFORE inference.
    groups={}
    for r in tr:groups.setdefault((r["prompt_id"],r["config"]),[]).append(r)
    samples=[(rs[0],statistics.median(r["wall_s"] for r in rs)) for rs in groups.values()]
    candidates_fit=[]
    for depth in [3,6]:
        pipe=make_pipeline(DictVectorizer(sparse=False),RandomForestRegressor(
            n_estimators=128,max_depth=depth,min_samples_leaf=2,random_state=42,n_jobs=2))
        pipe.fit([features(r,names[r["config"]]) for r,y in samples],np.log([y for r,y in samples]))
        pred=np.exp(pipe.predict([features(r,names[r["config"]]) for r in va]))
        mae=float(np.mean(np.abs(pred-np.array([r["wall_s"] for r in va]))))
        candidates_fit.append((mae,depth,pipe))
    mae,depth,pipe=min(candidates_fit,key=lambda x:x[0])
    def cost(name):
        subset=[r for r in va if r["config"]==name]
        return statistics.mean(r["wall_s"] for r in subset)
    best_fixed=min(candidates,key=cost)
    val_items=list({r["prompt_id"]:r for r in va}.values())
    predicted={name:float(np.exp(pipe.predict([features(it,names[name]) for it in val_items])).mean()) for name in candidates}
    proposed=min(predicted,key=predicted.get)
    # A learned rule earns deployment only when validation measures an advantage.
    use_learned=cost(proposed)<=cost(best_fixed)*1.03
    if BASELINE.name in candidates:
        use_learned=use_learned and cost(proposed)<cost(BASELINE.name)*0.95
    selected=proposed if use_learned else best_fixed
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    joblib.dump(pipe,root/"latency_predictor.joblib")
    out=dict(selected_config=asdict(names[selected]),best_fixed_config=asdict(names[best_fixed]),
             proposed_config=asdict(names[proposed]),eligible=candidates,
             policy="learned_session_selector" if use_learned else "measured_fixed_fallback",
             validation_mae_s=mae,max_depth=depth,budget_mib=budget_mib,quality_drop=quality_drop,
             training_prompt_count=len(train_ids),validation_prompt_count=len(val_ids),
             validation_predicted_mean_s=predicted,validation_measured_mean_s={n:cost(n) for n in candidates},
             train_data_sha256=digest(rows),notice="Small synthetic workload; not a claim of general LLM quality or universal speedup.")
    save_json(root/"selection.json",out);save_json(root/"candidate_audit.json",audit)
    export_csv(root/"candidate_audit.csv",audit)
    save_json(root/"training_log.json",dict(trials=[dict(depth=d,validation_mae_s=m) for m,d,_ in candidates_fit],target="log warm wall seconds",trained_weights="RandomForest only; Gemma unchanged"))
    return out,audit

def predict_workload(model_path,items,configs):
    """Analysis only. Switching is once per session, never a per-token control."""
    import joblib,numpy as np
    predictor=joblib.load(model_path)
    return {c.name:float(np.exp(predictor.predict([features(i,c) for i in items])).mean()) for c in configs}
