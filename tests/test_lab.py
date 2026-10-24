import sys,json,unittest,tempfile
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from gemma_lab import *
from dataset import build_dataset
from controller import train_selector,features

class FakeAPI:
    def __init__(self,chunks):self.chunks=chunks
    def stream(self,*args):yield from self.chunks

class LabTests(unittest.TestCase):
    def test_stream_metrics_and_empty_first_chunk(self):
        chunks=[dict(message=dict(content="")),dict(message=dict(content="Xin ")),
                dict(message=dict(content="chào")),dict(done=True,eval_count=4,eval_duration=2_000_000_000,
                prompt_eval_count=12,prompt_eval_cached_count=2,prompt_eval_duration=1_000_000_000)]
        r=stream_measure(FakeAPI(chunks),{})
        self.assertEqual(r["content"],"Xin chào");self.assertEqual(r["decode_tps_engine"],2)
        self.assertEqual(r["prefill_uncached_tps"],10);self.assertIsNotNone(r["ttft_content_s"])
    def test_incomplete_stream_rejected(self):
        with self.assertRaises(RuntimeError):stream_measure(FakeAPI([dict(message=dict(content="x"))]),{})
    def test_missing_metrics_not_zero(self):
        r=stream_measure(FakeAPI([dict(done=True,message=dict(content="x"))]),{})
        self.assertIsNone(r["decode_tps_engine"]);self.assertIsNone(r["eval_count"])
    def test_thinking_not_ttft_content(self):
        r=stream_measure(FakeAPI([dict(message=dict(thinking="abc")),dict(message=dict(content="x")),dict(done=True)]),{})
        self.assertEqual(r["thinking_chars"],3)
        self.assertLessEqual(r["ttft_generated_s"],r["ttft_content_s"])
    def test_quality_not_substring(self):
        i=dict(kind="quality",expected={"total":42})
        self.assertEqual(grade(i,dict(content='{"total": 42}',done_reason="stop")),1)
        self.assertEqual(grade(i,dict(content='{"total": 420}',done_reason="stop")),0)
        self.assertEqual(grade(i,dict(content='{"total": 42}',done_reason="length")),0)
    def test_context_refusal(self):
        with self.assertRaises(ValueError):payload(BASELINE,dict(prompt="x"*5000,cap=128))
    def test_config_no_silent_draft_override(self):
        self.assertNotIn("draft_num_predict",BASELINE.options(128))
        self.assertEqual(Config("no",draft=0).options(128)["draft_num_predict"],0)
    def test_dataset_split_and_fit(self):
        data=build_dataset();self.assertEqual(len(data),52)
        self.assertEqual(len({x["prompt"] for x in data}),len(data))
        for it in data:payload(BASELINE,it,"0123456789ab")
    def test_bootstrap_paired(self):
        rows=[]
        for i in range(8):
            for name,t in [("a",10),("b",5)]:
                rows.append(dict(config=name,prompt_id=str(i),wall_s=t,status="ok",kind="quality"))
        r=paired_comparison(rows,"a","b");self.assertEqual(r["speedup"],2)
        self.assertEqual(r["bootstrap_ci95"],[2,2])
    def test_train_refuses_test(self):
        with self.assertRaises(ValueError):train_selector([dict(split="test")],CONFIGS,"/tmp/unused")
    def test_train_and_constraints(self):
        rows=[];cs=CONFIGS[:3]
        for it in build_dataset(quick=True):
            if it["split"]=="test":continue
            for cfg,t,mem in [(cs[0],3.0,11000),(cs[1],2.0,10500),(cs[2],1.0,15000)]:
                rows.append(dict(prompt_id=it["id"],split=it["split"],kind=it["kind"],task=it["task"],cap=it["cap"],
                    prompt_bytes=len(it["prompt"].encode()),repeat=0,config=cfg.name,status="ok",wall_s=t,
                    peak_gpu_mib=mem,score=1.0 if it["kind"]=="quality" else None,fixed_length_ok=True,thinking_chars=0))
        with tempfile.TemporaryDirectory() as td:
            selection,audit=train_selector(rows,cs,td)
            self.assertEqual(selection["selected_config"]["name"],cs[1].name)
            self.assertNotIn(cs[2].name,selection["eligible"])
            self.assertTrue(Path(td,"latency_predictor.joblib").exists())
            self.assertEqual(selection["best_fixed_config"]["name"],cs[1].name)
    def test_pre_inference_features(self):
        x=features(dict(prompt="abc",cap=128,task="explanation",eval_count=999,wall_s=.1),BASELINE)
        self.assertNotIn("eval_count",x);self.assertNotIn("wall_s",x)
    def test_model_lock_rejects_changed_digest(self):
        class LocalMetadata:
            value="sha256:first"
            def json(self,path,body=None):
                if path=='/api/tags':return {'models':[{'name':MODEL,'digest':self.value}]}
                if path=='/api/show':return {'details':{'quantization_level':'Q4_K_M'},'template':'test'}
                if path=='/api/version':return {'version':'test-version'}
                raise AssertionError(path)
        api=LocalMetadata()
        with tempfile.TemporaryDirectory() as td,patch('gemma_lab.hardware',return_value={'gpu_name':'MOCK'}):
            save_json(Path(td)/'dataset.json',[{'id':'x'}])
            establish_lock(api,td,create=True)
            establish_lock(api,td)
            api.value='sha256:changed'
            with self.assertRaises(RuntimeError):establish_lock(api,td)
    def test_run_matrix_keeps_failures_and_manifest(self):
        class LocalServer:
            api=FakeAPI([dict(done=True,message={'content':'ok'})]);log_path='mock.log'
            def start(self,cfg):self.config=cfg
            def stop(self):pass
        server=LocalServer();server.api.json=lambda *args:{'models':[]}
        items=build_dataset(quick=True)[:2]
        def fake_one(server,cfg,item,nonce):
            if cfg.name=='bad':raise RuntimeError('simulated OOM')
            return dict(content='ok',wall_s=2,score=1)
        with tempfile.TemporaryDirectory() as td,patch('gemma_lab.hardware',return_value={'gpu_name':'MOCK'}),patch('gemma_lab.benchmark_one',side_effect=fake_one):
            rows,path=run_matrix(server,[BASELINE,Config('bad')],items,td,'mock_only',repeats=2)
            self.assertEqual(len(rows),8)
            self.assertEqual(sum(x['status']=='error' for x in rows),4)
            self.assertEqual(len(read_jsonl(path/'raw.jsonl')),8)
            self.assertEqual(load_json(path/'manifest.json')['status'],'complete')
    def test_performance_pair_requires_equal_length(self):
        rows=[dict(config='a',prompt_id='x',wall_s=10,status='ok',kind='perf',fixed_length_ok=True),
              dict(config='b',prompt_id='x',wall_s=1,status='ok',kind='perf',fixed_length_ok=False)]
        self.assertEqual(paired_comparison(rows,'a','b')['n_pairs'],0)

if __name__=="__main__":unittest.main()
