import os
import sys
import json
import torch
import torchaudio
import numpy as np
from datetime import datetime
from argparse import Namespace

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.stream_config import StreamConfig
from inference.model_init import model_init
from inference.inference_utils import synthesize_utt
from utils import load_audio
from experiment.metric import Metric
from infer_fortest import StreamingBuffer

class ExperimentStreamingBuffer(StreamingBuffer):
    
    def __init__(self, model, device, ref_audio, config: StreamConfig, output_path: str):

        mock_args = Namespace(
            mode='file_stream',
            save_audio=1,  # StreamWriter 사용
            output_path=output_path,
            top_k=config.top_k,
        )
        
        super().__init__(model, device, ref_audio, mock_args, stream_config=config)

class NonStreamInferenceWrapper:
    def __init__(self, config: StreamConfig, device='cpu', model_path="pre_trained/GenVC_small.pth"): 
        self.config = config
        self.device = device
        
        if not os.path.exists(model_path):
             # 루트에서 찾기 시도
            root_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path)
            if os.path.exists(root_model_path):
                model_path = root_model_path
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")

        self.model, _ = model_init(model_path, device)
        self.model.to(device)
        self.model.eval()
        
        self.model.config.top_k = config.top_k

    def run_file(self, src_wav_path, ref_wav_path, output_path):
        # 오디오 로드
        src_wav = load_audio(src_wav_path, 16000)
        ref_wav = load_audio(ref_wav_path, 24000) 
        
        if src_wav is None or ref_wav is None:
            return None

        src_wav = src_wav.to(self.device)
        ref_wav = ref_wav.to(self.device)

        start_time = datetime.now()
        
        # 논스트리밍 추론 (synthesize_utt 사용)
        with torch.no_grad():
            wav_gen = synthesize_utt(self.model, src_wav, ref_wav, seg_len=6.0) 
            
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 결과 저장
        if wav_gen is not None:
            final_audio = wav_gen.cpu().unsqueeze(0)
            torchaudio.save(output_path, final_audio, 24000)
            
            # RTF 계산: 생성된 오디오 길이 기준
            audio_len_sec = final_audio.shape[-1] / 24000
            rtf = duration / audio_len_sec
            
            return {
                "rtf": rtf,
                "latency": 0.0, 
                "gen_wav_path": output_path,
                "ref_wav_path": ref_wav_path
            }
        return None

def run_single_experiment(config: StreamConfig, run_nonstream: bool = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Running Experiment: {config.experiment_name}")
    
    # 결과 저장 경로
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results")
    exp_dir = os.path.join(results_dir, config.experiment_name)
    stream_results_dir = os.path.join(exp_dir, "streaming")
    non_stream_results_dir = os.path.join(exp_dir, "non_streaming")
    source_results_dir = os.path.join(exp_dir, "source")
    reference_results_dir = os.path.join(exp_dir, "reference")
    
    os.makedirs(stream_results_dir, exist_ok=True)
    os.makedirs(source_results_dir, exist_ok=True)
    os.makedirs(reference_results_dir, exist_ok=True)
    if run_nonstream:
        os.makedirs(non_stream_results_dir, exist_ok=True)
    
    # 테스트 데이터 로드
    json_path = os.path.join(base_dir, "data", "test_pairs.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run prepare_dataset.py first.")
        return
        
    with open(json_path, "r", encoding='utf-8') as f:
        test_pairs = json.load(f)
        
    print(f"Loaded {len(test_pairs)} test pairs.")
    
    # 모델 초기화 (공유)
    model_path = "pre_trained/GenVC_small.pth"
    if not os.path.exists(model_path):
        root_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path)
        if os.path.exists(root_model_path):
            model_path = root_model_path
    
    model, _ = model_init(model_path, device)
    model.to(device)
    model.eval()
    model.config.top_k = config.top_k
    
    # Non-Streaming Wrapper (모델 재사용)
    non_stream_wrapper = None
    if run_nonstream:
        non_stream_wrapper = NonStreamInferenceWrapper(config, device, model_path)
        # 모델 객체 교체 (이미 로드된 모델 사용)
        non_stream_wrapper.model = model 
    
    # 메트릭 초기화
    metric_stream = Metric()
    metric_non_stream = Metric()
    
    experiment_results = []
    
    for pair in test_pairs:
        pair_id = pair['id']
        src_path = pair['src_path']
        ref_path = pair['ref_path']
        
        # 상대 경로인 경우 절대 경로로 변환
        if not os.path.isabs(src_path):
            project_root = os.path.dirname(base_dir) # experiment 폴더의 상위
            src_path = os.path.join(project_root, src_path)
        
        if not os.path.isabs(ref_path):
            project_root = os.path.dirname(base_dir)
            ref_path = os.path.join(project_root, ref_path)
            
        print(f"\nProcessing Pair {pair_id}: {pair['src_speaker']} -> {pair['tgt_speaker']}")
        
        # 오디오 로드
        src_wav = load_audio(src_path, 16000) # Source는 16000Hz
        ref_wav = load_audio(ref_path, 24000) # Reference는 24000Hz
        
        if src_wav is None or ref_wav is None:
            print("Error loading audio")
            continue
            
        # 소스 및 참조 오디오 저장 (비교 청취용)
        src_save_path = os.path.join(source_results_dir, f"pair{pair_id}_src.wav")
        ref_save_path = os.path.join(reference_results_dir, f"pair{pair_id}_ref.wav")
        torchaudio.save(src_save_path, src_wav.cpu(), 16000)
        torchaudio.save(ref_save_path, ref_wav.cpu(), 24000)
        
        # 1. Streaming Inference
        stream_out_path = os.path.join(stream_results_dir, f"pair{pair_id}_stream.wav")
        print(f"  Running Streaming Inference...")
            
        # StreamingBuffer 초기화 및 실행
        try:
            streamer = ExperimentStreamingBuffer(model, device, ref_wav, config, stream_out_path)
            
            start_time = datetime.now()
            streamer.start(src_wav) # 실행
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            # RTF 계산: 생성된 오디오 길이 기준
            if os.path.exists(stream_out_path):
                gen_wav, gen_sr = torchaudio.load(stream_out_path)
                audio_len_sec = gen_wav.shape[-1] / gen_sr
                rtf = duration / audio_len_sec
            else:
                # 생성 실패 시 0.0 처리 (또는 에러)
                rtf = 0.0
                print("    Warning: Generated file not found, RTF set to 0.0")
            
            stream_res = {
                "rtf": rtf,
                "latency": 0.0,
                "gen_wav_path": stream_out_path,
                "ref_wav_path": ref_path
            }
            
            metric_stream.update(stream_res)
            print(f"    Stream RTF: {rtf:.3f}")
            
        except Exception as e:
            print(f"    Streaming Failed: {e}")
            import traceback
            traceback.print_exc()
            stream_res = None

        # 2. Non-Streaming Inference
        non_stream_res = None
        if run_nonstream:
            non_stream_out_path = os.path.join(non_stream_results_dir, f"pair{pair_id}_nonstream.wav")
            print(f"  Running Non-Streaming Inference...")
            non_stream_res = non_stream_wrapper.run_file(src_path, ref_path, non_stream_out_path)
            
            if non_stream_res:
                # Non-Streaming Wrapper 내부에서 RTF를 계산해서 주지만, 일관성을 위해 여기서 다시 계산할 수도 있음.
                # 하지만 Wrapper 내부 로직을 존중하여 그대로 사용.
                # 만약 Wrapper도 수정해야 한다면 알려주세요. 일단은 Wrapper 결과 사용.
                metric_non_stream.update(non_stream_res)
                print(f"    Non-Stream RTF: {non_stream_res['rtf']:.3f}")
            
        experiment_results.append({
            "pair_id": pair_id,
            "stream": stream_res,
            "non_stream": non_stream_res
        })

    # 최종 리포트 생성
    print("\nGenerating Final Report...")
    stream_report = metric_stream.report()
    non_stream_report = metric_non_stream.report()
    
    final_report = {
        "config": str(config),
        "streaming": stream_report,
        "non_streaming": non_stream_report,
        "details": experiment_results
    }
    
    report_path = os.path.join(exp_dir, "comparison_report.json")
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
        
    print(f"Experiment Complete. Report saved to {report_path}")
    print("Streaming Report:", json.dumps(stream_report, indent=2))
    print("Non-Streaming Report:", json.dumps(non_stream_report, indent=2))

def main():
    # 기본 실행 (Baseline)
    from experiment.config_pool import get_config
    config = get_config("baseline")
    run_single_experiment(config)

if __name__ == "__main__":
    main()