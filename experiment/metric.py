import torch
import torchaudio
import numpy as np
import os
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from torch.nn.functional import cosine_similarity

class SpeakerSimilarityMetric:
    def __init__(self, model_name="microsoft/wavlm-base-plus-sv", device="cuda" if torch.cuda.is_available() else "cpu", cache_dir="pre_trained/metrics/wavlm"):
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"[SpeakerSimilarityMetric] 모델 로딩 중: {model_name} (저장소: {self.cache_dir})")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.model = WavLMForXVector.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device)
        self.model.eval()

    def compute(self, ref_wav_path, deg_wav_path):
        """
        참조(Reference) 오디오와 생성된(Degraded) 오디오 간의 코사인 유사도를 계산
        """
        # 오디오 로드
        ref_wav, sr_ref = torchaudio.load(ref_wav_path)
        deg_wav, sr_deg = torchaudio.load(deg_wav_path)

        # 리샘플링 (WavLM은 보통 16kHz를 기대함)
        if sr_ref != 16000:
            ref_wav = torchaudio.transforms.Resample(sr_ref, 16000)(ref_wav)
        if sr_deg != 16000:
            deg_wav = torchaudio.transforms.Resample(sr_deg, 16000)(deg_wav)

        # 입력 처리
        inputs_ref = self.feature_extractor(ref_wav.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        inputs_deg = self.feature_extractor(deg_wav.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(self.device)

        with torch.no_grad():
            emb_ref = self.model(inputs_ref).embeddings
            emb_deg = self.model(inputs_deg).embeddings

        # 코사인 유사도 계산
        similarity = cosine_similarity(emb_ref, emb_deg).item()
        return similarity

class MOSMetric:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", cache_dir="pre_trained/metrics/utmos"):
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"[MOSMetric] 모델 로딩 중: UTMOSv2 (저장소: {self.cache_dir})")
        # torch.hub.set_dir를 사용하여 캐시 디렉토리 설정
        torch.hub.set_dir(self.cache_dir)
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(self.device)
        self.model.eval()

    def compute(self, wav_path):
        """
        주어진 오디오 파형에 대한 MOS 점수를 예측
        """
        wav, sr = torchaudio.load(wav_path)
        
        # UTMOS는 16kHz를 권장합니다.
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            
        wav = wav.to(self.device)
        
        with torch.no_grad():
            score = self.model(wav, 16000)
            
        return score.item()

class Metric:
    def __init__(self):
        self.results = []
        # 프로젝트 루트 기준으로 캐시 경로 설정
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        wavlm_cache = os.path.join(root_dir, "pre_trained/metrics/wavlm")
        utmos_cache = os.path.join(root_dir, "pre_trained/metrics/utmos")
        
        self.sim_metric = SpeakerSimilarityMetric(cache_dir=wavlm_cache)
        self.mos_metric = MOSMetric(cache_dir=utmos_cache)

    def update(self, result_dict):
        """
        result_dict는 다음을 포함해야 합니다:
        - 'ref_wav_path': 참조 오디오 경로
        - 'gen_wav_path': 생성된 오디오 경로
        - 'rtf': Real-Time Factor (선택)
        - 'latency': 지연 시간 (선택)
        """
        ref_path = result_dict.get('ref_wav_path')
        gen_path = result_dict.get('gen_wav_path')
        
        metrics = result_dict.copy()
        
        if ref_path and gen_path:
            sim_score = self.sim_metric.compute(ref_path, gen_path)
            mos_score = self.mos_metric.compute(gen_path)
            metrics['similarity'] = sim_score
            metrics['mos'] = mos_score
            
        self.results.append(metrics)

    def report(self):
        if not self.results:
            return {}
            
        avg_rtf = np.mean([r['rtf'] for r in self.results if 'rtf' in r]) if any('rtf' in r for r in self.results) else 0.0
        avg_sim = np.mean([r['similarity'] for r in self.results if 'similarity' in r]) if any('similarity' in r for r in self.results) else 0.0
        avg_mos = np.mean([r['mos'] for r in self.results if 'mos' in r]) if any('mos' in r for r in self.results) else 0.0
        
        return {
            "avg_rtf": avg_rtf,
            "avg_similarity": avg_sim,
            "avg_mos": avg_mos,
            "count": len(self.results)
        } 