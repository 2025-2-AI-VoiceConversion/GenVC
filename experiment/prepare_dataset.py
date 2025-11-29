import os
import random
import json
import torch
import torchaudio
from torchaudio.datasets import CMUARCTIC
from tqdm import tqdm

def main():
    # 설정
    base_dir = os.path.dirname(__file__)
    data_root = os.path.join(base_dir, "data", "cmu_arctic") # 원본 데이터 저장 위치
    output_dir = os.path.join(base_dir, "data", "test_set")   # 테스트용 샘플 저장 위치
    json_path = os.path.join(base_dir, "data", "test_pairs.json")
    num_pairs = 10
    
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Dataset] CMU Arctic 데이터셋 준비 중...")
    
    # 사용할 화자 목록 (남/녀 균형)
    target_speakers = ['slt', 'clb', 'bdl', 'rms']
    loaded_datasets = {}
    
    for spk in target_speakers:
        print(f"  - 화자 다운로드 및 로딩 중: {spk}")
        try:
            # download=True로 설정하여 자동 다운로드
            ds = CMUARCTIC(root=data_root, url=spk, download=True)
            loaded_datasets[spk] = ds
        except Exception as e:
            print(f"  [Error] {spk} 로딩 실패: {e}")
            
    if len(loaded_datasets) < 2:
        print("[Error] 충분한 화자 데이터를 로드하지 못했습니다.")
        return

    pairs = []
    
    print(f"[Dataset] {num_pairs}개의 랜덤 테스트 쌍 생성 중...")
    
    # 랜덤 시드 고정 (재현성)
    random.seed(42)
    
    for i in tqdm(range(num_pairs)):
        # 랜덤 소스 및 타겟 화자 선택
        src_spk = random.choice(list(loaded_datasets.keys()))
        tgt_spk = random.choice(list(loaded_datasets.keys()))
        
        # 랜덤 발화 선택
        # CMUARCTIC dataset item: (waveform, sample_rate, transcript, utterance_id)
        src_idx = random.randint(0, len(loaded_datasets[src_spk]) - 1)
        tgt_idx = random.randint(0, len(loaded_datasets[tgt_spk]) - 1)
        
        src_item = loaded_datasets[src_spk][src_idx]
        tgt_item = loaded_datasets[tgt_spk][tgt_idx]
        
        src_wav, src_sr, src_text, src_id = src_item
        tgt_wav, tgt_sr, tgt_text, tgt_id = tgt_item
        
        # 파일명 생성
        src_filename = f"pair{i}_src_{src_spk}_{src_id}.wav"
        tgt_filename = f"pair{i}_tgt_{tgt_spk}_{tgt_id}.wav"
        
        src_path = os.path.join(output_dir, src_filename)
        tgt_path = os.path.join(output_dir, tgt_filename)
        
        # 오디오 저장
        # src_wav는 이미 Tensor [Channels, Time] 형태임
        torchaudio.save(src_path, src_wav, src_sr)
        torchaudio.save(tgt_path, tgt_wav, tgt_sr)
        
        # 상대 경로로 변환 (프로젝트 루트 기준)
        # base_dir은 experiment 폴더이므로, 그 상위 폴더가 프로젝트 루트
        project_root = os.path.dirname(base_dir)
        rel_src_path = os.path.relpath(src_path, project_root)
        rel_tgt_path = os.path.relpath(tgt_path, project_root)
        
        pairs.append({
            "id": i,
            "src_path": rel_src_path,
            "ref_path": rel_tgt_path,
            "src_speaker": src_spk,
            "tgt_speaker": tgt_spk,
            "src_text": src_text,
            "tgt_text": tgt_text
        })
        
    # JSON 저장
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(pairs, f, indent=4, ensure_ascii=False)
        
    print(f"[Dataset] 완료! {num_pairs}개의 쌍이 {json_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()
