from dataclasses import dataclass

@dataclass
class StreamConfig:
    # 1. 구조적 파라미터
    chunk_size: int = 10240
    dvae_context: int = 320   # DVAE 인코딩 시 주변 정보를 얼마나 참조할 지 
    past_chunk_size: int = 0 # 0이면 현재 청크만 사용함 
    num_content_token: int = 8 # Input Contents Token Count - 이에 따라 입력 기대청크프레임 수는 1280 * num_content_token. 

    # 2. 추론 파라미터
    use_kv_cache: bool = True
    kv_cache_window: int = 100  # 0이면 슬라이드 윈도우 안함 
    top_k: int = 1 # 디코딩 top-k 
    
    # 3. 후처리 파라미터
    cross_fade_duration: int = 1024 # HiFi-GAN Cross Fading Duration
    
    # 4. 메타 정보
    experiment_name: str = "baseline"