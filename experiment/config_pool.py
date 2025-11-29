from stream_config import StreamConfig

# Preset Configurations
CONFIGS = {
    
    # Baseline
    "baseline": StreamConfig(
        chunk_size=10240,
        dvae_context=0,
        past_chunk_size=0,
        num_content_token=8,
        use_kv_cache=True,
        kv_cache_window=100,
        top_k=10,
        cross_fade_duration=100,
        experiment_name="baseline"
    ),
    
    # Low Latency
    "low_latency": StreamConfig(
        chunk_size=5120,  
        kv_cache_window=50,
        top_k=5, 
        cross_fade_duration=50,
        experiment_name="low_latency"
    ),
    
    # High Quality
    "high": StreamConfig(
        chunk_size=10240,  
        dvae_context=0, 
        past_chunk_size=0,  
        kv_cache_window=1000,
        use_kv_cache=False,
        top_k=1,
        cross_fade_duration=1024,
        experiment_name="high" 
    ),
    
    # KV Cache Optimization
    "optimized_cache": StreamConfig(
        chunk_size=10240,
        kv_cache_window=150,
        cross_fade_duration=100,
        experiment_name="optimized_cache"
    ),
    
    # Memory Efficient
    "memory_efficient": StreamConfig(
        chunk_size=10240,
        use_kv_cache=True,
        kv_cache_window=30,  # 작은 윈도우
        experiment_name="memory_efficient"
    ),
}
def get_config(name: str) -> StreamConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]

def get_all_configs() -> dict:
    """모든 프리셋 반환"""
    return CONFIGS

# 그리드서치용 설정 생성기
def generate_grid_configs():
    """그리드서치용 config 조합 생성"""
    import itertools
    
    param_grid = {
        'chunk_size': [5120, 10240, 20480],
        'kv_cache_window': [50, 100, 200],
        'top_k': [5, 10, 15],
        'cross_fade_duration': [50, 100, 200],
    }
    
    configs = []
    for idx, values in enumerate(itertools.product(*param_grid.values())):
        config_dict = dict(zip(param_grid.keys(), values))
        
        # StreamConfig 객체 생성 (기본값 포함)
        temp_config = StreamConfig(**config_dict)
        
        # 모든 필드를 이름에 포함
        # 순서: chunk -> dvae -> past -> num -> use -> win -> topk -> fade
        name_parts = ["grid"]
        name_parts.append(f"chunk{temp_config.chunk_size}")
        name_parts.append(f"dvae{temp_config.dvae_context}")
        name_parts.append(f"past{temp_config.past_chunk_size}")
        name_parts.append(f"num{temp_config.num_content_token}")
        name_parts.append(f"use{'T' if temp_config.use_kv_cache else 'F'}") # True/False -> T/F
        name_parts.append(f"win{temp_config.kv_cache_window}")
        name_parts.append(f"topk{temp_config.top_k}")
        name_parts.append(f"fade{temp_config.cross_fade_duration}")
            
        config_dict['experiment_name'] = "_".join(name_parts)
        configs.append(StreamConfig(**config_dict))
    
    return configs
    
    