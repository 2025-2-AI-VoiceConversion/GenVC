"""
모든 프리셋 또는 그리드서치를 자동 실행하는 스크립트
Usage:
  python experiment/run_all.py --mode presets     # 프리셋만 실행
  python experiment/run_all.py --mode grid        # 그리드서치 실행
  python experiment/run_all.py --config baseline  # 특정 프리셋만 실행
"""

import argparse
from config_pool import get_config, get_all_configs, generate_grid_configs
from run_experiment import run_single_experiment

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['presets', 'grid', 'single'], default='single')
    parser.add_argument('--config', type=str, default='baseline')
    parser.add_argument('--run-nonstream', action='store_true', help='Run non-streaming inference for comparison')
    args = parser.parse_args()
    
    # 단일 모델 실험 
    if args.mode == 'single':
        config = get_config(args.config)
        run_single_experiment(config, run_nonstream=args.run_nonstream)
    
    # 프리셋 전부 실험 
    elif args.mode == 'presets':
        for name, config in get_all_configs().items():
            print(f"\n{'='*60}")
            print(f"Running Preset: {name}")
            print(f"{'='*60}")
            run_single_experiment(config, run_nonstream=args.run_nonstream)
    
    # 그리드 서치 실험 
    elif args.mode == 'grid':
        configs = generate_grid_configs()
        print(f"Total {len(configs)} grid configurations")

        for config in configs:
            print(f"\n{'='*60}")
            print(f"Running {config.experiment_name}")
            print(f"{'='*60}")
            run_single_experiment(config, run_nonstream=args.run_nonstream)

if __name__ == "__main__":
    main()