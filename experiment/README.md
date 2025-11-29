# GenVC Streaming Experiment Guide

이 문서는 GenVC 모델의 스트리밍 추론 성능을 평가하고, 논스트리밍(Baseline)과 비교하기 위한 실험 가이드입니다.

## 1. 환경 설정 (Requirements)

실험을 실행하기 위해 다음 패키지들이 필요합니다. `requirements.txt`에 포함되어 있는지 확인해주세요.

```txt
pyaudio
transformers
tqdm
torchaudio
numpy
scipy
```

*   `pyaudio`: 스트리밍 오디오 입출력 및 버퍼링
*   `transformers`: WavLM(화자 유사도 측정) 모델 로딩
*   `tqdm`: 데이터셋 준비 과정의 진행률 표시

## 2. 데이터셋 준비

최초 1회, 다음 명령어를 실행하여 CMU Arctic 데이터셋을 다운로드하고 테스트 셋을 준비합니다.

```bash
python experiment/prepare_dataset.py
```

*   자동으로 데이터셋을 다운로드하고, `experiment/data/test_set`에 테스트용 오디오 파일을 생성합니다.
*   랜덤 시드가 고정되어 있어 언제 실행해도 동일한 테스트 쌍이 생성됩니다.

## 3. 실험 실행

다음 명령어로 실험을 진행합니다.

```bash
python experiment/run.py --mode single --config high
```

*   `--config high`: 현재 가장 최적화된 설정('high')으로 실험을 진행합니다.
*   **최초 실행 시**: 평가에 필요한 모델(`metrics/utmos`, `wavlm`)을 `pre_trained` 폴더에 자동으로 다운로드합니다.
    *   `utmos`: 음성 품질(MOS) 예측 모델
    *   `wavlm`: 화자 유사도(Speaker Similarity) 측정 모델 (GenVC 논문 참조)

### 논스트리밍 비교 (Optional)

스트리밍 추론과 기존 논스트리밍(GenVC 기본) 추론을 비교하려면 `--run-nonstream` 옵션을 추가하세요.
추가 하지 않는 경우 스트리밍 평가 실험만 진행합니다. 

```bash
python experiment/run.py --mode single --config high --run-nonstream
```

## 4. 커스텀 설정 추가 방법

새로운 실험 설정을 추가하려면 `experiment/config_pool.py` 파일을 수정하세요.

```python
# experiment/config_pool.py

CONFIGS = {
    # ... 기존 설정들 ...
    
    # 새로운 설정 추가
    "my_custom_config": StreamConfig(
        chunk_size=10240,
        use_kv_cache=True,
        kv_cache_window=50,
        experiment_name="my_experiment_v1" # 결과가 저장될 폴더 이름
    ),
}
```

설정을 추가한 후, 다음 명령어로 실행할 수 있습니다.

```bash
python experiment/run.py --mode single --config my_custom_config
```

## 5. 결과 확인

실험 결과는 `experiment/results/high` 폴더에 저장됩니다. (`high`는 실험 설정 이름입니다)

### 폴더 구조
*   `streaming/`: 스트리밍 추론 결과 오디오 (10개)
*   `non_streaming/`: 논스트리밍 추론 결과 오디오 (옵션 활성화 시)
*   `source/`: 원본 소스 오디오 (Ground Truth)
*   `reference/`: 참조 오디오 (Ground Truth)

`source`와 `reference` 폴더의 오디오를 직접 들어보며 생성된 결과물과 비교해볼 수 있습니다.

### 리포트 (comparison_report.json)

실험 완료 후 생성되는 `comparison_report.json` 파일에서 정량적 지표를 확인할 수 있습니다.

```json
"streaming": {
    "avg_rtf": 1.44,        // Real-Time Factor (낮을수록 빠름)
    "avg_similarity": 0.86, // 화자 유사도 (높을수록 좋음)
    "avg_mos": 2.87,        // MOS (음성 품질, 높을수록 좋음)
    "count": 10
}
```

*   **RTF (Real-Time Factor)**: (처리 시간 / 생성된 오디오 길이). 1.0 미만이면 실시간 처리보다 빠름을 의미합니다.
*   **Similarity**: WavLM 기반 코사인 유사도. (GenVC Small 논문 기준 약 0.869)
*   **MOS**: UTMOS 기반 예측 점수. (GenVC Small 논문 기준 약 2.66)

GenVC와 테스트 데이터셋 샘플 개수 (2000개 vs 10개)가 좀 다르다 보니, 논스트리밍 결과와 비교하여 스트리밍 적용 시 성능 저하가 어느 정도인지 분석하는 데 활용해봅시다. 
