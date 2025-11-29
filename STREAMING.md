# Confidence-Based Pseudo-Context Streaming-VC : Improving Streaming Voice Conversion via Future Frame Prediction

GenVC 모델을 Baseline 모델로 사용하여 스트리밍 최적화 모델로 개선하는 연구를 진행한다. 

## Setup
Create a new conda environment and install the required packages:
```sh
# Create a new Conda environment
conda create -n genVC python=3.10

# Activate the environment
conda activate genVC

# Install necessary dependencies
pip install pip==20.3.4
pip install transformers==4.33.0
pip install fairseq
pip install torch==2.3.0 torchaudio==2.3.0

# Install additional requirements
pip install -r requirements.txt
```

- 기존 infer.py 는 소스 파일 wav를 오프라인 파일 처리로 음성 변환 하므로 실시간 상황 테스트가 불가능하다.
- infer_stream.py 를 사용해 추론을 진행하면 된다.
- 추가된 Python 라이브러리 의존성인 `sounddevice`는 시스템에 `Portaudio` 를 필요로 하므로, 미리 설치해야 한다. 

```sh
# Ex) MacOS
brew install portaudio
```

### Streaming inference
```sh
python infer_stream.py --model_path pre_trained/GenVC_small.pth --ref_audio samples/EM1_ENG_0037_1.wav 
```

## Mode
```sh
python infer_stream.py --model_path pre_trained/GenVC_small.pth --ref_audio samples/EM1_ENG_0037_1.wav --mode default 
# 딥러닝 모델(GenVC)사용하여 처리

python infer_stream.py --model_path pre_trained/GenVC_small.pth --ref_audio samples/EM1_ENG_0037_1.wav --mode test
# 내가 말한 음성을 그대로 다시 출력함 
```

- 현재 녹음 및 출력 분리나 하울링 문제가 전혀 해결되어 있지 않으므로 볼륨을 줄여서 녹음 / 마이크를 사용해 녹음하는 것이 권장된다. 

# 리서치 요구 사항
- 인터리빙 방식 디코딩 변경
- 미래 문맥 생성
- 미래 문맥 Confidence 기반 핸들링
- Low Confidence 상황 딜레이 구현

# 엔지니어링 구현 사항 
- 하울링 문제 해결 
- 버퍼 처리 구현 (진행중)
- 최대 N초까지 과거문맥 어텐션 가능하게 해야함 (지금 계속 다른사람이 말하는 것과 동급임) 


# Citations
```
@inproceedings{baba2024utmosv2,
  title     = {The T05 System for The {V}oice{MOS} {C}hallenge 2024: Transfer Learning from Deep Image Classifier to Naturalness {MOS} Prediction of High-Quality Synthetic Speech},
  author    = {Baba, Kaito and Nakata, Wataru and Saito, Yuki and Saruwatari, Hiroshi},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2024},
}
```


