from inference.model_init import model_init
from inference.inference_utils import synthesize_utt, synthesize_utt_real_streaming, StreamingVCGenerator
from utils import load_audio
import torch
import torchaudio
import argparse
import pyaudio
import time
import numpy as np
import queue

class StreamingBuffer:
    def __init__(self, model, device, input_rate, output_rate, ref_audio):
        self.model = model
        self.device = device
        self.p = pyaudio.PyAudio()
        self.chunk = 1600 # 이걸 올리면 퀄리티가 올라갈거
        self.ref_audio = ref_audio
        self.ref_audio = self.ref_audio.to(self.device)
        self.cond_latent = model.get_gpt_cond_latents(self.ref_audio, model.config.audio.sample_rate)
        self.context_buffer = []

        self.input_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=input_rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        self.output_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=output_rate,
            output=True
        )

    def start(self):
        print("곧 시작합니다")
        try:
            # while문으로 계속 '입력버퍼 확인->변환->출력' 반복
            # 변환이 오래걸리면 인생 망함(RTF>1)
            while True:
                in_data = self.input_stream.read(self.chunk, exception_on_overflow=False)
                
                input_np = np.frombuffer(in_data, dtype=np.float32).copy()
                input_np *= 15 # 소리가 너무 작음
                current_tensor = torch.from_numpy(input_np).to(self.device).unsqueeze(0) 
                
                # 버퍼에 있는 데이터 + 현재 데이터
                padding_size = (2 - len(self.context_buffer)) * self.chunk
                if self.context_buffer:
                    input_tensor = torch.cat(self.context_buffer + [current_tensor], dim=1)
                else:
                    input_tensor = current_tensor
                input_tensor = torch.nn.functional.pad(input_tensor, (padding_size, 0), "constant", 0)
                
                self.context_buffer.append(current_tensor)
                if len(self.context_buffer) > 2:
                    self.context_buffer.pop(0)

                with torch.no_grad():
                    converted_tensor = synthesize_utt_real_streaming(self.model, input_tensor, self.cond_latent)
                    if(converted_tensor is None):
                        continue
                
                output_np = converted_tensor.squeeze().cpu().detach().numpy().astype(np.float32)
                self.output_stream.write(output_np.tobytes())

        except KeyboardInterrupt:
            # 닫기
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.p.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pre_trained/GenVC_small.pth')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--src_wav', type=str, default='samples/EF4_ENG_0112_1.wav')
    parser.add_argument('--ref_audio', type=str, default='samples/EM1_ENG_0037_1.wav')
    parser.add_argument('--output_path', type=str, default='samples/converted.wav')
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--streaming', default=True, action='store_true')
    args = parser.parse_args()

    model, config = model_init(args.model_path, args.device)

    # top_k is one of the important hyperparameters for inference, so you can tune it to get better results
    # for streaming inference, greedy decoding is preferred, you can set top_k to 1
    model.config.top_k = args.top_k
    src_wav = load_audio(args.src_wav, model.content_sample_rate)
    ref_audio = load_audio(args.ref_audio, model.config.audio.sample_rate)

    if args.streaming:
        streaming = StreamingBuffer(model, args.device, 44100, 44100, ref_audio)
        streaming.start()
    else:
        pre_audio = synthesize_utt(model, src_wav, ref_audio)
        torchaudio.save(args.output_path, pre_audio.unsqueeze(0).detach().cpu(), config.audio.sample_rate)