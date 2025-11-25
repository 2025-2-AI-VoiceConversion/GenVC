from inference.model_init import model_init
from inference.inference_utils import synthesize_utt_streaming_testflow, synthesize_utt_streaming
from utils import load_audio
import torch
import torchaudio
import argparse
import pyaudio
import time
import numpy as np
import queue

class StreamingBuffer:
    def __init__(self, model, device, ref_audio, args):
        self.model = model
        self.device = device
        self.p = pyaudio.PyAudio()
        self.chunk = 1600 # 이걸 올리면 퀄리티가 올라갈거
        if(args.test): self.chunk = 16000
        self.ref_audio = ref_audio
        self.ref_audio = self.ref_audio.to(self.device)
        self.cond_latent = model.get_gpt_cond_latents(self.ref_audio, model.config.audio.sample_rate)
        self.context_buffer = []
        self.FUTURE_CHUNK = 2
        if(args.test): self.FUTURE_CHUNK = 0
        input_info = self.p.get_default_input_device_info()
        input_rate = 24000 
        output_rate = 24000


        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()


        def input_callback(in_data, frame_count, time_info, status):
            self.input_queue.put(in_data) # 들어온 오디오를 큐에 쌓음
            return (None, pyaudio.paContinue)

        def output_callback(in_data, frame_count, time_info, status):
            try:
                data = self.output_queue.get_nowait()
            except queue.Empty:
                data = b'\x00' * frame_count * 4 
            return (data, pyaudio.paContinue)

        if(args.test and args.streaming):
            #test, streaming은 파일 읽기
            pass
        else:
            self.input_stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=input_rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=input_callback
            )

        self.output_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=output_rate,
            output=True,
            frames_per_buffer=self.chunk,
            stream_callback=output_callback
        )

    def start(self):
        print("곧 시작합니다")

        # --- 상태 변수 ---
        past_key_values = None 
        global_pos = 0
        last_audio_token = None 
        # ---------------

        self.input_stream.start_stream()
        self.output_stream.start_stream()
        try:
            while True:
                in_data = self.input_queue.get()
                
                input_np = np.frombuffer(in_data, dtype=np.float32).copy()
                input_np *= 15 # 소리가 너무 작아서 올렸는데 귀터짐 주의

                # -1.0 ~ 1.0 사이로 클리핑 (Contribution), 15배는 너무 커질 수 있음 
                input_np = np.clip(input_np, -1.0, 1.0) 

                current_tensor = torch.from_numpy(input_np).to(self.device).unsqueeze(0) 
                
                # 버퍼에 있는 데이터 + 현재 데이터
                padding_size = (self.FUTURE_CHUNK - len(self.context_buffer)) * self.chunk
                if self.context_buffer:
                    input_tensor = torch.cat(self.context_buffer + [current_tensor], dim=1)
                else:
                    input_tensor = current_tensor
                input_tensor = torch.nn.functional.pad(input_tensor, (padding_size, 0), "constant", 0)
                
                self.context_buffer.append(current_tensor)
                if len(self.context_buffer) > self.FUTURE_CHUNK:
                    self.context_buffer.pop(0)

                # 이거지우면 변환함수 돌아감
                if(args.test): 
                    converted_tensor = current_tensor
                else:
                    with torch.no_grad():
                        converted_tensor, past_key_values, last_audio_token, global_pos = synthesize_utt_streaming_testflow(
                        self.model, 
                        input_tensor, # [1,1,chunksize * 3]
                        self.cond_latent,
                        self.chunk,
                        past_key_values,
                        global_pos,
                        last_audio_token
                        )

                    if(converted_tensor is None):
                        continue 

                output_np = converted_tensor.squeeze().cpu().detach().numpy().astype(np.float32)
                self.output_queue.put(output_np.tobytes())

        except KeyboardInterrupt:
            # 닫기
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.p.terminate()

    def start_for_nonstreaming(self, src_wav):
        # 파일 읽고 변환
        self.output_stream.start_stream()
        src_wav = src_wav.to(self.device)
        for i in range(0, src_wav.shape[1], self.chunk):
            self.input_queue.put(src_wav[:, i:i + self.chunk])
            
        try:
            while True:
                current_tensor = self.input_queue.get()
                if current_tensor.shape[1] == 0 or self.input_queue.empty():
                    time.sleep(5)
                    break

                # 버퍼에 있는 데이터 + 현재 데이터
                padding_size = (self.FUTURE_CHUNK - len(self.context_buffer)) * self.chunk
                if self.context_buffer:
                    input_tensor = torch.cat(self.context_buffer + [current_tensor], dim=1)
                else:
                    input_tensor = current_tensor
                input_tensor = torch.nn.functional.pad(input_tensor, (padding_size, 0), "constant", 0)
                
                self.context_buffer.append(current_tensor)
                if len(self.context_buffer) > self.FUTURE_CHUNK:
                    self.context_buffer.pop(0)
                    
                with torch.no_grad(): 
                    converted_tensor = synthesize_utt_streaming_v2(self.model, input_tensor, self.cond_latent)
                    if(converted_tensor is None):
                        continue

                output_np = converted_tensor.squeeze().cpu().detach().numpy().astype(np.float32)
                self.output_queue.put(output_np.tobytes())

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
    parser.add_argument('--src_wav', type=str, default='samples/EM1_ENG_0037_1.wav')
    parser.add_argument('--ref_audio', type=str, default='samples/EF4_ENG_0112_1.wav')
    parser.add_argument('--output_path', type=str, default='samples/converted.wav')
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--streaming', type=str, default='0')
    parser.add_argument('--test', type=str, default='0')

    args = parser.parse_args()
    args.streaming = args.streaming == '1'
    args.test = args.test == '1'
    model, config = model_init(args.model_path, args.device)

    # top_k is one of the important hyperparameters for inference, so you can tune it to get better results
    # for streaming inference, greedy decoding is preferred, you can set top_k to 1
    model.config.top_k = args.top_k
    src_wav = load_audio(args.src_wav, model.content_sample_rate)
    ref_audio = load_audio(args.ref_audio, model.config.audio.sample_rate)

    if args.streaming:
        print(args.streaming)
        streaming = StreamingBuffer(model, args.device, ref_audio, args)
        streaming.start()
    else:
        if(args.test):
            # 음성파일 읽고 StreamingBuffer 수행
            non_streaming = StreamingBuffer(model, args.device, ref_audio, args)
            non_streaming.start_for_nonstreaming(src_wav)
        else:
            pre_audio = synthesize_utt_streaming(model, src_wav, ref_audio) 
            torchaudio.save(args.output_path, pre_audio.unsqueeze(0).detach().cpu(), config.audio.sample_rate)