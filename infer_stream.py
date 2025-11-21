# Audio Processing
import sounddevice as sd
import numpy as np
import queue
import threading
import time

# Model
from inference.model_init import model_init
from inference.inference_utils import synthesize_utt_streaming_mic
from utils import load_audio
import argparse
import torch 

args = None 
cond_latent = None 
context_wav = None # Sliding Window [dtype : torch.Tensor]
context_vector = None # Sliding Window [type : SSL Vector] # 현재 안쓰임. 

MAX_BUFFER_LEN = None 
model = None
config = None
SAMPLE_RATE = None
BLOCKSIZE = None 

# Constants
stream_chunk_size = 1
num_token = None 
CHUNK_DURATION_SECONDS = 1
BUFFER_DURATION_SECONDS = 3  # Audio Buffer (6 Second) 
CHANNELS = 1                 # Mono Channel

# Output buffer
output_buffer = None # numpy array
output_buffer_lock = threading.Lock()  # buffer sync 
q_in = None

# Control Thread Flag
stop_event = threading.Event()

# Test Mode Only 
@torch.inference_mode()
def dummy_voice_changer_model(audio_tensor):
    """
    Dummy voice changer for test mode
    Input: torch.Tensor [1, frames]
    Output: torch.Tensor [1, frames]
    """
    time.sleep(0.034)  # 34ms Process
    
    # tensor -> numpy 
    audio_np = audio_tensor.detach().cpu().numpy()
    
    # Simple Voice Change 
    processed_np = np.roll(audio_np, shift=-200, axis=1)  
    
    # numpy -> tensor
    processed_tensor = torch.from_numpy(processed_np).to(audio_tensor.device)
    
    return processed_tensor 

@torch.inference_mode()
def processing_thread_func():
    """
    Input Q - Audio -> Conversion -> Output Q
    """
    
    global context_wav, output_buffer 
    
    print("-- Start Audio Processing Thread --")

    wav_gen_prev, wav_overlap = None, None
    
    while not stop_event.is_set():

        try:
            # --- Indata -> Tensor --- 
            indata = q_in.get(timeout=1)
            # to tensor (shape: [frames] -> [1, frames])
            indata_wav = torch.from_numpy(indata).to(args.device)
            
            if indata_wav.dim() == 2:
                # [frames, channels] -> [1, frames] (Mono Channel squeeze)
                indata_wav = indata_wav.squeeze(-1).unsqueeze(0)  
            elif indata_wav.dim() == 1:
                # [frames] -> [1, frames] 
                indata_wav = indata_wav.unsqueeze(0)
            
            # 1. Context Update (Concatenate)
            # 현재 들어온 indata음성 합체 
            context_wav = torch.cat([context_wav, indata_wav], dim=1) 
            
            # 2. Sliding Window (Truncate)
            # 최대 길이(6초)를 초과하면 앞부분을 잘라냄
            if context_wav.shape[1] > MAX_BUFFER_LEN: # MAX_BUFFER_LEN은 init에서 6초 분량으로 설정됨
                 context_wav = context_wav[:, -MAX_BUFFER_LEN:]
            
            # --- Test Mode ---
            if args.mode == 'test': 
                processed_data = dummy_voice_changer_model(indata_wav) 
                # tensor -> numpy
                processed_data = processed_data.detach().cpu().numpy()
                # [1, frames] -> [frames, 1]x`x`
                processed_data = processed_data.T  
            
            # --- Default Mode --- 
            else: 
                t0 = time.time()
                
                # 3. Feature Extraction (Full Context)
                full_content_feat = model.content_extractor.extract_content_features(context_wav)
                t1 = time.time()
                
                # 4. Tokenization (Full Context)
                full_content_codes = model.content_dvae.get_codebook_indices(full_content_feat.transpose(1, 2))
                t2 = time.time()
                
                # print(f'Full Context Codes Length: {full_content_codes.shape}')
                
                # 5. Inference (Streaming Function Call)
                processed_data = synthesize_utt_streaming_mic(
                    genVC_mdl=model, 
                    content_token_sequence=full_content_codes,  
                    cond_latent=cond_latent, 
                    stream_chunk_size=stream_chunk_size, 
                    num_token=num_token,
                    wav_gen_prev=wav_gen_prev,
                    wav_overlap=wav_overlap,
                )
                t3 = time.time()

                # Profiling Log 
                q_size = q_in.qsize()
                print(f"[Profile] Q:{q_size} | Feat: {t1-t0:.3f}s | DVAE: {t2-t1:.3f}s  | GPT+Vocoder: {t3-t2:.3f}s | Total: {t3-t0:.3f}s")

                # processed_data는 리스트 형태이므로 텐서로 변환 및 결합
                if processed_data:
                    if len(processed_data) > 0 and isinstance(processed_data[0], list):
                        import itertools
                        processed_data = [item for sublist in processed_data if sublist is not None for item in (sublist if isinstance(sublist, list) else [sublist])]
                    processed_data = [p.view(-1) if p.dim() == 0 else p for p in processed_data]

                    processed_data = torch.cat(processed_data, dim=0).unsqueeze(1).cpu().numpy()
                else:
                    continue             
                
                # ----- 결과 음성 Wav 로변환하는 부분 -----
                # tensor -> numpy, [frames] -> [frames, 1]
                if isinstance(processed_data, torch.Tensor):
                    processed_data = processed_data.detach().cpu().numpy() 
                    if processed_data.ndim == 1:
                         processed_data = processed_data[:, np.newaxis]
                
                # Logging
                duration_sec = processed_data.shape[0] / SAMPLE_RATE
                print(f"[Process] Generated Audio: {processed_data.shape} ({duration_sec:.2f}s)")

                # Slicing Logic (마지막 1초에 해당하는 음성 잘라내기)
                processed_data = processed_data[:, -SAMPLE_RATE:]
                duration_sec = processed_data.shape[0] / SAMPLE_RATE
                print(f"[Process] Sliced Audio: {processed_data.shape} ({duration_sec:.2f}s)")

                # 바로 출력 버퍼에 추가
                with output_buffer_lock: # Changed from 'buffer_lock' to 'output_buffer_lock' for consistency
                    if output_buffer is None or output_buffer.shape[0] == 0:
                        output_buffer = processed_data.copy()
                    else:
                        output_buffer = np.vstack([output_buffer, processed_data])
                
                # Debugging Log
                print(f"[Buffer] Added {processed_data.shape[0]} frames. Current Buffer: {len(output_buffer)}")
            
        except queue.Empty:
            continue
        
    print("-- Stop Audio Processing Thread --")

def audio_callback(indata, outdata, frames, time, status):
    """
    PortAudio - Realtime Callback Function 
    No Delay 
    """
    
    global output_buffer
    
    if status:
        print(status) 
    outdata.fill(0)

    # Producer : Mic -> Input Queue
    try:
        q_in.put_nowait(indata.copy()) 
    
    # 입력 큐가 꽉 찼음 
    except queue.Full:
        # Drop Audio Frame
        print("Warning: Input queue is full. Dropping audio frame.")

    # Consumer : Output Buffer -> Speaker (Output block size)
    with output_buffer_lock: 
        buffer_len = output_buffer.shape[0] if output_buffer is not None else 0
        
        # Logging (Every call)
        if hasattr(audio_callback, "call_count"):
            audio_callback.call_count += 1
        else:
            audio_callback.call_count = 0
            
        # 1초에 약 3번 호출됨 (0.34s 주기) -> 10번마다 로그 출력 (약 3.4초 주기)
        if audio_callback.call_count % 10 == 0 or buffer_len < BLOCKSIZE:
            # Visual Bar
            max_bar = 20
            fill_ratio = min(buffer_len / (BLOCKSIZE * 3), 1.0) # 기준을 3블록 정도로 잡음
            filled = int(max_bar * fill_ratio)
            bar = "█" * filled + "░" * (max_bar - filled)
            
            status = "OK"
            if buffer_len < BLOCKSIZE: status = "LOW (Underrun)"
            elif buffer_len > BLOCKSIZE * 5: status = "HIGH (Latency)"
            
            print(f"\r[Callback] Buffer: {buffer_len:5d} / Req: {BLOCKSIZE:5d} |{bar}| {status}", end="")
            if buffer_len < BLOCKSIZE: print() # 줄바꿈해서 경고 남기기

        if output_buffer is not None and output_buffer.shape[0] >= BLOCKSIZE:
            outdata[:] = output_buffer[:BLOCKSIZE]
            output_buffer = output_buffer[BLOCKSIZE:]
            
        elif output_buffer is not None and output_buffer.shape[0] > 0:
            remaining = output_buffer.shape[0]
            outdata[:remaining] = output_buffer
            outdata[remaining:] = 0  # zero padding
            output_buffer = np.zeros((0, 1)) 
        else:
            # Buffer Empty
            outdata.fill(0) 

def init():
    
    global model, config, SAMPLE_RATE, BLOCKSIZE, MAX_BUFFER_LEN, q_in, cond_latent, context_wav, output_buffer 
    
    # Model Loading
    print('-- Model Loading ... --')
    model, config = model_init(args.model_path, args.device)
    model.config.top_k = args.top_k
    
    # init 
    SAMPLE_RATE = model.content_sample_rate
    SAMPLE_RATE = 24000
    BLOCKSIZE = int(SAMPLE_RATE * CHUNK_DURATION_SECONDS * stream_chunk_size)
    MAX_BUFFER_LEN = int(BUFFER_DURATION_SECONDS * SAMPLE_RATE)
    
    # Calculate num_token automatically
    # 1 token covers 1024 samples (hop_size)
    # Add slight buffer (+2) to ensure we cover the full duration
    global num_token
    num_token = int((BLOCKSIZE / 1024)) + 2
    print(f"[Init] Chunk Duration: {CHUNK_DURATION_SECONDS}s | BlockSize: {BLOCKSIZE} | Calculated num_token: {num_token}")
    
    # Set Maxsize
    q_in = queue.Queue(maxsize=(SAMPLE_RATE * BUFFER_DURATION_SECONDS) // BLOCKSIZE) 
    
    # Prompt Audio
    ref_audio = load_audio(args.ref_audio, model.config.audio.sample_rate)
    # Get Prompt Latent
    cond_latent = model.get_gpt_cond_latents(ref_audio, model.config.audio.sample_rate)
    
    # context_wav (Empty Tensor, shape: [1, 0])
    context_wav = torch.zeros(1, 0).to(model.device)
    output_buffer = np.zeros((0, 1))  

if __name__ == "__main__":
    
    # args parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pre_trained/GenVC_small.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ref_audio', type=str, default='samples/EM1_ENG_0037_1.wav')
    parser.add_argument('--top_k', type=int, default=1) # greedy sampling (streaming)
    parser.add_argument('--mode', type=str, default='default') # mode : default / test 
    args = parser.parse_args()

    # init
    init()
    
    processing_thread = threading.Thread(target=processing_thread_func)
    processing_thread.start()

    print("[Start Real Time Voice Conversion] (Exit : Ctrl+C)")     
    
    try:
        # Block Size : SD Processing Unit at once
        # Channel : Mono 
        # Callback : Record
        with sd.Stream(samplerate=SAMPLE_RATE, 
                       blocksize=BLOCKSIZE,
                       channels=CHANNELS, callback=audio_callback):
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nProgram End...")
        stop_event.set()
        processing_thread.join()
