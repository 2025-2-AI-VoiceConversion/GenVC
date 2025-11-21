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

# Options
BUFFER_DURATION_SECONDS = 2  # Audio Buffer (3 Second) 
CHANNELS = 1                 # Mono Channel

CHUNK_SIZE = 2
args = None 
cond_latent = None 
context_wav = None # Sliding Window 
MAX_BUFFER_LEN = None
model = None
config = None
SAMPLE_RATE = None
BLOCKSIZE = None 

# Output buffer
output_buffer = None # numpy array
output_buffer_lock = threading.Lock()  # buffer sync 

q_in = None

# Control Thread Flag
stop_event = threading.Event()
 
# Test Mode Only 
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

def processing_thread_func():
    """
    Input Q - Audio -> Conversion -> Output Q
    """
    
    global context_wav, output_buffer 
    
    print("-- Start Audio Processing Thread --")
    
    while not stop_event.is_set():
        try:
            indata = q_in.get(timeout=1)
            # to tensor (shape: [frames] -> [1, frames])
            indata_tensor = torch.from_numpy(indata).to(args.device)
            
            if indata_tensor.dim() == 2:
                # [frames, channels] -> [1, frames] (Mono Channel squeeze)
                indata_tensor = indata_tensor.squeeze(-1).unsqueeze(0)  
            elif indata_tensor.dim() == 1:
                # [frames] -> [1, frames] 
                indata_tensor = indata_tensor.unsqueeze(0)
            
            context_wav = torch.cat([context_wav, indata_tensor], dim=1) 
            # Overflow Cut: Keep only the most recent MAX_BUFFER_LEN samples
            # (Remove old data, keep recent data for sliding window)
            if context_wav.shape[1] > MAX_BUFFER_LEN:
                context_wav = context_wav[:, -MAX_BUFFER_LEN:] 
            
            # test mode
            if args.mode == 'test': 
                processed_data = dummy_voice_changer_model(indata_tensor)
                # tensor -> numpy
                processed_data = processed_data.detach().cpu().numpy()
                # [1, frames] -> [frames, 1]
                processed_data = processed_data.T  
            
            # default mode 
            else: 
                processed_data = synthesize_utt_streaming_mic(genVC_mdl=model, 
                                                              src_wav=indata_tensor, 
                                                              cond_latent=cond_latent,
                                                              stream_chunk_size=8 * CHUNK_SIZE)
            
                if isinstance(processed_data, list):
                    if len(processed_data) > 0:
                        processed_data = torch.cat(processed_data, dim=0)
                    else:
                        processed_data = torch.zeros(0)
                
                # tensor -> numpy, [frames] -> [frames, 1]
                if isinstance(processed_data, torch.Tensor):
                    processed_data = processed_data.detach().cpu().numpy() 
                
                # [frames] -> [frames, 1]
                if processed_data.ndim == 1:
                    processed_data = processed_data[:, np.newaxis]
                elif processed_data.ndim == 2 and processed_data.shape[0] == 1:
                    processed_data = processed_data.T  # [1, frames] -> [frames, 1]
                    
            # generated audio -> output buffer 
            with output_buffer_lock:
                if output_buffer is None or output_buffer.shape[0] == 0:
                    output_buffer = processed_data.copy()
                else:
                    output_buffer = np.vstack([output_buffer, processed_data])
            
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
    except queue.Full:
        # Drop Audio Frame
        print("Warning: Input queue is full. Dropping audio frame.")

    # Consumer : Output Buffer -> Speaker (Output block size)
    with output_buffer_lock: 
        if output_buffer is not None and output_buffer.shape[0] >= BLOCKSIZE:
            outdata[:] = output_buffer[:BLOCKSIZE]
            output_buffer = output_buffer[BLOCKSIZE:]
            
        elif output_buffer is not None and output_buffer.shape[0] > 0:
            remaining = output_buffer.shape[0]
            outdata[:remaining] = output_buffer
            outdata[remaining:] = 0  # zero padding
            output_buffer = np.zeros((0, 1)) 

def init():
    
    global model, config, SAMPLE_RATE, BLOCKSIZE, MAX_BUFFER_LEN, q_in, cond_latent, context_wav, output_buffer 
    
    # Model Loading
    print('-- Model Loading ... --')
    model, config = model_init(args.model_path, args.device)
    model.config.top_k = args.top_k
    
    # init 
    SAMPLE_RATE = model.content_sample_rate
    SAMPLE_RATE = 24000
    BLOCKSIZE = int(model.content_sample_rate * 0.34 * CHUNK_SIZE)
    MAX_BUFFER_LEN = int(BUFFER_DURATION_SECONDS * model.content_sample_rate)
    
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
