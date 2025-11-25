import torch
import time

@torch.inference_mode()
def handle_chunks(wav_gen, wav_gen_prev, wav_overlap, overlap_len=1024):
    """Handle chunk formatting in streaming mode"""
    wav_chunk = wav_gen[:-overlap_len]
    if wav_overlap is not None:
        # cross fade the overlap section
        if overlap_len > len(wav_chunk):
            wav_chunk = wav_gen[-overlap_len:]
            return wav_chunk, wav_gen, None
        else:
            crossfade_wav = wav_chunk[:overlap_len]
            crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
            wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
            wav_chunk[:overlap_len] += crossfade_wav

    wav_overlap = wav_gen[-overlap_len:]
    wav_gen_prev = wav_gen
    return wav_chunk, wav_gen_prev, wav_overlap

@torch.inference_mode()
def synthesize_utt(
    genVC_mdl, 
    src_wav, 
    tgt_audio, 
    seg_len=6.0):
    """Synthesize audio in chunks, used for non-streaming mode
    The concatenation is performed at the latent feature level"""
    wav_gen_prev, wav_overlap = None, None
    total_wavlen = src_wav.shape[-1]
    pred_audios = []
    min_chunk_duration = int(0.32 * genVC_mdl.content_sample_rate)

    src_wav = src_wav.to(genVC_mdl.device)
    seg_len = int(seg_len * genVC_mdl.content_sample_rate)
    # get the conditioning latent
    tgt_audio = tgt_audio.to(genVC_mdl.device)
    cond_latent = genVC_mdl.get_gpt_cond_latents(tgt_audio, genVC_mdl.config.audio.sample_rate)
    final_latents = []

    for i in range(0, total_wavlen, seg_len):
        seg_end = i+seg_len if i+seg_len < total_wavlen else total_wavlen
        if seg_end == total_wavlen:
            src_wav_seg = src_wav[:, i:]
            if src_wav_seg.shape[-1] < min_chunk_duration:
                src_wav_seg = torch.nn.functional.pad(src_wav_seg, (0, min_chunk_duration-src_wav_seg.shape[-1]), "constant", 0)
        else:
            src_wav_seg = src_wav[:, i:i+seg_len]

        content_feat = genVC_mdl.content_extractor.extract_content_features(src_wav_seg)
        content_codes = genVC_mdl.content_dvae.get_codebook_indices(content_feat.transpose(1, 2))
        
        gen_codes = genVC_mdl.gpt.generate(
            cond_latent,
            content_codes,
            do_sample=True,
            top_p=genVC_mdl.config.top_p,
            top_k=genVC_mdl.config.top_k,
            temperature=genVC_mdl.config.temperature,
            num_beams=1,
            length_penalty=genVC_mdl.config.length_penalty,
            repetition_penalty=genVC_mdl.config.repetition_penalty,
            output_attentions=False,
        )[0]

        gen_codes = gen_codes[(gen_codes!=genVC_mdl.gpt.stop_audio_token).nonzero().squeeze()]
        expected_output_len = torch.tensor([gen_codes.shape[-1] * genVC_mdl.config.model_args.gpt_code_stride_len], device=genVC_mdl.device)
        content_len = torch.tensor([content_codes.shape[-1]], device=genVC_mdl.device)
        acoustic_latents = genVC_mdl.gpt(content_codes,
                                    content_len,
                                    gen_codes.unsqueeze(0),
                                    expected_output_len,
                                    cond_latents=cond_latent,
                                    return_latent=True)
        final_latents.append(acoustic_latents)
    
    # concatenate the latents
    final_latents = torch.cat(final_latents, dim=1)
    mel_input = torch.nn.functional.interpolate(
        final_latents.transpose(1, 2),
        scale_factor=[genVC_mdl.hifigan_scale_factor],
        mode="linear",
    ).squeeze(1)

    synthesized_audio = genVC_mdl.hifigan(mel_input)

    return synthesized_audio[0].squeeze()

@torch.inference_mode()
def synthesize_utt_chunked(
    genVC_mdl, 
    src_wav, 
    tgt_audio, 
    seg_len=6.0):
    """Synthesize audio in chunks, used for non-streaming mode
    The concatenation is performed at the waveform level"""
    wav_gen_prev, wav_overlap = None, None
    total_wavlen = src_wav.shape[-1]
    pred_audios = []
    min_chunk_duration = int(0.32 * genVC_mdl.content_sample_rate)

    src_wav = src_wav.to(genVC_mdl.device)
    seg_len = int(seg_len * genVC_mdl.content_sample_rate)
    # get the conditioning latent
    tgt_audio = tgt_audio.to(genVC_mdl.device)
    cond_latent = genVC_mdl.get_gpt_cond_latents(tgt_audio, genVC_mdl.config.audio.sample_rate)

    for i in range(0, total_wavlen, seg_len):
        seg_end = i+seg_len if i+seg_len < total_wavlen else total_wavlen
        if seg_end == total_wavlen:
            src_wav_seg = src_wav[:, i:]
            if src_wav_seg.shape[-1] < min_chunk_duration:
                src_wav_seg = torch.nn.functional.pad(src_wav_seg, (0, min_chunk_duration-src_wav_seg.shape[-1]), "constant", 0)
        else:
            src_wav_seg = src_wav[:, i:i+seg_len]
        audio_pred = genVC_mdl.inference(
            src_wav_seg, 
            cond_latent,
            top_p=genVC_mdl.config.top_p,
            top_k=genVC_mdl.config.top_k,
            temperature=genVC_mdl.config.temperature,
            length_penalty=genVC_mdl.config.length_penalty,
            repetition_penalty=genVC_mdl.config.repetition_penalty)
        
        wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
            audio_pred.squeeze(), wav_gen_prev, wav_overlap, 1024)
        pred_audios.append(wav_chunk)
    
    synthesized_audio = torch.cat(pred_audios, dim=-1)

    return synthesized_audio

@torch.inference_mode() # Do not modify
def synthesize_utt_streaming(
    genVC_mdl, 
    src_wav, 
    tgt_audio,
    seg_len=6.0,
    stream_chunk_size=8):

    wav_gen_prev, wav_overlap = None, None
    
    total_wavlen = src_wav.shape[-1]
    pred_audios = []
    min_chunk_duration = int(0.32 * genVC_mdl.content_sample_rate)

    begin_time = time.time()

    src_wav = src_wav.to(genVC_mdl.device)
    seg_len = int(seg_len * genVC_mdl.content_sample_rate)
    # get the conditioning latent
    tgt_audio = tgt_audio.to(genVC_mdl.device)
    cond_latent = genVC_mdl.get_gpt_cond_latents(tgt_audio, genVC_mdl.config.audio.sample_rate)
    is_begin = True
    for i in range(0, total_wavlen, seg_len):
        seg_end = i+seg_len if i+seg_len < total_wavlen else total_wavlen
        if seg_end == total_wavlen:
            src_wav_seg = src_wav[:, i:]
            if src_wav_seg.shape[-1] < min_chunk_duration:
                src_wav_seg = torch.nn.functional.pad(src_wav_seg, (0, min_chunk_duration-src_wav_seg.shape[-1]), "constant", 0)
        else:
            src_wav_seg = src_wav[:, i:i+seg_len]

        content_feat = genVC_mdl.content_extractor.extract_content_features(src_wav_seg)
        content_codes = genVC_mdl.content_dvae.get_codebook_indices(content_feat.transpose(1, 2))
        gpt_inputs = genVC_mdl.gpt.compute_embeddings(cond_latent, content_codes)

        gpt_generator = genVC_mdl.gpt.get_generator(
            fake_inputs=gpt_inputs,
            top_p=genVC_mdl.config.top_p,
            top_k=genVC_mdl.config.top_k,
            temperature=genVC_mdl.config.temperature,
            length_penalty=genVC_mdl.config.length_penalty,
            repetition_penalty=genVC_mdl.config.repetition_penalty,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1,
            output_attentions=False,
            output_hidden_states=True,
        )

        last_tokens = []
        all_latents = []
        is_end = False
        while not is_end:
            try:
                x, latent = next(gpt_generator)
                last_tokens += [x]
                all_latents += [latent]
            except StopIteration:
                is_end = True

            if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                acoustic_latents = torch.cat(all_latents, dim=0)[None, :]
                mel_input = torch.nn.functional.interpolate(
                    acoustic_latents.transpose(1, 2),
                    scale_factor=[genVC_mdl.hifigan_scale_factor],
                    mode="linear",
                ).squeeze(1)
                audio_pred = genVC_mdl.hifigan.forward(mel_input)
                wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
                    audio_pred.squeeze(), wav_gen_prev, wav_overlap, 1024)
                pred_audios.append(wav_chunk)
                last_tokens = []
                all_latents = []
                if is_begin:
                    is_begin = False
                    latency = time.time() - begin_time
                    print(f"Latency: {latency:.3f}s")
    
    synthesized_audio = torch.cat(pred_audios, dim=-1)
    processed_time = time.time() - begin_time
    real_time_factor = processed_time / (total_wavlen / genVC_mdl.content_sample_rate)
    print(f"Real-time factor: {real_time_factor:.3f}")
    return synthesized_audio


@torch.inference_mode()
def synthesize_utt_streaming_v2(
    genVC_mdl, 
    src_wav, 
    cond_latent,
    seg_len=6.0,
    stream_chunk_size=8):

    wav_gen_prev, wav_overlap = None, None
    
    total_wavlen = src_wav.shape[-1]
    pred_audios = []
    min_chunk_duration = int(0.32 * genVC_mdl.content_sample_rate)

    begin_time = time.time()

    src_wav = src_wav.to(genVC_mdl.device)
    seg_len = int(seg_len * genVC_mdl.content_sample_rate)
    is_begin = True
    for i in range(0, total_wavlen, seg_len):
        seg_end = i+seg_len if i+seg_len < total_wavlen else total_wavlen
        if seg_end == total_wavlen:
            src_wav_seg = src_wav[:, i:]
            if src_wav_seg.shape[-1] < min_chunk_duration:
                src_wav_seg = torch.nn.functional.pad(src_wav_seg, (0, min_chunk_duration-src_wav_seg.shape[-1]), "constant", 0)
        else:
            src_wav_seg = src_wav[:, i:i+seg_len]

        content_feat = genVC_mdl.content_extractor.extract_content_features(src_wav_seg)
        content_codes = genVC_mdl.content_dvae.get_codebook_indices(content_feat.transpose(1, 2))
        gpt_inputs = genVC_mdl.gpt.compute_embeddings(cond_latent, content_codes)

        gpt_generator = genVC_mdl.gpt.get_generator(
            fake_inputs=gpt_inputs,
            top_p=genVC_mdl.config.top_p,
            top_k=genVC_mdl.config.top_k,
            temperature=genVC_mdl.config.temperature,
            length_penalty=genVC_mdl.config.length_penalty,
            repetition_penalty=genVC_mdl.config.repetition_penalty,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1,
            output_attentions=False,
            output_hidden_states=True,
        )

        last_tokens = []
        all_latents = []
        is_end = False
        while not is_end:
            try:
                x, latent = next(gpt_generator)
                last_tokens += [x]
                all_latents += [latent]
            except StopIteration:
                is_end = True

            if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                acoustic_latents = torch.cat(all_latents, dim=0)[None, :]
                mel_input = torch.nn.functional.interpolate(
                    acoustic_latents.transpose(1, 2),
                    scale_factor=[genVC_mdl.hifigan_scale_factor],
                    mode="linear",
                ).squeeze(1)
                audio_pred = genVC_mdl.hifigan.forward(mel_input)
                wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
                    audio_pred.squeeze(), wav_gen_prev, wav_overlap, 1024)
                pred_audios.append(wav_chunk)
                last_tokens = []
                all_latents = []
                if is_begin:
                    is_begin = False
                    latency = time.time() - begin_time
                    print(f"Latency: {latency:.3f}s")
    
    synthesized_audio = torch.cat(pred_audios, dim=-1)
    processed_time = time.time() - begin_time
    real_time_factor = processed_time / (total_wavlen / genVC_mdl.content_sample_rate)
    print(f"Real-time factor: {real_time_factor:.3f}")
    return synthesized_audio

@torch.inference_mode() # Now use 
def synthesize_utt_streaming_mic(
    genVC_mdl, 
    content_token_sequence, # 최대 6초 분량의 context token sequence 
    cond_latent, # 프롬프트 임베딩 : 외부에서 한번만 미리 계산 (전달만 받음)
    stream_chunk_size=1,
    num_token=25, # 한 청크가 몇개의 토큰을 담당하는지 설정 
    wav_gen_prev=None, 
    wav_overlap=None,
    ):

    pred_audios = [] 

    '''
    [동료 개발자 구현 영역]
    입력받은 src_content(전체 토큰 시퀀스)를 기반으로,
    가장 최근 청크(1초)에 해당하는 음성만을 생성하여 리턴해야 합니다.
    
    - Generator 상태 관리 (Caching)
    - Look-ahead / Look-behind 적용, Output Slicing [단 이부분은 보코더 조사 후 pysunn 구현 예정]
    등의 로직이 이곳에 구현됩니다.

    # 현재 사용하지 않음 
    min_chunk_duration = int(0.32 * genVC_mdl.content_sample_rate) # current not use
    ''' 
    
    # 지금 상태는 현재 1초랑 과거 5초의 토큰을 사용하고 있습니다. 

    begin_time = time.time()
    is_begin = True
    
    t_gpt_start = time.time()
    # 총 chunk_size 개 만큼 반복합니다. 만약 chunk_size = 2라면 두개의 청크에 대해서 처리합니다. 
    for i in range(0, stream_chunk_size): 
        
        # 임베딩 제작 : [화자 프롬프트, 내용 문맥, START_AUDIO] 
        # 슬라이딩 윈도우 어텐션 구현 없이, 이전에 만들었던 모든 임베딩을 전부 저장합니다. 
        gpt_inputs = genVC_mdl.gpt.compute_embeddings(cond_latent, content_token_sequence) 

        gpt_generator = genVC_mdl.gpt.get_generator( 
            fake_inputs=gpt_inputs,
            top_p=genVC_mdl.config.top_p,
            top_k=genVC_mdl.config.top_k,
            temperature=genVC_mdl.config.temperature,
            length_penalty=genVC_mdl.config.length_penalty,
            repetition_penalty=genVC_mdl.config.repetition_penalty,
            do_sample=True, # 이거 False로 하면 샘플 스트림 추론 안되더라 
            num_beams=1,
            num_return_sequences=1,
            output_attentions=False,
            output_hidden_states=True,
        )
        
        # 2. Generate Tokens
        all_latents = []
        last_tokens = []
        is_end = False

        t_gpt_start = time.time()
        while not is_end:
            try:
                x, latent = next(gpt_generator)
                last_tokens += [x]
                all_latents += [latent]
            except StopIteration:
                is_end = True
            
            # 8개의 음성 토큰을 GPT가 만들어 내면, 보코더로 음성 조각을 만들어 리턴한다 
            if is_end or (num_token > 0 and len(last_tokens) >= num_token):
                t_gpt_end = time.time()
                
                acoustic_latents = torch.cat(all_latents, dim=0)[None, :]
                mel_input = torch.nn.functional.interpolate(
                    acoustic_latents.transpose(1, 2),
                    scale_factor=[genVC_mdl.hifigan_scale_factor],
                    mode="linear",
                ).squeeze(1)
                audio_pred = genVC_mdl.hifigan.forward(mel_input)
            
                t_vocoder_end = time.time()
                print(f"   [Detail] GPT: {t_gpt_end - t_gpt_start:.3f}s | Vocoder: {t_vocoder_end - t_gpt_end:.3f}s")

                # 크로스 페이딩 안함 
                wav_chunk = audio_pred.squeeze()
                # Cross-Fading 적용 (일단은 나중에 생각) (지금은 청크 띡띡거림 있음.)
                #wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
                #   audio_pred.squeeze(), wav_gen_prev, wav_overlap, overlap_len=1024
                #)
                
                pred_audios.append(wav_chunk)
                
                # Speak
                last_tokens = []
                all_latents = []
                
                if is_begin:
                    is_begin = False
                    latency = time.time() - begin_time
                    print(f"Latency: {latency:.3f}s")
                    
                # 한 청크만 만들고 탈출 (스트리밍 루프 제어용)
                break
                    
    # 일단 지금 구현은 6초만큼 생성하고 바로 넘겨주는 쓰레기 구현임.. 
    return pred_audios

@torch.inference_mode() 
def synthesize_utt_streaming_testflow(
    genVC_mdl, 
    input_tensor,
    cond_latent, 
    chunk_size, 
    past_key_values=None, 
    global_pos = 0,
    last_audio_token=None,
    ):
    """
    Stateful Streaming Inference Function
    
    Args:
        genVC_mdl: GenVC model instance
        input_tensor: Audio input [1, 1, S] (Includes past context + current chunk)
        cond_latent: Speaker style embedding
        chunk_size: The size of the 'new' audio chunk (in samples) to generate
        past_key_values: KV Cache from previous step
        global_pos: Current absolute position index for positional embedding
        last_audio_token: The last generated audio token from previous step
    
    Returns:
        wav_chunk (audio tensor), past_key_values, last_audio_token, global_pos
    """ 
    
    # =========================================================================
    # 0. Constants & Timing Setup
    # =========================================================================
    import time # 딜레이 로깅용 
    
    timing_log = {}
    t_total_start = time.time()
    
    GPT_CODE_STRIDE = 1024 
    tokens_to_generate = int(chunk_size / GPT_CODE_STRIDE) # chunk size 는 1024 배수여야 함 
    
    # 청크 사이즈가 너무 작아서 토큰을 만들 수 없는 경우 (예외처리)
    if tokens_to_generate == 0:
        print("[Error] chunk size is too small")
        return None, past_key_values, last_audio_token, global_pos

    device = input_tensor.device
    gpt = genVC_mdl.gpt

    # =========================================================================
    # 1. Content Extraction [Audio Processing]
    # =========================================================================
    t1_start = time.time()
    
    # 1.1 Content Feature 추출 
    # Note: extract_content_features expects (batch, T) shape
    content_feat = genVC_mdl.content_extractor.extract_content_features(input_tensor)
    
    t1_feature = time.time()
    timing_log['1_feature_extraction'] = (t1_feature - t1_start) * 1000  # ms
    
    # 1.2 Content Code 추출 (DVAE)
    full_codes = genVC_mdl.content_dvae.get_codebook_indices(content_feat.transpose(1, 2))
    
    t1_dvae = time.time()
    timing_log['2_dvae_quantization'] = (t1_dvae - t1_feature) * 1000  # ms

    # 1.3 Content Code 개수 계산하기 
    '''
        If 컨텍스트가 꽉 찬 상태
        Else 아직 꽉 차지는 않은 상태 
        분기 나눠서 정확히 처리할 수 있어야함. 지금 구현 바보구현 
    '''
    
    # 1.4 이번 내용에만 딱 맞는 Content Code 슬라이싱
    # 3청크 입력 중 맨 뒤(현재)에 해당하는 토큰만 가져옴
    # 시간축 동기화를 위해 정확히 계산된 개수만큼 뒤에서 자름.
    target_content_tokens = full_codes[:, -tokens_to_generate:]
    
    # =========================================================================
    # 2. GPT 임베딩 준비 [Cond + Target_Content + Audio Prompt]
    # =========================================================================
    '''
    목표 : 기존 KV Cache 뒤에 새로운 Text 를 붙인다. 
    상황 : [Prompt ... Content A Audio A] + [Content B] 를 붙임 
    '''

    # 2-1-1. Content 임베딩
    txt_emb = gpt.text_embedding(target_content_tokens) # [B, T, Dim]

    # 2-1-2. Content Positional Embedding
    seq_len = target_content_tokens.shape[1]
    pos_ids_txt = torch.arange(global_pos, global_pos + seq_len, device=device) # 얘는 뭐임?
    
    # Positional Limit Clamping (학습된 길이를 초과하는 것을 방지) 
    max_pos_txt = gpt.text_pos_embedding.emb.num_embeddings # 얘는 뭐임?

    if((global_pos + seq_len) >= max_pos_txt):
                print(f"[WARNING] Text Positional Limit Reached! Current: {global_pos + seq_len}, Max: {max_pos_txt}")

    pos_ids_txt = torch.clamp(pos_ids_txt, max=max_pos_txt-1) # 얘는 뭐임? 

    txt_pos = gpt.text_pos_embedding.emb(pos_ids_txt).unsqueeze(0)
    emb_content = txt_emb + txt_pos

    # 2-2 Input Embedding 구성 

    # 2-2-1. 최초 실행 
    if past_key_values is None:
        inputs_embeds = torch.cat([cond_latent, emb_content], dim=1)
        # Start Token 초기화
        last_audio_token = torch.tensor([[gpt.start_audio_token]], device=device)
    else:
        # [스트리밍 중]: 이전 KV Cache 뒤에 이번 Content만 붙임 
        inputs_embeds = emb_content

    # 2-3. Forward (Text Prefill)
    t2_prefill_start = time.time()
    
    out = gpt.gpt(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
    past_key_values = out.past_key_values
    
    t2_prefill_end = time.time()
    timing_log['3_kv_prefill'] = (t2_prefill_end - t2_prefill_start) * 1000  # ms
    
    '''
    2.3 추가 설명 )

    Before KVCache : [스타일 + 옛날 내용 + 옛날 음향]
        [2.3 포워드 진행 후]
    After KVCache : [스타일 + 옛날 내용 + 옛날 음향 + 이번 텍스트 내용]
    '''
    
    # 2-4. 글로벌 커서 업데이트 (추가된 만큼) 
    global_pos += inputs_embeds.shape[1] 

    '''
    3. GPT_Forward [Audio Generation]
    목표: tokens_to_generate 만큼 오디오 토큰을 생성하면 된다. 
    '''

    # =========================================================================
    # 3. GPT_Forward [Audio Generation]
    # =========================================================================
    t3_gpt_start = time.time()
    
    curr_token = last_audio_token
    curr_pos = global_pos
    all_latents = [] # for Vocoder
    all_tokens = []
    # Mel Head의 Positional Embedding 한계 
    max_pos_mel = gpt.mel_pos_embedding.emb.num_embeddings

    # (Generation Loop) 
    for _ in range(tokens_to_generate):

        # 3.1.1 Mel Embedding 
        mel_emb = gpt.mel_embedding(curr_token)
        
        # 3.1.2. Positional Embedding 
        p_id = torch.tensor([curr_pos], device=device) # 이게뭔데
    
        if(curr_pos >= max_pos_mel):
            print(f"[WARNING] Mel Positional Limit Reached! Current: {curr_pos}, Max: {max_pos_mel}")

        p_id = torch.clamp(p_id, max=max_pos_mel-1) #이게 뭔데
        mel_pos = gpt.mel_pos_embedding.emb(p_id).unsqueeze(0) # 위치 임베딩 얻기 

        # 3.1.3. Mel Input Embedding 
        curr_input = mel_emb + mel_pos 
        
        # 3.2. GPT Forward (Next Token Prediction)

        # 이전 기억 past_key_values 와 현재 입력 curr_input 을 넣는다. 
        out = gpt.gpt(inputs_embeds=curr_input, past_key_values=past_key_values, use_cache=True)
        # 기억 업데이트 
        past_key_values = out.past_key_values

        # 3.3. Decode 
        hidden = gpt.final_norm(out.last_hidden_state) # [1,1,Dim] ? 
        logits = gpt.mel_head(hidden) # 히든에서 음성 헤드 꺼내기 
        # * gpt.text_head(hidden) 내용 헤드꺼내면 pseudo context 구현 가능할듯. 

        # =================================================================
        # [🛡️ Safety Net] 모델 멘탈 상태 점검 (Confidence & Entropy)
        # =================================================================
        
        # 1. 확률 분포 계산 (Softmax)
        probs = torch.nn.functional.softmax(logits, dim=-1) # [1, 1, Vocab]
        
        # 2. 주요 지표 추출
        # (1) 1등 토큰과 그 확신도(Confidence)
        top_prob, top_id = torch.max(probs, dim=-1)
        top_prob = top_prob.item() # 0.0 ~ 1.0
        
        # (2) Stop Token 확신도
        stop_id = getattr(gpt, 'stop_audio_token', 8195)
        stop_prob = probs[0, 0, stop_id].item()
        
        # (3) 엔트로피 (혼란도) 계산
        # P * log(P)의 합. 높을수록 혼란스러움.
        # 1e-9는 log(0) 방지용
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()

        print(f"Confidence: {top_prob:.4f}, Stop Confidence: {stop_prob:.4f}, Entropy: {entropy:.4f}")

        # stop token 방지 
        # stop_token_id = gpt.stop_audio_token
        # logits[:, :, stop_token_id] = -float('inf')

        # 3.4. Greedy Sampling 
        next_token = torch.argmax(logits, dim=-1) 
        all_tokens.append(next_token.item())

        # 3.5 Setup for Next Prediction 
        all_latents.append(hidden)
        curr_token = next_token
        curr_pos += 1

        # 3.6 Stop Check 필요 

        '''
        모델이 인터리빙을 잘 이해하지 못하고 바로 end_token을 뱉는 상황에 대한 예외 처리 가능성이 필요할 수 있음 
        ''' 
        if next_token.item() == gpt.stop_audio_token:
            print("End Token reached...")
            #TODO: 중단된 상태에서 토큰을 얼마나 많이 만들었었는지 출력 
            print(f"Generated {len(all_latents)}, {len(all_tokens)} tokens before end token. goal: {tokens_to_generate}")
            break
        
    t3_gpt_end = time.time()
    timing_log['4_gpt_generation'] = (t3_gpt_end - t3_gpt_start) * 1000  # ms
    timing_log['4_gpt_per_token'] = (t3_gpt_end - t3_gpt_start) * 1000 / max(len(all_latents), 1)  # ms/token
    
    last_audio_token = curr_token 
    # =========================================================================
    # 4. Sliding Window KVCache - 구현 해야함. 
    # =========================================================================
    # 캐시가 너무 커지면 OOM 방지를 위해 앞을 자름

    NUM_STYLE_TOKENS = cond_latent.shape[1] if cond_latent is not None else 0
    KEEP_RECENT_TOKENS = 100

    MAX_WINDOW = NUM_STYLE_TOKENS + KEEP_RECENT_TOKENS
    
    #TODO: layer_past shape 로깅으로 실제 검증 확인하기 
    if past_key_values is not None:
        # past_key_values[0]은 (Key, Value) 튜플임
        # Key Shape: (Batch, Num_Heads, Seq_Len, Head_Dim) -> Index 2가 Seq_Len
        current_seq_len = past_key_values[0][0].shape[2] 
        
        if current_seq_len > MAX_WINDOW:
            new_kv = []
            for layer_past in past_key_values: 
                # layer_past: (Key, Value)
                k, v = layer_past
                
                # 1. Key Pruning
                k_style = k[:, :, :NUM_STYLE_TOKENS, :]
                k_recent = k[:, :, -KEEP_RECENT_TOKENS:, :]
                k_pruned = torch.cat([k_style, k_recent], dim=2)
                
                # 2. Value Pruning
                v_style = v[:, :, :NUM_STYLE_TOKENS, :]
                v_recent = v[:, :, -KEEP_RECENT_TOKENS:, :]
                v_pruned = torch.cat([v_style, v_recent], dim=2)
                
                new_kv.append((k_pruned, v_pruned))
            
            past_key_values = tuple(new_kv)
        
    # =========================================================================
    # 5. Vocoding (HiFi-GAN)
    # =========================================================================
    t4_vocoder_start = time.time()

    if len(all_latents) == 0:
        print("Warning: No audio generated. Returning None.")
        return None, past_key_values, last_audio_token, global_pos

    # 5.1 Acoustic Latent 제작  
    acoustic_latents = torch.cat(all_latents, dim=1) # [B, tokens_to_generate, Dim]
    
    '''
    gpt_code_stride_len = 1024 로 음성 토큰 1개당 오디오 1024샘플이다.
    hop_length = 256 으로 하이파이갠 홉 사이즈는 256.
    즉 4배의 시간 해상도 차이가 존재한다. 
    1GPT token = 4 Mel frame 이다. 

    원본 GPT 토큰:    [A]             [B]                 [C]          [D]
                       ↓                ↓                ↓           ↓
    보간 후:        [A] [a1][a2][a3][B][b1][b2][b3][C][c1][c2][c3][D]...
                      └─────┘          └─────┘        └─────┘
                      4 frames         4 frames      4 frames
    '''
    # 5.2 선형 보간을 활용해서 Mel Input (to Vocoder)
    mel_input = torch.nn.functional.interpolate(
        acoustic_latents.transpose(1, 2),
        scale_factor=[genVC_mdl.hifigan_scale_factor],
        mode="linear",
    ).squeeze(1)
    
    # 5.3 Hifi-GAN 음성 합성 
    wav_chunk = genVC_mdl.hifigan.forward(mel_input).squeeze()
    
    t4_vocoder_end = time.time()
    timing_log['5_vocoding'] = (t4_vocoder_end - t4_vocoder_start) * 1000  # ms
    
    # =========================================================================
    # 6. Timing Summary
    # =========================================================================
    t_total_end = time.time()
    timing_log['total_time'] = (t_total_end - t_total_start) * 1000  # ms
    
    # 로그 출력 (Fast I/O)
    import sys
    
    # RTF 계산 (Real-Time Factor)
    audio_duration_ms = (chunk_size / 24000) * 1000  # 24kHz sample rate
    rtf = timing_log['total_time'] / audio_duration_ms
    
    # 한 번에 문자열 생성 후 출력 (버퍼링 최소화)
    log_output = (
        f"\n[⏱️  Timing Log - Chunk {chunk_size} samples]\n"
        f"  1️⃣  Feature Extraction:  {timing_log['1_feature_extraction']:6.2f} ms\n"
        f"  2️⃣  DVAE Quantization:   {timing_log['2_dvae_quantization']:6.2f} ms\n"
        f"  3️⃣  KV Cache Prefill:    {timing_log['3_kv_prefill']:6.2f} ms\n"
        f"  4️⃣  GPT Generation:      {timing_log['4_gpt_generation']:6.2f} ms ({timing_log['4_gpt_per_token']:.2f} ms/token)\n"
        f"  5️⃣  Vocoding (HiFiGAN):  {timing_log['5_vocoding']:6.2f} ms\n"
        f"  {'─'*50}\n"
        f"  🔥 Total:               {timing_log['total_time']:6.2f} ms\n"
        f"  📊 RTF (Real-Time Factor): {rtf:.3f}\n\n"
    )
    
    sys.stdout.write(log_output)
    sys.stdout.flush()
    
    
    # 5.5 Cross-Fading Overlap 구현
    '''
    wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
        wav_chunk, wav_gen_prev, wav_overlap, overlap_len=1024
    )
    '''             
    return wav_chunk, past_key_values, last_audio_token, global_pos 

