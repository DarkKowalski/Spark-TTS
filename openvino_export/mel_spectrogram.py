# Originally implemented by https://github.com/arghyasur1991/Spark-TTS

import torch
from torch import nn
import math  # For pi
import torchaudio.functional as TF  # For melscale_fbanks and window functions

class MelSpectrogram(nn.Module):
    """
    A PyTorch module to compute Mel spectrograms from raw audio, designed for ONNX export.

    This module manually implements Short-Time Fourier Transform (STFT) and mel scaling
    to ensure ONNX compatibility.
    """
    def __init__(self, mel_params: dict, device: torch.device = torch.device("cpu")):
        super().__init__()
        
        self.n_fft = mel_params['n_fft']
        self.hop_length = mel_params.get('hop_length', self.n_fft // 4)
        self.win_length = mel_params.get('win_length', self.n_fft)
        self.sample_rate = mel_params['sample_rate']
        
        window_fn_name = mel_params.get('window_fn', 'hann_window')
        if window_fn_name == 'hann_window':
            window_tensor = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        elif window_fn_name == 'hamming_window':
            window_tensor = torch.hamming_window(self.win_length, periodic=True, dtype=torch.float32)
        else:
            print(f"[WARNING] Unrecognized window_fn '{window_fn_name}', defaulting to Hann window.")
            window_tensor = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        self.register_buffer('window', window_tensor.to(device)) # Ensure window is on the correct device

        self.center = mel_params.get('center', True)
        self.pad_mode = mel_params.get('pad_mode', "reflect") 
        self.power = mel_params.get('power', 1.0) 
        
        n_stft = self.n_fft // 2 + 1
        f_min = mel_params.get('mel_fmin', 0.0)
        f_max_param = mel_params.get('mel_fmax')
        f_max = f_max_param if f_max_param is not None else self.sample_rate / 2.0
            
        n_mels = mel_params['num_mels']
        mel_norm = mel_params.get('norm', 'slaney') 
        mel_scale_type = mel_params.get('mel_scale', 'slaney')

        mel_fbanks_tensor = TF.melscale_fbanks(
            n_freqs=n_stft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=self.sample_rate,
            norm=mel_norm,
            mel_scale=mel_scale_type
        )
        self.register_buffer('mel_fbanks', mel_fbanks_tensor.to(device)) # Ensure on device

        # Precompute RFFT matrices (real and imaginary parts)
        # These matrices are used to perform RFFT via matrix multiplication.
        k_range = torch.arange(0, self.n_fft // 2 + 1, dtype=torch.float32, device=device)
        n_range = torch.arange(0, self.n_fft, dtype=torch.float32, device=device)
        angle = -2 * math.pi * k_range.unsqueeze(1) * n_range.unsqueeze(0) / self.n_fft
        
        rfft_mat_real_tensor = torch.cos(angle)
        rfft_mat_imag_tensor = torch.sin(angle)
        # Store transposed versions for efficient matmul later: (n_fft, n_fft // 2 + 1)
        self.register_buffer('rfft_mat_real_t', rfft_mat_real_tensor.T)
        self.register_buffer('rfft_mat_imag_t', rfft_mat_imag_tensor.T)

    def forward(self, wav_with_channel: torch.Tensor) -> torch.Tensor:
            """
            Computes the Mel spectrogram from a batch of raw audio waveforms.

            Args:
                wav_with_channel (torch.Tensor): Input waveform tensor with shape (B, 1, T_audio).

            Returns:
                torch.Tensor: Mel spectrogram tensor with shape (B, n_mels, num_frames).
            """
            if wav_with_channel.ndim != 3 or wav_with_channel.shape[1] != 1:
                # This should ideally raise an error that propagates, or be handled before ONNX export if shapes are fixed
                print(f"[ERROR] Expected input shape (B, 1, T_audio), got {wav_with_channel.shape}")
                # For ONNX export, it's better to conform to dummy input shape or ensure model handles variability.
                # If this occurs during export with dummy_input, it's a setup error.
                raise ValueError(f"MelSpectrogramONNXWrapper: Invalid input shape. Expected (B, 1, T_audio), got {wav_with_channel.shape}")
            
            wav = wav_with_channel.squeeze(1) # Shape: (B, T_audio)
            batch_size = wav.shape[0]

            # 1. Padding (if center=True)
            padded_wav = wav
            if self.center:
                padding_amount = self.n_fft // 2
                padded_wav = torch.nn.functional.pad(wav, (padding_amount, padding_amount), mode=self.pad_mode)
            
            padded_sequence_length = padded_wav.shape[1]

            # 2. Framing using a loop with torch.narrow (to replace tensor.unfold)
            frame_list = []
            # Calculate the number of frames. Equivalent to: (L_padded - win_length) // hop_length + 1
            num_frames = (padded_sequence_length - self.win_length) // self.hop_length + 1
            
            for i in range(num_frames):
                start = i * self.hop_length
                frame = padded_wav.narrow(1, start, self.win_length)
                frame_list.append(frame)
            
            if not frame_list: # Handle case where input is too short for any frames
                # This case should ideally be handled by ensuring minimum input length or by defining
                # expected output for zero frames (e.g., empty tensor with correct dims).
                # For now, create an empty tensor that matches expected downstream dimensions if possible,
                # or raise an error. Let's assume we expect a valid mel output shape even if empty.
                # Output n_mels, 0 time steps
                return torch.empty((batch_size, self.mel_fbanks.shape[0], 0), device=wav.device, dtype=wav.dtype)

            frames = torch.stack(frame_list, dim=1)
            # frames shape: (B, num_frames, self.win_length)

            # 3. Windowing
            windowed_frames = frames * self.window 

            # 4. Pad windowed frames to n_fft for FFT if win_length < n_fft
            fft_ready_frames = windowed_frames
            if self.n_fft > self.win_length:
                pad_right = self.n_fft - self.win_length
                fft_ready_frames = torch.nn.functional.pad(windowed_frames, (0, pad_right), mode='constant', value=0)
            elif self.n_fft < self.win_length: 
                fft_ready_frames = windowed_frames[:, :, :self.n_fft]

            # 5. Manual RFFT using precomputed matrices
            real_part = torch.matmul(fft_ready_frames, self.rfft_mat_real_t)
            imag_part = torch.matmul(fft_ready_frames, self.rfft_mat_imag_t)

            # 6. Magnitude (Complex modulus)
            magnitude = torch.sqrt(real_part.pow(2) + imag_part.pow(2))

            # 7. Power Spectrum (if self.power is not 1.0, e.g., 2.0 for power)
            if self.power != 1.0:
                magnitude = magnitude.pow(self.power)

            # 8. Apply Mel Filterbank
            mel_output = torch.matmul(magnitude, self.mel_fbanks)

            # 9. Transpose to conventional (B, n_mels, num_frames)
            mel_output = mel_output.transpose(1, 2)
            
            return mel_output
