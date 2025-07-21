import torch
from torch import nn
    
class BiCodecTokenizer(nn.Module):
    def __init__(self, bicodec_model):
        super().__init__()
        self.encoder = bicodec_model.encoder
        self.quantizer = bicodec_model.quantizer
        self.speaker_encoder = bicodec_model.speaker_encoder

    def forward(self, feat, mel):
        # semantic tokens
        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
    
        # global tokens
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))

        return semantic_tokens, global_tokens

class BiCodecDetokenizer(nn.Module):
    def __init__(self, bicodec_model):
        super().__init__()
        self.quantizer = bicodec_model.quantizer
        self.speaker_encoder = bicodec_model.speaker_encoder
        self.prenet = bicodec_model.prenet
        self.decoder = bicodec_model.decoder

    def forward(self, semantic_tokens, global_tokens):
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon
