import torch
from torch import nn

class BiCodecSemanticTokenizer(nn.Module):
    def __init__(self, bicodec_model):
        super().__init__()
        self.encoder = bicodec_model.encoder
        self.quantizer = bicodec_model.quantizer

    def forward(self, feat):
        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)

        return semantic_tokens

class BiCodecGlobalTokenizer(nn.Module):
    def __init__(self, bicodec_model):
        super().__init__()
        self.speaker_encoder = bicodec_model.speaker_encoder

    def forward(self, mel):
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))

        return global_tokens

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

class BiCodecSemanticDetokenizer(nn.Module):
    def __init__(self, bicodec_model):
        super().__init__()
        self.quantizer = bicodec_model.quantizer

    def forward(self, semantic_tokens):
        z_q = self.quantizer.detokenize(semantic_tokens)
        return z_q
    
class BiCodecGlobalDetokenizer(nn.Module):
    def __init__(self, bicodec_model):
        super().__init__()
        self.speaker_encoder = bicodec_model.speaker_encoder

    def forward(self, global_tokens):
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        return d_vector
    
class BiCodecPreNet(nn.Module):
    def __init__(self, bicodec_model):
        super().__init__()
        self.prenet = bicodec_model.prenet

    def forward(self, z_q, d_vector):
        x = self.prenet(z_q, d_vector)
        return x
    
class BiCodecVocoder(nn.Module):
    def __init__(self, bicodec_model):
        super().__init__()
        self.decoder = bicodec_model.decoder

    def forward(self, prenet_output, d_vector):
        x = prenet_output + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)
        return wav_recon
