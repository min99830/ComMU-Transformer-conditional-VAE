from typing import Optional, Any, Union, Callable, Tuple

import torch
import math
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoderLayer
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn import Embedding
from torch.nn import CrossEntropyLoss


# This model is from torch github https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
# I just edited some parts to use it as transformer_VAE
# Transformer VAE paper : https://ieeexplore.ieee.org/document/9054554
# notion : https://www.notion.so/binne/YAI-X-POZAlabs-852ef538af984d99abee33037751547c

class Transformer_CVAE(Module):
    """
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
    """

    def __init__(self, d_model: int = 512, d_latent: int = 64, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, pad_idx: int = 0, vocab_size: int = 729, seq_len: int = 256, cdt_len: int = 11, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer_CVAE, self).__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.sample_layer = VAEsample(d_model, d_latent, batch_first, **factory_kwargs)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.d_latent = d_latent
        self.seq_len = seq_len
        self.nhead = nhead
        self.batch_first = batch_first
        self.device = device
        self.cdt_len = cdt_len

        self.local_encoder = Linear(d_model, d_model, **factory_kwargs)
        self.local_decoder = Linear(d_model, vocab_size, **factory_kwargs)
        self.condition_embed = TokenEmbedding(d_latent, vocab_size=vocab_size, **factory_kwargs)
        self.src_tgt_embed = TransformerEmbedding(d_model, vocab_size, seq_len, batch_first=batch_first, **factory_kwargs)
        self.criterion = VAE_Loss(beta=1, pad_idx=0) # beta = 1 from Transformer VAE paper, pad_idx from ComMU dataset

    def forward_(self, src: Tensor, tgt: Tensor, cdt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the Tensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the Tensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the Tensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decoder.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> # xdoctest: +SKIP
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        encoder_out = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        latent_sample, latent_mu, latent_std = self.sample_layer(encoder_out, cdt)
        output = self.decoder(tgt, latent_sample, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return (output, latent_mu, latent_std)
    
    # src : word sequence
    # tgt : word sequence that starts with start token
    # cdt : condition sequence
    def forward(self, src: Tensor, tgt: Tensor, cdt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        src_tgt_mask = self.generate_square_subsequent_mask(self.seq_len, device=self.device)
        cross_attention_mask = self.generate_rectangle_subsequent_mask(self.seq_len + self.cdt_len, self.seq_len, device=self.device)
        
        src_embed = self.src_tgt_embed(src)
        tgt_embed = self.src_tgt_embed(tgt)
        cdt_embed = self.condition_embed(cdt)

        src_embed = self.local_encoder(src_embed)
        tgt_embed = self.local_encoder(tgt_embed)
        out, latent_mu, latent_std = self.forward_(src_embed, tgt_embed, cdt_embed, src_tgt_mask, src_tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        pred = F.log_softmax(self.local_decoder(out), dim = 2)
        loss = self.criterion(pred, src, latent_mu, latent_std)
        return (loss, out)


    @staticmethod
    def generate_square_subsequent_mask(sz: int, device='cpu') -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
    
    @staticmethod
    def generate_rectangle_subsequent_mask(xsz: int, ysz: int, device='cpu') -> Tensor:
        return torch.triu(torch.full((xsz, ysz), float('-inf'), device=device), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class VAEsample(Module):
    def __init__(self, d_model: int, d_latent: int, batch_first: bool, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(VAEsample, self).__init__()
        self.device = device
        self.d_latent = d_latent
        self.d_model = d_model
        if batch_first:
            self.seq_pos = 1
        else:
            self.seq_pos = 0
        self.model2sample = Linear(d_model, d_latent * 2, **factory_kwargs)
        self.latent2model = Linear(d_latent, d_model, **factory_kwargs)
    
    # input size : 
    # (batch_size, seq_length, d_model) if batch_first is True
    # (seq_length, batch_size, d_model) if batch_first is False
    def forward(self, x, cdt):
        lin_out = self.model2sample(x)
        latent_mean = lin_out[:, :, :self.d_latent]
        latent_std = torch.exp(lin_out[:, :, self.d_latent:] / 2)
        norm_sample = torch.normal(mean = 0, std = 1, size=latent_mean.size(), device=self.device)
        out = latent_mean + latent_std * norm_sample
        out = torch.concat((out, cdt), dim = self.seq_pos)
        out = self.latent2model(out)
        return out, latent_mean, latent_std


class TokenEmbedding(Module):
    def __init__(self, d_model, vocab_size, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TokenEmbedding, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, **factory_kwargs)
        self.d_model = d_model
    
    # input size : 
    # (batch_size, seq_length) if batch_first is True
    # (seq_length, batch_size) if batch_first is False
    def forward(self, x) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(Module):
    def __init__(self, d_model, max_len, batch_first: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PositionalEncoding, self).__init__()

        self.batch_first = batch_first
        encoding = torch.zeros(max_len, d_model)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            self.encoding = encoding.unsqueeze(0).to(device)
        else:
            self.encoding = encoding.unsqueeze(1).to(device)
    
    # input size : 
    # (batch_size, seq_length, d_model) if batch_first is True
    # (seq_length, batch_size, d_model) if batch_first is False
    def forward(self, x) -> Tensor:
        if self.batch_first : # input size : (batch_size, seq_length, d_model)
            seq_len = x.size(1)
            pos_embed = self.encoding[:, :seq_len, :]
        else :                # input size : (seq_length, batch_size, d_model)
            seq_len = x.size(0)
            pos_embed = self.encoding[:seq_len, :, :]
        return x + pos_embed

class TransformerEmbedding(Module):
    def __init__(self, d_model: int, vocab_size: int, max_len: int, batch_first: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEmbedding, self).__init__()
        self.embed = TokenEmbedding(d_model=d_model, vocab_size=vocab_size, **factory_kwargs)
        self.positional = PositionalEncoding(d_model=d_model, max_len=max_len, batch_first=batch_first, **factory_kwargs)
    
    # input size : 
    # (batch_size, seq_length) if batch_first is True
    # (seq_length, batch_size) if batch_first is False
    def forward(self, x) -> Tensor:
        return self.positional(self.embed(x))

class VAE_Loss(Module):
    def __init__(self, beta: int, pad_idx: int) -> None:
        super(VAE_Loss, self).__init__()
        self.beta = beta
        self.reconstruction_loss = CrossEntropyLoss(ignore_index=pad_idx)
    
    def forward(self, pred: Tensor, output: Tensor, latent_mu: Tensor, latent_std: Tensor) -> Tensor:
        kld_loss = - 0.5 * torch.sum(1 + 2 * torch.log(torch.abs(latent_mu)) - latent_mu.pow(2) - latent_std.pow(2))
        sz = output.size(0) * output.size(1)
        rec_loss = self.reconstruction_loss(pred.contiguous().view(sz, -1), output.contiguous().view(sz))

        return kld_loss / sz * self.beta + rec_loss

