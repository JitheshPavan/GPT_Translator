
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class MultiHeadAttention(nn.Module):
  def __init__(self, h: int, d_k: int, d_v: int, d_model: int) -> None:
    """
    Initializes the MultiHeadAttention module.

    Args:
        h (int): Number of attention heads.
        d_k (int): Dimensionality of keys and queries.
        d_v (int): Dimensionality of values.
        d_model (int): Dimensionality of the model.
    """
    super().__init__()
    self.heads = h
    self.d_k = d_k
    self.d_v = d_v
    self.d_model = d_model

    # Linear transformations for queries, keys, values, and the output
    self.W_q = nn.Linear(d_model, d_k)
    self.W_k = nn.Linear(d_model, d_k)
    self.W_v = nn.Linear(d_model, d_v)
    self.W_o = nn.Linear(d_v, d_model)

  def forward( self,  queries: torch.Tensor, keys: torch.Tensor,
    values: torch.Tensor,  mask: Optional[torch.Tensor] = None ) -> torch.Tensor:
    """
    Performs the forward pass of the multi-head attention mechanism.

    Args:
        queries (torch.Tensor): Query tensor of shape (batch_size, seq_length, d_model).
        keys (torch.Tensor): Key tensor of shape (batch_size, seq_length, d_model).
        values (torch.Tensor): Value tensor of shape (batch_size, seq_length, d_model).
        mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, seq_length, seq_length) or None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
    """
    # 1. Linear projections for queries, keys, and values

    q = self.W_q(queries)
    k = self.W_k(keys)
    v = self.W_v(values)

    # 2. Reshape into multi-head format
    q = self.reshape_tensor(q)
    k = self.reshape_tensor(k)
    v = self.reshape_tensor(v)
    # 3. Compute scaled dot-product attention
    wei = self.attention(q, k, v, self.d_k, mask)

    # 4. Reshape back to original format

    wei = self.reshape_tensor(wei, reverse=True)

    # 5. Apply final linear transformation
    return self.W_o(wei)

  def reshape_tensor(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """
    Reshapes the tensor for multi-head attention computation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
        reverse (bool): If True, reshapes back to the original format.

    Returns:
        torch.Tensor: Reshaped tensor.

    Note: Transpose has to be applied here- to turn [B,T,H,h/d_k]==> [B,H,T,h/d_k].
          Because during the attention two dimensions has to participate, namely time/token dimension and embedding dimension (not the head dimension).
          So transpose is necessary here.
    """
    if not reverse:
      b, t, c = x.size()
      return x.view(b, t, self.heads, c // self.heads).transpose(1, 2)
    else:
      x = x.transpose(1, 2)
      return x.contiguous().view(x.shape[0], x.shape[1], self.d_v)

  def attention(self,  q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,d_k: int, mask: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
      """
      Computes the scaled dot-product attention.

      Args:
          q (torch.Tensor): Query tensor of shape (batch_size, heads, seq_length, d_k).
          k (torch.Tensor): Key tensor of shape (batch_size, heads, seq_length, d_k).
          v (torch.Tensor): Value tensor of shape (batch_size, heads, seq_length, d_v).
          d_k (int): Dimensionality of keys and queries.
          mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, seq_length, seq_length) or None.

      Returns:
          torch.Tensor: Output tensor of shape (batch_size, heads, seq_length, d_v).
      """
      wei = q @ k.transpose(-2, -1) * d_k**-0.5  # (B, H, T, D_k) @ (B, H, D_k, T) -> (B, H, T, T)
      if mask is not None:
          wei += -1e9 * mask  # Large negative values give zero for softmax,
      wei = F.softmax(wei, dim=-1)  # Normalize attention scores
      return wei @ v  # (B, H, T, T) @ (B, H, T, D_v) -> (B, H, T, D_v)
    
class Feed_Forward(nn.Module):
  def __init__(self, d_model : int, d_ff: int)-> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, d_ff),
      nn.ReLU(),
      nn.Linear(d_ff, d_model),
    )
  def forward(self,x:torch.Tensor)-> torch.Tensor:
    return self.net(x)

class PositionalEncoding(nn.Module):
  def __init__(self, model_dimension: int, expected_max_sequence_length: int)-> None:
    super().__init__()
    position_id = torch.arange(expected_max_sequence_length).unsqueeze(1)
    frequencies = 10000 ** (-torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

    # Precompute the positional encodings
    positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
    positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)
    positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)

    # Save the encodings as a non-trainable buffer
    self.register_buffer('positional_encodings_table', positional_encodings_table)

  def forward(self, embeddings_batch: torch.Tensor)-> torch.Tensor:

    assert embeddings_batch.shape[-1] == self.positional_encodings_table.size(1), \
      f"Model dimension mismatch: {embeddings_batch.shape[-1]} != {self.positional_encodings_table.size(1)}"

    # Select and return positional encodings matching the sequence length
    return self.positional_encodings_table[:embeddings_batch.size(1)]

from typing import Optional

class EncoderBlock(nn.Module):
  def __init__(self, h: int, d_k: int, d_v: int, d_model: int, rate: float, d_ff: int= d_ff) -> None:
    super().__init__()

    self.MHA = MultiHeadAttention(h, d_k, d_v, d_model)

    self.dropout1 = nn.Dropout(rate)
    self.dropout2 = nn.Dropout(rate)

    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)

    self.Feed_Forward = Feed_Forward(d_model, d_ff)

  def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None ) -> torch.Tensor:

    x_sideline = self.layernorm1(x)

    x = x + self.dropout1(self.MHA(x_sideline, x_sideline, x_sideline, padding_mask))

    x_sideline1 = self.layernorm2(x)  # mha -> dropout -> residual -> layer norm

    x = x + self.dropout2(self.Feed_Forward(x_sideline1))  # FFNN -> dropout -> residual -> layer norm

    return x

class Encoder(nn.Module):
  def __init__(self, vocab_size: int, max_length: int, d_model: int, h: int, d_k: int, d_v: int, d_ff: int, n_layers: int, rate: float) -> None:
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model, max_length)
    self.dropout = nn.Dropout(rate)

    self.layers = nn.ModuleList([EncoderBlock(h, d_k, d_v, d_model, rate) for _ in range(n_layers)])

  def forward(self, sentence: torch.Tensor, padding_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    x = self.embedding(sentence)
    x = x + self.positional_encoding(x)
    x = self.dropout(x)
    for layer in self.layers:
        x = layer(x, padding_mask)
    return x


from typing import Optional

class DecoderBlock(nn.Module):
  def __init__(self, h: int, d_k: int, d_v: int, d_model: int, d_ff: int, rate: float) -> None:
    super().__init__()
    self.MHA = MultiHeadAttention(h, d_k, d_v, d_model)
    self.MHA2 = MultiHeadAttention(h, d_k, d_v, d_model)

    # It is not necessary to define dropout layers mulitple times. It is stateless.
    self.dropout3 = nn.Dropout(rate)
    self.dropout2 = nn.Dropout(rate)
    self.dropout1 = nn.Dropout(rate)

    # Layernorm have learned parameters(Gamma and beta)
    self.add_norm = nn.LayerNorm(d_model)
    self.add_norm2 = nn.LayerNorm(d_model)
    self.add_norm3 = nn.LayerNorm(d_model)

    self.Feed_Forward = Feed_Forward(d_model, d_ff)

  def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, look_ahead_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

    x_sideline = self.add_norm(x)
    x = x + self.dropout1(self.MHA(x_sideline, x_sideline, x_sideline, look_ahead_mask))

    x_sideline1 = self.add_norm2(x)
# Encoder output is used because this is cross-attention. Padding mask is constructed from encoder input
    x = x + self.dropout2(self.MHA2(x_sideline1, encoder_output, encoder_output, padding_mask))


    x_sideline2 = self.add_norm3(x)
    x = x + self.dropout3(self.Feed_Forward(x_sideline2))
    return x


from typing import Optional

class Decoder(nn.Module):
  def __init__(self, vocab_size: int, max_length: int, d_model: int, h: int, d_k: int, d_v: int, d_ff: int, n_layers: int, rate: float) -> None:
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)

    self.positional_encoding = PositionalEncoding(d_model, max_length)
    self.dropout = nn.Dropout(rate)

    self.layers = nn.ModuleList([DecoderBlock(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n_layers)])

  def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor, lookahead_mask: Optional[torch.Tensor], padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    x = self.embedding(decoder_input)

    x = x + self.positional_encoding(x)
    x = self.dropout(x)

    for layer in self.layers:

        x = layer(x, encoder_output, padding_mask, lookahead_mask)
    return x

from typing import Optional, Union, Type

class Transformer(nn.Module):
  def __init__(self, enc_vocab_size: int, dec_vocab_size: int, enc_seq_len: int, dec_seq_len: int,
               d_model: int, h: int, d_k: int, d_v: int, d_ff: int, n_layers: int, rate: float) -> None:
    super().__init__()

    self.encoder = Encoder(enc_vocab_size, enc_seq_len, d_model, h, d_k, d_v, d_ff, n_layers, rate)
    self.decoder = Decoder(dec_vocab_size, dec_seq_len, d_model, h, d_k, d_v, d_ff, n_layers, rate)
    self.model_output = nn.Linear(d_model, dec_vocab_size)

    self.init_weights() # Xavier Initialization

  def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
    # Creating padding mask. This will remove padded tokens out of contibution during attention.
    padding_mask_enc = self.padding_mask(encoder_input)
    padding_mask_dec = self.padding_mask(decoder_input)

    lookahead_mask_dec = self.look_ahead_mask(decoder_input.shape[1])
    lookahead_mask_dec = torch.maximum(lookahead_mask_dec, padding_mask_dec)

    encoder_output = self.encoder(encoder_input, padding_mask_enc)
    decoder_output = self.decoder(decoder_input, encoder_output, lookahead_mask_dec, padding_mask_enc)
    return self.model_output(decoder_output)


  def padding_mask(self, x: torch.Tensor) -> torch.Tensor:
    return (x == 0).float().unsqueeze(1).unsqueeze(1) # Shape =[Batch,1,1,seq_length]. This will broadcast during attention to [Batch,heads,seq_length,seq_length].

  def look_ahead_mask(self, shape: int) -> torch.Tensor:
    x = torch.tril(torch.ones((shape, shape), device=device))
    return x.type(torch.float32)

  def init_weights(self) -> None:
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)


if __name__ == "__main__":
  # Define the model parameters
  h = 8  # Number of self-attention heads
  d_k = 64  # Dimensionality of the linearly projected queries and keys # The dimension is divided among the heads.  Thus every head key value will be d_k/h= 8
  d_v = 64  # Dimensionality of the linearly projected values
  d_model = 512  # Dimensionality of model layers' outputs
  d_ff = 2048  # Dimensionality of the inner fully connected layer # Usually 4* d_model
  n = 6  # Number of layers in the encoder stack
  
  #Training
  batch_size = 64
  beta_1 = 0.9
  beta_2 = 0.98
  epsilon = 1e-9
  dropout_rate = 0.1

  transformer = Transformer(enc_vocab_size,dec_vocab_size,enc_seq_len,dec_seq_len,d_model,h,d_k,d_v,d_ff,n,dropout_rate)
  
  # Dummy data
  encoder_input = torch.randint(0, enc_vocab_size, (1, enc_seq_len))
  decoder_input = torch.randint(0, dec_vocab_size, (1, dec_seq_len))
  
  # Forward pass
  output = transformer(encoder_input, decoder_input)
  print(output.shape)  # Should print (1, dec_seq_len, dec_vocab_size)
