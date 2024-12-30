
import torch
import torch.nn as nn
import torch.nn.functional as F

"""# Model"""

#Hyperparameters
d_model= 512 # This is embedding dimension. n_embed
d_ff  = 4* d_model # Inner dimension of the FFNN layer. It is 2048 in the paper.
max_length= 50  # Max sequence length allowed.
# vocab_size =
#n_layers =
#rate =



class MultiHeadAttention(nn.Module):
  def __init__(self, h, d_k,d_v,d_model):
    super().__init__()
    self.heads= h
    self.d_k = d_k # dimensionality of k,q
    self.d_v = d_v # dimensionality of Values
    self.d_model = d_model # Model dimension ( 512)
    self.W_q  = nn.Linear(d_model, d_k) #
    self.W_k  = nn.Linear(d_model, d_k)
    self.W_v  = nn.Linear(d_model, d_v)
    self.W_o  = nn.Linear(d_v, d_model)
    # Mask matrix. We make it untrainable. Register buffer will save this matrix too during torch.save()
    # self.register_buffer('tril', torch.tril(torch.ones(max_length, max_length)))

  def forward(self, queries, keys, values,mask=None):
    #1
    q= self.W_q(queries)
    k= self.W_k(keys)
    v= self.W_v(values)
    #2
    q= self.reshape_tensor(q)
    k= self.reshape_tensor(k)
    v= self.reshape_tensor(v)
    #3
    wei = self.attention(q,k,v,self.d_k,mask)
    #4
    wei = self.reshape_tensor(wei,reverse=True)
    return self.W_o(wei)
  def reshape_tensor(self,x,reverse=False):
    if not reverse:
      b,t,c = x.size()
      return x.view(b,t,self.heads,c//self.heads).transpose(1,2)
    else:
      x= x.transpose(1,2)
      return x.contiguous().view(x.shape[0],x.shape[1],self.d_v)
# Attention mechanism
  def attention(self, q,k,v,d_k,mask=None):
    wei = q @ k.transpose(-2,-1) * d_k**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
    if mask is not None:
      wei += -1e9 *mask
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    return wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)


class Feed_Forward(nn.Module):
  def __init__(self, d_model, d_ff):
    super().__init__()
    self.net = nn.Sequential(
    nn.Linear(d_model, 4 * d_model),
    nn.ReLU(),
    nn.Linear(4 * d_model,d_model),
    )
  def forward(self,x):
    return self.net(x)
class layernorm(nn.Module):
  def __init__(self,d_model):
    super().__init__()
    self.ln=nn.LayerNorm(d_model) # d_model dim is consisten through layers. It fascialltes residual connection as given in forward
  def forward(self,x,sublayer_x): # Takes in as output the residual connection as well the output of the sublayer.( attention or FFNN)
    return self.ln(x+sublayer_x)


class PositionalEncoding(nn.Module):
  def __init__(self, model_dimension,expected_max_sequence_length=max_length):
      super().__init__()
      position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
      frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

      positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
      positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
      positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions
      self.register_buffer('positional_encodings_table', positional_encodings_table)

  def forward(self, embeddings_batch):
      assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
          f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

      positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]
      return  positional_encodings

class EncoderBlock(nn.Module):
  def __init__(self,h,d_k,d_v,d_model,rate):
    super().__init__()
    self.MHA = MultiHeadAttention(h,d_k,d_v,d_model)
    self.dropout1= nn.Dropout(rate)
    self.layernorm1= layernorm(d_model)
    self.Feed_Forward= Feed_Forward(d_model,d_ff)
    self.layernorm2= layernorm(d_model)
    self.dropout2 = nn.Dropout(rate)
  def forward(self,x,padding_mask=None):
    x = self.layernorm1(x,self.dropout1(self.MHA(x,x,x,padding_mask))) # mha->dropout-> residual -> layer
    x = self.layernorm2(x,self.dropout2(self.Feed_Forward(x)))  #FFNN->dropout-> residual -> layer
    return x


class Encoder(nn.Module):
  def __init__(self,vocab_size,max_length,d_model,h,d_k,d_v,d_ff, n_layers,rate):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model, max_length)
    self.dropout = nn.Dropout(rate)
    self.layers = nn.ModuleList([EncoderBlock(h,d_k,d_v,d_model,rate) for _ in range(n_layers)])

  def forward(self,sentence, padding_mask):
    x = self.embedding(sentence)
    x = x + self.positional_encoding(x)
    x = self.dropout(x)
    for layer in self.layers:
      x = layer(x, padding_mask)
    return x


class DecoderBlock(nn.Module):
  def __init__(self,h,d_k,d_v,d_model,d_ff,rate):
    super().__init__()
    self.MHA = MultiHeadAttention(h,d_k,d_v,d_model)
    self.dropout1= nn.Dropout(rate)
    self.add_norm = layernorm(d_model)
    self.MHA2 = MultiHeadAttention(h,d_k,d_v,d_model)
    self.dropout2= nn.Dropout(rate)
    self.add_norm2 = layernorm(d_model)
    self.Feed_Forward= Feed_Forward(d_model,d_ff)
    self.dropout3= nn.Dropout(rate)
    self.add_norm3 = layernorm(d_model)

  def forward(self,x,encoder_output,padding_mask=None,look_ahead_mask=None):
    x = self.add_norm(x,self.dropout1(self.MHA(x,x,x,look_ahead_mask)))
    x = self.add_norm2(x,self.dropout2(self.MHA2(x,encoder_output,encoder_output,padding_mask))) # Enocoder output because cross attention. Padding mask because keys and values are from encoder
    x = self.add_norm3(x,self.dropout3(self.Feed_Forward(x)))
    return x

class Decoder(nn.Module):
  def __init__(self,vocab_size,max_length, d_model,h,d_k,d_v,d_ff,n_layers,rate):
    super().__init__()
    self.embedding= nn.Embedding(vocab_size,d_model)
    self.positional_encoding = PositionalEncoding(d_model, max_length)
    self.dropout = nn.Dropout(rate)
    self.layers = nn.ModuleList([DecoderBlock(h,d_k,d_v,d_model,d_ff,rate) for _ in range(n_layers)])
  def forward(self,decoder_input,encoder_output,lookahead_mask,padding_mask):
    x = self.embedding(decoder_input)
    x = x + self.positional_encoding(x)
    x = self.dropout(x)
    for layer in self.layers:
      x = layer(x,encoder_output,padding_mask,lookahead_mask)
    return x


class Transformer(nn.Module):
  def __init__(self,enc_vocab_size,dec_vocab_size,enc_seq_len,dec_seq_len,d_model,h,d_k,d_v,d_ff,n_layers,rate):
    super().__init__()
    self.encoder = Encoder(enc_vocab_size,enc_seq_len,d_model,h,d_k,d_v,d_ff, n_layers,rate)
    self.decoder = Decoder(dec_vocab_size,dec_seq_len,d_model,h,d_k,d_v,d_ff,n_layers,rate)
    self.model_output = nn.Linear(d_model,dec_vocab_size) #converting d_model to dec_vocab_size)

  def forward(self,encoder_input,decoder_input):
    padding_mask_enc   = self.padding_mask(encoder_input)
    padding_mask_dec   = self.padding_mask(decoder_input)
    lookahead_mask_dec = self.look_ahead_mask(decoder_input.shape[1])
    lookahead_mask_dec = torch.maximum(lookahead_mask_dec,padding_mask_dec) # important
    encoder_output     = self.encoder(encoder_input,padding_mask_enc)
    decoder_output     = self.decoder(decoder_input,encoder_output,lookahead_mask_dec,padding_mask_enc)
    return self.model_output(decoder_output)
  def padding_mask(self,x): # Changes the values to 1 from 0. Where zero is pad vocab
    x=(x==0).float()
    x=x.type(torch.float32)
    return x[:,None,None,:]

  def look_ahead_mask(self,shape):
    x=torch.triu(torch.ones((shape,shape)),diagonal=1)
    return x.type(torch.float32)
