# GPT_Translator

The most innovative concept introduced in Transformers was Attention. What is attention?  
Attention is used to capture the relations between different tokens or words in sequence. For example: "How are you?". Here, there are three tokens (3 words). Attention calculates the relations between these three tokens. Doing so can train it for different tasks, such as text generation. Attention works as follows.  

1) The input to attention mechanism is [Tokens, Embeddings]. Here, Tokens are words that were converted into integers. Embeddings are then used to map these tokens into an embedding space. So, each word will have its own representation or point in the embedding space.  
2) Each token there generates three different vectors. Keys represent the information a token contains. Query which represents what the token needs. Value vector, which is used to map the information generated from key-query interaction into relevant tasks.  
3) First, the information is generated using a dot product between queries and keys. Suppose we have Queries = [Q1, Q2, Q3], where Q1 represents the Query vector generated by token1 and Keys = [K1, K2, K3], then  
    Q. K.transpose. gives = [[Q1*K1, Q1*K2, Q1*K3], [Q2*K1, Q2*K2, Q2*K3], [Q3*K1, Q3*K2, Q3*K3]]. Each row represents the interaction between a query and all of the keys. If we apply a softmax function to each row, we are left with probabilities that sum up to 1. Thus, we are left with a matrix with information about each word's importance in the sequence of a token.  

**Note:**   
- K.Transpose is necessary because it gives an inner product between tokens ==> [T, Q] * [K, T] ==> [T, T]. So, the dimensions of the query and keys should be the same.  
- These multiplication works irrespective of the length of the sequence (T). Thus, the attention mechanism can have an infinite context.
- If we are to do a Keys * query, we will have information about a token for others token. The structure of both keys and queries is the same. Thus, both of these formulations are correct. Which matrix functions as the key and query is determined by which side the value token is multiplied from. 

4) softmax(Q.dot(K.Transpose())) * V. What does this do? [T, T] * [T, Embedding_dim] ==> [T, embedding_dim]. We have the original input dimensions back.  
   softmax([[Q1*K1, Q1*K2, Q1*K3]]) * ([[V1], [V2], [V3]]) = [V1[1]*p(Q1*K1) + V2[1]*p(Q1*K2) + V3[1]*p(Q1*K3), V1[2]*p(Q1*K1) + V2[2]*p(Q1*K2) + V3[2]*p(Q1*K3), ...]  
   Thus, we end up with a convex linear combination of individual scalars of values vector.
   
**Note**
- What happens if we do (V.Transpose * softmax(Q.dot(K.Transpose))).Transpose. This will reverse the functions of Queries and keys. 
- How do different sequence lengths come into the picture? One sequence functions as the query, and one as the keys. [T1,E] * [E,T2] ==> [T1,T2]. Then [T1,T2] * [T2,E] ==> [T1,E]. Thus, the T1 is preserved. To preserve the dimension of output, the values and keys matrix are provided by the sequence, which acts as the context or encoder rather than the decoder.  
- Why individual scalar values of vector? Why not a combination of the values vector itself? Such a combination is incapable of accessing the whole embedding space. For if there are two tokens, it can access a plane only (called column space of the matrix).  
- Is not the combination of every dimension with the same weights undesirable? Yes.  
- Why have a values vector? Why not do the linear combination of the vectors itself? Attention * X instead of V?

 ## Masking
1) Although the transformer can handle different sequence lengths, it is advisable to use the same sequence length in a batch; this makes it easier to train it using GPUs. Such a padded sequence should not be considered, so we introduce padding_mask. This is after the Q.dot(K) mechanism sets all the padded values in the sequence to -infinity. exponential of -infinity is zero, thus removing it from softmax calculations. 

2) "Look-Ahead Mask"-->This preserves auto-regressive property. The transformer generates a single new token for every turn during inference or generation. The transformer( decoder in particular) can only access the tokens it has generated before. This property must be mimicked during training, so we added a look-ahead mask. During the output calculations, queries should not access any keys of the future token. So, we add a look-ahead mask. The upper triangular mask with infinity for every non-zero value acts as the mask. The look-ahead mask remains an upper triangular matrix, but the size increases with every new token generation. The token generated is added to the input during the next generation cycle.

3) The encoder only requires a padding mask. In a Decoderblock, the first Attention block requires a look_ahead and padded mask. The second block requires only a padded mask. A look-ahead mask should not be used because keys come from the encoder. Let us use the machine translation task to understand this- Transformer will always have access to the complete sentence that needs to be translated. This sentence is fed into the encoder.

What does a look_ahead mask + padding mask look like?
Ex: sequence=[1,2,0,0,0] , then mask will look like=[[T1,T2,0 0,0],[T3,T2,0,0,0],-----]. This mask is used during the first Attention block in the Decoder

## MultiHead Attention:

MultiHead Attention, as the name suggests, multiple attention is done in parallel. I would like to do that. Keys, queries, and values received from the input are indexed into distinct matrices in the embedding dimensions. Attention is carried out individually in every set of matrices. 
Thus, with the same computational complexity, we can capture context many times. At the end, the output matrices are combined to achieve the same output. The main point here is that K, Q, and V are indexed in the embedding dim, not the context/time/token dimension. This preserves the ability to access every context.

## Transformer

### Encoder
The encoder consists of attention and FFNN with residual and layer norm connections. We start with the attention block, which acts as the residual input to the input. Layernorm is applied to the output(x+attention(x)). Then, FFNN is applied as another residual connection with another layer of the norm layer at the end. The FFNN consists of two linear layers. One maps emb_dim to 4* emb_dim, and the layer maps it back to the original dimension. The encoder part is used for the input if we have eng to ger translation. Eng sentence is the input to Encoder. The encoder thus does not require a look_ahead mask as the whole eng sentence should be accessed for translation in every German token generation. Dropout is added after every sub-layer, that is, after the attention and FFNN layer.

### Decoder 

The decoder works in the same manner except for an MHA layer in between, which receives the keys and values from the encoder. If we are to implement a text generation model without any input whatsoever, then we would not need an encoder, and thus, the transformer/decoder block would more resemble an encoder. 


## Learning Rate

![alt text](https://github.com/JitheshPavan/GPT_Translator/blob/main/data/lr_formula.PNG)

