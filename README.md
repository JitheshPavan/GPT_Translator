<h1 align="center"> GPT Translator </h1>

- [Training](#training)
  - [Inference](#inference)  <!-- Inference is now a sub-item of Training -->
- [Theory](#theory)
  - [Masking](#Masking)
  - [MultiHead Attention](#MultiHead_Attention)
  - [Transformer](#Transformer)
    -[Encoder](#Encoder)
    -[Decoder](#Decoder)
  - [Learning Rate](#Learning_Rate)
  - [Input](#Inputs)
  - [BLEU Score](#BLEU_Score)
  - [Positional Encoding](#Positional_Encoding)

## Training
In the original paper, the authors used warmup steps during training. The output gradient from the transformer is large in magnitude. So, we use warmup to gradually build up to an LR so that that model can stabilize. But, if we shift the position of the Layernorm layer to pre-multi-head attention rather than the post-residual layer, gradients remain stable. So, you don't need a warmup. Note: both types of transformers can achieve the same accuracies. One is significantly faster.

<p align="center">
  <img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/loss.png" width="400"/>
  <img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/bleu.png" width="400"/>
</p>
number of parameters in my model is 32790680.

### Inference
An example of successful translation is 'How are you,' translated to [wie,geht, es, dir]. 

[what are you doing] translates to [was geht es gut], which, when translated back, means what is going well. This is a failed case. 

## Theory
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

![](https://github.com/JitheshPavan/GPT_Translator/blob/main/data/lr_formula.PNG)

The learning rate increases with warm-up steps and then gradually decreases. The bigger the warmup rate, the smaller the slope and peak in the lr. 
With increased dimension of the model, the learning rate decreases. It decreases by 10 points for every 100 increase in dimension. 
<img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/learning_rate.png" width="800" height="500">

### Inputs
We add two tokens to each input: A beginning-of-sequence token and an end-of-sequence token. Doing so gives model markers/ boundaries to avoid stopping too early or blabbering endless tokens. It helps inference, where the model outputs these tokens, which means the generation is over.

The transformer predicts one token at a time. Suppose we have decoder input as "`<START>` How are you? `<END`". Decoder input and output follow this pattern. 

First iteration="`<START>`" --> "How".

Second Iteration=" `<START>` How" --> "are" 

Third Iteration ="`<START>` How are" --> "you?"

Fourth Iteration= "`<START>` How are you" -->" "`<END>`"

So, we need to shift our decoder output expectation to the right. Thus, we get for a token of 5, 4 examples. This is called Shifted Inputs.

### BLEU Score

Machine translation can have multiple correct answers. The output may not be similar to the input with respect to the words and their positions, but that does not make it a bad translation. So, we calculate the precision by taking single, bigram, trigram, and tetragram pairs from the outputs. We then compare how many times each pair appears in our target. We have the number of times a pair appears in the target (clipped so that it does not exceed the denominator) divided by the number of times the pair appears in the prediction. The geometric mean is taken over the numbers received from all the pairs of different sizes. We also add a penalty if the length of the prediction is smaller than the length of the target.

### Positional Encoding

The attention mechanism is independent of the length of tokens; it is also independent of order. It cannot know the order of the tokens. Why is it so?

1) The attention mechanism works on the embedding space, not the tokens. That is, throughout the ordeal, the last dimension is worked on. So, it works on every token independently, just like the batch size. Even at the end, the last neural network layer maps the embedding to vocabulary size done on the embedding dimension. That is why it is independent of the order. So, if you input two tokens to the decoder, it will output two tokens after having worked on both of these tokens independently. Now, you may think that since it works on both inputs independently, context is non-existent. But tokens do interact during the attention mechanism, and only during that mechanism. This fuels its infinite attention span. This interaction captures the relation between the tokens, not the order of the tokens.
2) We use a positional encoding vector to capture the order of tokens. The positional encoding vector displaces the embedding vector in a certain direction. Thus, positional encoding can be thought of as an attempt to group the first tokens together. Thus, we have two opposing forces during encoding: embedding vector, which attempts to put similar vectors in a similar cluster, and positional encoding vector, which attempts to group tokens together based on their position. The positional embedding vector has to follow two properties.
- it should not get large in magnitude as the number of tokens increases. This will cause the embedding vector to grow small in influence. 
- The positional vector should not get smaller in magnitude as the token size increases; the embedding vector will overshadow the positional vector.
3) The trigonometric functions sin follows the first property. But it is periodic. That is why we decrease its frequency to such a level that it does not repeat itself. But if you do that, it will slowly decrease on one end. That is why we alternate between cosine and sine functions. When the sin value is small, the cosine value is large, and vice-versa. So, the second option is actually satisfied.
  
- **Positional Encoding Equation:**
  
PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
Where:
- `pos` is the position of the token in the sequence.
- `i` is the dimension of the embedding.
- `d_model` is the total dimensionality of the embeddings.

**Note**
- The frequency remains the same with changes in position. That is, we get a constant frequency wave in the direction of position.
- The frequency decreases for the direction in dimension, leading to slower oscillations. Thus, we get a wave that is getting longer periods as we go deeper into the embedding dimension.
- The larger the d_model value, the faster the change in frequency.

Thus we have a plot like this

<img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/scaling.png" width="800" height="500">

How do we infer from this image? The usual sin wave is depicted in this image through colors. The light color represents the peak, and the dark color represents the negative peak. Now, if you look at the columns spacing between the same colors does not change, indicating no change in freq. Rowwise, the spacing increases the deeper you go, indicating an increase in frequency. Now, this change can be inferred as a decrease in magnitude for the sin wave, but since we are using the cos wave, alternatively, the effect is counteracted. This is why freq must change with dimension because the sin wave follows the cosine wave. if we do not change the frequency, the end value and the begging value come out the same.
