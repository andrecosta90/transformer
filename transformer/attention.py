import math
import torch

'''
    ```
    tensor([[0.6096, 0.0473, 0.0543, 0.0662, 0.2226],  # Attention weights for Query 1
            [0.1249, 0.3162, 0.0651, 0.1539, 0.3399],  # Attention weights for Query 2
            [0.1652, 0.0749, 0.3702, 0.1524, 0.2373],  # Attention weights for Query 3
            [0.1738, 0.1530, 0.1316, 0.2399, 0.3016],  # Attention weights for Query 4
            [0.1875, 0.1083, 0.0657, 0.0967, 0.5418]]) # Attention weights for Query 5
    ```

    + Each row in the tensor corresponds to a different "query" (or position in the input sequence).
    + Each column within a row corresponds to a different "key" (or part of the input that the model is considering).
    + For the first query, the model is focusing 60.96% of its attention on the first part of the input, 
        4.73% on the second part, and so on. The model considers the first part of the input to be the most important for this query.
    + For the second query, the model is spreading its attention more across the second and fifth parts of the input, 
        with 31.62% and 33.99% attention, respectively. The second and fifth parts are the most important for this query.

'''
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    # Calculates some raw scores (using the dot product of query and key) that tell you 
    #   how related different parts of the input are to each other.
    # It is like similarity between the query and each key.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # p_attn is the attention weights, which tell the model how much focus to give to each part of the input.
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # Imagine you have a bunch of important facts (these are the value), 
    #   and you know how important each fact is (these are the p_attn). 
    #   Now, you want to combine these facts in a way that reflects their importance.
    #
    # `torch.matmul(p_attn, value)` takes the attention weights (p_attn) and 
    #   uses them to create a meaningful combination of the input information (value). 
    #   This is how the model decides what information to focus on when making decisions.
    bp = torch.matmul(p_attn, value)
    return bp, p_attn

