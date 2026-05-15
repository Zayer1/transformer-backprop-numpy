import numpy as np

np.random.seed(42)
lr = 0.1

# ==========================================
# 1. THE DATA PIPELINE (Your Code)
# ==========================================
vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
sentence = ["the", "cat", "sat"]



# Tokenize
tokens = [vocab[word] for word in sentence]


# Embedding Matrix
vocab_size = len(vocab)
embedding_dim = 6 #we update this to 6

# ==========================================
# POSITIONAL EMBEDDINGS (The Time Stamps)
# ==========================================
# The Problem: Matrix multiplication doesn't care about order. 
# If 'cat' = [1, 2] and 'dog' = [3, 4], their dot product is always 11. 
# Whether the sentence is "cat chases dog" or "dog chases cat", the math is 100% identical.
# The network is completely blind to time.
# 
# The Solution: We create a dedicated "Position Vector" for each slot in the sentence.
# Position 1 gets its own vector, Position 2 gets its own vector, etc.
# Instead of hard-coding these vectors using complex math (like the original paper), 
# we start them as random numbers and let the calculus figure out the optimal values!
P = np.random.randn(len(sentence),embedding_dim)

# ==========================================
# THE CAUSAL MASK (The "Blindfold")
# ==========================================
# Problem: When predicting a word, the AI shouldn't be allowed to look at future words.
# Solution: In our scores matrix (Q @ K.T), Row 'i' is the Query word, and Col 'j' is the Key word.
# If j > i, it means the Key word comes AFTER the Query word in the sentence (The Future).
# 
# 1. np.ones: Creates a grid of 1s.
# 2. np.triu(..., k=1): Targets the upper triangle. It keeps 1s exactly where j > i, and zeroes out the rest.
# 3. * -1e9: Turns all those "future" 1s into Negative Infinity.
# 
# Later, when Softmax processes this mask, the math of e^(-infinity) will crush all future 
# probabilities down to exactly 0.000%, physically making it impossible for the AI to "see" ahead.
mask = np.triu(np.ones((len(sentence), len(sentence))), k = 1) * -1e9

num_heads = 2 #we're building 2 parallel heads
head_dim = embedding_dim // num_heads #Each head gets 3 dimensions
E = np.random.randn(vocab_size, embedding_dim)


# ==========================================
# 2. THE NEURAL NETWORK 
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z)
    return exp_z/np.sum(exp_z, axis = 1, keepdims = True)
# --- LAYER 1: HIDDEN STATE ---
# Connects 3 input features to 5 hidden neurons
W1 = np.random.randn(embedding_dim, 5) # Shape: (6, 5)
b1 = np.zeros((1, vocab_size)) # Shape: (1, 5)
W2 = np.random.randn(5, vocab_size) # Shape: (5, 5)
b2 = np.zeros((1, vocab_size)) # Shape: (1, 1)

#Layer 1: Self-attention weights
W_q = np.random.randn(embedding_dim, embedding_dim)
W_k = np.random.randn(embedding_dim, embedding_dim)
W_v = np.random.randn(embedding_dim, embedding_dim)

for epoch in range(1000):
    # -------------------------------------------------------------------------
    # THE ADDITION: Word Meaning + Time Stamp
    # -------------------------------------------------------------------------
    # Why do we ADD (E + P) instead of gluing them together (concatenating)?
    # 1. Math Cost: Gluing a 6-number word and a 6-number position makes a 12-number array. 
    #    This would double the size of all our matrices (W_q, W_k, W_v) and make training incredibly slow.
    # 
    # 2. How Addition works here: Imagine a 2D graph. The word "Cat" is at coordinates (x=100, y=100).
    #    If we add the Position 1 vector (x=1, y=0), Cat becomes (101, 100).
    #    If we add the Position 2 vector (x=0, y=1), Cat becomes (100, 101).
    #    When this new mixed coordinate hits our weight matrix (W_q), the matrix multiplication 
    #    is mathematically capable of separating the massive base number (100) from the tiny 
    #    shift (+1). The network decodes both "What is this word?" and "Where is it?" at the exact same time!
    sentence_embedding = E[tokens] + P

    #From that we generate the values for Q, K, V. These are deterministic projections, the weights (W_q, W_k, W_v) are what we randomized
    Q = sentence_embedding @ W_q #Shape (3,6)
    K = sentence_embedding @ W_k #Shape (3,6)
    V = sentence_embedding @ W_v #Shape (3,6)

    #Split Q,K and V into multiple heads
    #Head 1 gets the first 3 columns
    Q1 = Q[:, :head_dim]
    K1 = K[:, :head_dim]
    V1 = V[:, :head_dim]
    #Head 2 gets the second 3 columns
    Q2 = Q[:, head_dim:]
    K2 = K[:, head_dim:]
    V2 = V[:, head_dim:]

    #Head 1 processing
    scores1 = Q1 @ K1.T + mask # The Bouncer slaps -1e9 on future coordinates
    weights1 = softmax(scores1) # Softmax converts -1e9 to exactly 0.0% attention
    context1 = weights1 @ V1 #(3,3)

    #Head 2 processing
    scores2 = Q2 @ K2.T + mask # The Bouncer slaps -1e9 on future coordinates
    weights2 = softmax(scores2) # Softmax converts -1e9 to exactly 0.0% attention
    context2 = weights2 @ V2 #(3,3)

    #Now we glue those 2 heads together
    context_vector = np.concatenate((context1, context2), axis = -1)
    
    sentence_vector = context_vector[-1:] #This basically grabs the last row, shape (1, 6)

    #Layer 1
    Z1 = sentence_vector @ W1 + b1 # (1, 6) @ (6, 5) + (1, 5) -> Shape: (1, 5)
    A1 = sigmoid(Z1) # Shape: (1, 5)

    #LAYER 2: THE OUTPUT LAYER
    # Task: We need to compress the (1, 5) hidden state down to a (1, 1) final prediction.
    Z2 = A1 @ W2 + b2 # (1, 5) @ (5, 5) + (1, 1) -> Shape: (1, 5)
    A2 = softmax(Z2) # Shape: (1, 5)

    # The Corrected Loss Calculation
    target = np.array([[0.0,0.0,0.0,1.0,0.0]]) #Shape: (1,5)

    # Notice how 'target' is only multiplied by the first log term now
    #loss = -(target * np.log(A2 + 1e-9) + (1 - target) * np.log(1 - A2 + 1e-9))
    loss = -np.sum(target *np.log(A2 +1e-9))

    #Backpropagation
    #dZ2 = dL/dZ = dL/dA * dA/dZ
    #dZ2 = -(target*(1/(A2 + 1e-9)) + (1-target)* (-1/(1-A2+1e-9))) * (A2*(1-A2))
    dZ2 = A2 - target # (1, 5) - Scalar -> Shape: (1, 5)

    #dW2 = dL/dW2 = dL/dZ2 * dZ2/dW2
    dW2 = A1.T @ dZ2 # (5, 1) @ (1, 5) -> Shape: (5, 5)

    #db2 = dL/dB2 = dL/dZ2 * dZ2/db2
    db2 = np.sum(dZ2, axis = 0, keepdims = True) # Shape: (1, 5)
    #dZ1 = dL/dZ1 = dL/dA1 * dA1/dZ1
    #dL/dA1 = dL/dZ2 * dZ2/dA1
    dZ1 = dZ2 @ W2.T * A1 * (1-A1) # (1, 5) @ (5, 5) * (1, 5) -> Shape: (1, 5)

    #dW1 = dL/dW1 = dL/dZ1 * dZ1/dW1
    #dZ1/dW1 = sentence_vector
    dW1 = sentence_vector.T @ dZ1 # (3, 1) @ (1, 5) -> Shape: (3, 5)

    #db1 = dL/db1 = dL/dZ1 * dZ1/db1
    #dZ1/db1 = 1
    db1 = np.sum(dZ1, axis = 0, keepdims = True) # Shape: (1, 5)

    #Tracing the blame back to the beginning
    #d_sentence_vector = dL/d_sentence_vector = dL/dZ1 * dZ1/d_sentence_vector
    d_sentence_vector = dZ1 @ W1.T # Shape: (1, 5) @ (5, 6) = (1, 6)

    #Backpropagation for attention algorithm:

    #Blame for context vector calculation
    #d_context_vector = dL/d_context_vector = dL/dSvec * dSvec/dC
    #dSvec/dC = M.T (M here stands for matrix, also the formal actual formula for Svec is actually M @ C)
    #dL/dSvec = dL/dZ1 * dZ1/dSvec = dZ1.W1
    #dZ1/dSvec = W1.T
    d_context_vector = np.zeros((len(tokens), embedding_dim)) #shape (3, 6); this is basically making an empty placeholder matrix for storing the value
    d_context_vector[-1:] = d_sentence_vector #Actual formula woud be dZ1 @ W1.T @ M.T
    
    #Now we split the blame
    #First 3 columns of error to head 1
    d_context1 = d_context_vector[:, :head_dim]
    #Second 3 columns of error to head 2
    d_context2 = d_context_vector[:, head_dim:]

    #Now we calculate the blame of head 1 and 2
    d_weights1 = d_context1 @ V1.T
    dV1 = weights1.T @ d_context1
    d_weights2 = d_context2 @ V2.T
    dV2 = weights2.T @ d_context2
    # Note on Causal Mask: Because the mask is a static constant (no weights), 
    # its derivative is 0. We literally throw it away, so the backward pass requires ZERO changes!
    #d_scores =  Ei*Si*(1-Si) - sum(Ej*Si*Sj)
    #S is attention weights, E is d_attention_weights
    d_scores1 = weights1 * (d_weights1 - np.sum(d_weights1 * weights1, axis = -1, keepdims = True))
    d_scores2 = weights2 * (d_weights2 - np.sum(d_weights2 * weights2, axis = -1, keepdims = True))

    #Next up, we find the blame for Q1 and K1
    #dQ = dL/dQ = dL/d_scores * d_scores/dQ
    dQ1 = d_scores1 @ K1 #(3, 3) @ (3, 3) -> Shape: (3, 3)
    #dK = dL/dK = dL/d_scores * d_scores/dK
    dK1 = d_scores1.T @ Q1 #(3, 3) @ (3, 3) -> Shape: (3, 3)

    #And now, we find the blame for Q2 and K2
    dQ2 = d_scores2 @ K2 # (3, 3) @ (3, 3) -> Shape: (3, 3)
    dK2 = d_scores2.T @ Q2 # (3, 3) @ (3, 3) -> Shape: (3, 3)
    
    # --- RECOMBINE THE PARALLEL BRAINS ---
    # Glue the Head 1 and Head 2 errors back together side-by-side
    dQ = np.concatenate((dQ1, dQ2), axis = -1) # (3,3) + (3,3) -> Shape: (3, 6)
    dK = np.concatenate((dK1, dK2), axis = -1) # (3,3) + (3,3) -> Shape: (3, 6)
    dV = np.concatenate((dV1, dV2), axis = -1) # (3,3) + (3,3) -> Shape: (3, 6)

    #After that we do dW_q, dW_k and dW_v
    # sentence_embedding is shape (3, 6), so its transpose is (6, 3)
    dW_q = sentence_embedding.T @ dQ # (6, 3) @ (3, 6) -> Shape: (6, 6)
    #And the same goes for the other 2
    dW_k = sentence_embedding.T @ dK # (6, 3) @ (3, 6) -> Shape: (6, 6)
    dW_v = sentence_embedding.T @ dV # (6, 3) @ (3, 6) -> Shape: (6, 6)
    #Note: the reason we apply transpose here is because of the content within the matrices themselves
    #sentence_embedding is a matrix with it's row 1 as features for the word "the"
    #dQ's first colum is error for word 1, 2 and 3
    #So if we forgot the transposition, word 1's feature would be multiplied with word 1's, 2's and 3's errors, which would make 0 sense
    #Same case for dK and dV

    #Now we calculate the blame of the sentence embedding
    # dQ is shape (3, 6), W_q is shape (6, 6), so W_q.T is (6, 6)
    d_sentence_embedding_q = dQ @ W_q.T # (3, 6) @ (6, 6) -> Shape: (3, 6)
    #Calculate similarly for the others
    d_sentence_embedding_k = dK @ W_k.T # (3, 6) @ (6, 6) -> Shape: (3, 6)
    d_sentence_embedding_v = dV @ W_v.T # (3, 6) @ (6, 6) -> Shape: (3, 6)

    #Now we wrap it all up with a simple addition. The reason it's not seperate and instead addition here is
    #because all those 3 are connected into the full on Q, K and V calculation. However deriving them
    #all at once is mathematically impossible with 3 different variables, and we need a total derivative here
    # Adding three (3, 6) matrices together results in a single (3, 6) matrix.
    d_sentence_embedding = d_sentence_embedding_q + d_sentence_embedding_k + d_sentence_embedding_v

    #Update rule
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    W_q = W_q - lr * dW_q
    W_k = W_k - lr * dW_k
    W_v = W_v - lr * dW_v
    
    for i, token in enumerate(tokens):
        E[token] = E[token] - lr * d_sentence_embedding[i]
        #We also update the positional index too
        P[i] = P[i] - lr * d_sentence_embedding[i]

    #Every 10 steps printing out progess
    if epoch % 10 == 0:
        print(f"Epoch {epoch} , Loss {loss:.6f} , Prediction: {A2[0][3]:.4f}")

print("\nFinal Attention Weights for 'sat':")
print(weights1[-1])

print("\nFinal Attention Weights for 'sat' (Head 2):")
print(weights2[-1])

print("\nFinal Attention Weights (Head 1):")
print(weights1)

print("\nFinal Attention Weights (Head 2):")
print(weights2)
