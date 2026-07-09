#import "template.typ": * 
#import "shortcuts.typ": * 

#import "@preview/latexlike-report:1.0.0": *

#show: doc => conf(doc)

// Todo add footer
#set page(numbering: "1", number-align: top+right)

#set page(
    paper: "us-letter",
)

#set par(
    first-line-indent: 0em,
    leading: 0.6em,
    linebreaks: "simple",
    justify: true,
)

#set text(
    size: 11.5pt,
    font: "New Computer Modern",
    hyphenate: false
)
#show math.equation: set text(font: "New Computer Modern Math")

#let ip(x, y) = $chevron.l #x, #y chevron.r$

// Todo fix indenting after figures/theorems

#set cite(style: "alphanumeric")

#outline(depth: 2, title: [#smallcaps("Contents")])

#set align(left)

= Introduction

At the most basic level, a natural language model is a function that takes as input a sequence of words and attempts to predict the word that follows. This prediction takes the form of a probability distribution over the set of all possible words, indicating the level of confidence the model has that any given word might follow. This model generates text on a given input by sampling from the output distribution. Once a word is generated, it is concatenated to the end of the original input, leading to a new input for the model to once again sample from. Repeating this process yields output text as desired.

More precisely, text is segmented into _tokens_. Although tokens are often words, they take on many other forms: punctuation, prefixes and postfixes, single characters. 
Ideally, these tokens would allow for a model that is not restricted to one language or type of text, and could make meaningful predications on any form of structured text with sufficient training data (other languages, code, etc.). 
We will assume that the set $cal(X)$ of tokens for our model is chosen beforehand, but note that determining this set is an important part of engineering an efficient language model.

#linebreak()

#figure(
    image("tokens.png", width: 40%),
    caption: [GPT-4 tokenizer on sample text.]
)
#linebreak()
Let $n$ be the number of tokens in the sequence of text the model takes in as input, known as _context length_. The model can therefore be seen as a function $f: cal(X)^n -> cal(P)_(cal(X))$, where $cal(P)_cal(X)$ is the set of probability distributions over $cal(X)$. To develop a model, we design the model's parametric form $f(bold(X) | theta)$ and use likelihood estimation on a large set of sample data to determine parameters $theta$ for which model performs sufficiently well. These are not simple tasks: for any decent model, $f$ is extremely complex and highly nonlinear, and the size of $cal(X)$, $n$, and $theta$ are several orders of magnitude beyond the regime on which classical methods in statistics are effective.

#figure(
    table(
          columns: (auto, auto, auto, auto, auto),
          inset: 5pt,
          align: center,
          table.header(
            [Model], [Tokens $cal(X)$], [Context $n$], [Parameters $theta$],
            [Release]
          ),
          [BERT (Large)], [30000], [512], [340 Million], [2018],
          [GPT-2], [50257], [1024], [1.5 Billion], [2019],
          [GPT-3], [50257], [2048], [175 Billion], [2020],
          [Llama 3], [128000], [128000], [405 Billion], [2024]
        ),
    caption: [Size of $cal(X)$, $n$, and $theta$ of some modern language models.]
)

Assuming that there exists methods to optimize parameters $theta$ with respect to model performance, this approach to developing a model is inherently powerful. Models can first be designed using intuition about how a machine may interpret and predict language under a mathematical framework.
If the model performs sufficiently well after training, the goal is accomplished. From a purely predictive perspective, it is not necessary that the learned internal mechanisms resemble the original intuition motivating the architechture.

Earlier approaches to language modeling relied on architectures (RNN, CNN) which primarily capture information in a limited or compressed way as text is processed sequentially. These models have difficulty maintaining a truly global understanding of a longer passage, since dependencies from distant parts of the text are either truncated or gradually lost as information is passed through successive steps.

Introduced in 2017, the Transformer architecture @vaswani2017attention resolves this issue by introducing a completely different mechanism, one which allows every token to directly interact with every other token in the input. This shift fundamentally changed the field of natural language processing, ultimately powering the large language models that now underpin modern AI systems. Section 2 gives an intuition-based overview of the Transformer architecture.

Among other ways to scale a model, of particular interest is the context length $n$. In theory, this would allow the model to use much longer-range context, enabling it to connect information across distant parts of a document and maintain a more coherent global understanding of the input. In real-world applications, this is especially useful for tasks like summarizing long reports, analyzing legal or financial documents, and answering questions that require information spread across many pages. It also improves performance in settings like code generation, where earlier definitions or functions must be remembered much later in the file.

However, scaling $n$ has been shown to be a non-trivial task. Among other reasons, the attention mechanism core to the Transformer is suceptible to degenerate behaviour in the large-$n$ limit, which leads to complete loss of expressive power of the model. Recent work has focused on improving the scaling behaviour of softmax attention by introducing an extra parameter, which can be found proposed in @nakanishi2025scalable. Empirically, various modifications have been shown to mitigate the effects of this collapsing attention (REFERENCES), but mathematical justification has remained incomplete. The rest of this report summarizes relevant results in an attempt to unify existing empirical observations and theoretical perspectives on attention scaling.

= The Transformer

This section provides a precise description of the architecture of the transformer model as outlined in @vaswani2017attention. For each step, a brief heuristic justification of the design choice is given based on intuition about how a machine might use the architecture to model language effectively, however it is important to keep in mind that all that is being done is designing the parametric form $f_theta (bf(X))$, from which the model will optimize $theta$ over a set of training data. Whether the model "thinks" this way in practice is hard to determine, and likely not the case.

Throughout, any _weight matrix_ $W$ or _bias vector_ $b$ is a learned parameter of the model, and any other value is either an input or an intermediary value. In a computational spirit, we will use the "$<-$" symbol to indicate variable assignment. 
Vector arrows will be used in order to clarify when individual rows or columns of matrices are in question.

== Softmax

Beyond language modelling, it is frequently needed that a model output a probability distribution over a finite set of candidate outcomes. This arises in tasks such as classification, retrival, and decision-making, where the model must assign relative likelihoods to discrete alternatives rather than produce a single deterministic value.
In these settings, the outputs are typically unconstrained real-valued scores that must be converted into a valid probability distribution.

#definition("Softmax")[
    Let $s in RR^n$. Define the function $"softmax": RR^n -> cal(P)_n$ as
    $
        "softmax"(s)_j = e^(s_j) / (sumkn e^(s_k)).
    $
    For a matrix $S in RR^(n times m)$, $"softmax"(S)$ is the $RR^(n times m)$ matrix where $"softmax"$ has been applied either row-wise or column-wise, depending on context.
]
It is clear that $"softmax"$ outputs a valid probability distribution, is differentiable, and preserves the ordering of scores. While many other functions could in principle be used, softmax is the most widely adopted choice.


== Embedding and Positional Encoding

We assign each token in $cal(X)$ is assigned a unique number from $1$ to $|cal(X)|$ and represent it as one-hot vector
#footnote[A one-hot encoding is a vector that has a $1$ in the position corresponding to the number it represents, and $0$ everywhere else.].
Thus, our input $X = (arrow(X)_1, ..., arrow(X)_n) in cal(X)^n$ is represented as a $|cal(X)| times n$ matrix, where each column is a one-hot encoding of a token.


Modeling and learning are much more difficult in high-dimensional discrete spaces, as the tools of calculus are not available. Hence, we choose an _embedding dimension_ $d$ which allows for a continuous representation of token "meaning" in $RR^d$. 
One interpretation motivating this embedding space is the idea that orthogonal directions could encode different concepts or ideas, and that linear combinations of vectors along these directions could represent the meaning of a given token.

A matrix $W_E in RR^(d times abs(cal(X)))$ contains all token embeddings, where column $i$ is the embedding for token $i$. For any token $arrow(X)_i$, multiplication by this matrix gives the corresponding column of $W_E$.

$
underbrace(
  mat(
    0.12, -0.33, dots.h, 0.07;
    0.54, 0.21, dots.h, -0.11;
    dots.v, dots.down, dots.down, dots.v;
    -0.08, 0.44, dots.h, 0.19
  ),
  W_E,
)
quad
underbrace(
  mat(
    0;
    dots.v;
    1;
    dots.v;
    0
  ),
  arrow(X)_i,
)
$


For the complete input $X$, the columns of the matrix multiplication $W_E X$ give the corresponding embeddings for each token. This process is equivalent to a lookup table.

We also allow for the model to obtain information of the position of a token in the sequence. This is given by the matrix $W_P$, where column $i$ contains the encoding for position $i$. 
Historically, this has been implemented in different ways, including hard-coding this matrix to follow a sinusoidal pattern, as in the original Transformer architecture, or learning the positional embeddings directly during training. More recent approaches incorporate positional information within the attention mechanism itself through techniques such as relative or rotary positional embeddings. Regardless of the implementation, the goal is to provide the model with information about the ordering of tokens, which would otherwise be absent from the attention mechanism. We will assume this is a learned parameter matrix.

Combined with the embedding step, we compute

$
    E <- W_E X + W_P,
$
which contains all token-level information that is independent of any interactions between tokens.

== The Attention Layer

Given that words in any language rarely have a unique meaning independent of context, it is essential to allow tokens to "communicate" with each other.
The attention layer modifies the positions of tokens in embedding space to better suit their meaning in context.

=== Single-Headed Attention

Let $W_K, W_Q in RR^(d_h times d)$, where $d_h$ is known as the _head dimension_. Given the previously obtained token embeddings $E$, we compute

$
    Q <- W_Q E, "    " K <- W_K E,
$
called the _query matrix_ and _key matrix_, respectively. Note that the columns of these matrices are transformations of embedding vectors: $arrow(Q)_i = W_Q arrow(E)_i$ and $arrow(K)_i = W_K arrow(E)_i$. The query $arrow(Q)_i$ represents what information the token is looking for from other tokens in order to refine its contextual meaning. The key $arrow(K)_i$ represents what kinds of information the token can provide to other tokens. Both these vectors are projections of their corresponding token embeddings into a shared lower-dimensional space $RR^(d_h)$, specifically used for relational matching. 

To capture the extent to which the key $arrow(K)_i$ is relevant to query $arrow(Q)_j$, we compute the _attention score_ $s_(i j) = arrow(K)_i dot arrow(Q)_j$, where larger values correspond to alignment in the query-key space. Thus, the matrix

$
    S <- (K^T Q) / sqrt(d_h)
$
has scaled attention score $s_(i j) \/ sqrt(d_h)$ as entry $(i, j)$. This scaling factor keeps dot-product similarity dimension-invariant, meaning scores stay probabilistically $O(1)$ scale regardless of head dimension and is derived assuming $W_Q$ and $W_K$ have entries with mean $0$ and variance $1\/d_h$, as is common in the initialization of models and often enforced through various normalization procedures beyond the scope of this overview.

Given a query, we would like these scores to correspond to weights indicating the relative importance of each key's token for updating the embedding of the queried token. We therefore apply the softmax function column-wise to $S$ giving

$
   A <- "softmax"(S).
$

The following is an example of the above step.

#let score-table = table(
  columns: 5,
  align: center,
  inset: 5pt,
  stroke: none,

  [], [*The*], [*dog*], [*sat*], [*down*],
  [*The*], [1.1], [-0.2], [-0.8], [-0.5],
  [*dog*], [-0.3], [2.4], [3.6], [0.7],
  [*sat*], [-0.7], [1.8], [2.9], [1.2],
  [*down*], [-0.4], [0.6], [1.3], [2.1],

  table.hline(y: 1),
  table.vline(x: 1),
)

#let weight-table = table(
  columns: 5,
  align: center,
  inset: 5pt,
  stroke: none,

  [], [*The*], [*dog*], [*sat*], [*down*],
  [*The*], [0.61], [0.04], [0.01], [0.04],
  [*dog*], [0.15], [0.56], [0.62], [0.14],
  [*sat*], [0.10], [0.31], [0.30], [0.24],
  [*down*], [0.14], [0.09], [0.06], [0.58],

    table.hline(y: 1),
    table.vline(x: 1),
)

#align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,

    [
      #score-table
      #align(center)[Scores $S$]
    ],

    [
        #align(horizon)[$ mapsto^"softmax"$]
    ],

    [
      #weight-table
      #align(center)[Weights $A$]
    ],
  )
]

This completes the conversion of token embeddings into a normalized distribution over which tokens are most relevant to each other.

Returning to the original embeddings $E$, let $W_V in RR^(d times d)$. we compute the _value matrix_

$
    V <- W_V E.
$
The columns $arrow(V)_i = W_V arrow(E)_i$ represent the information each token contributes when it is used by other tokens. Recalling that the columns $arrow(A)_i$ of the matrix $A$ contain the attention weights assigned by token $i$, the update to a token's embedding is given by the weighted sum of the value vectors according to weights $arrow(A)_i$. In matrix form, this can be written as

$
    Delta E <- V A.
$
This update is then applied to $E$ to obtain the final output of the attention layer,
$
     E_"out" <- E + Delta E.
$

=== Multi-Headed Attention

We wish to use multiple single-headed attention layers in parallel to allow different heads to specialize in different types of relationships and patterns within the input sequence. Let $h$ be the number of heads in this multi-headed attention layer. We choose the head dimension $d_h$ such that $d = h d_h$.

As before, for each head, we have matrices $W_(Q)^((i)), W_(K)^((i)) in RR^(d_h times d)$ and compute

$
    A^((i)) <- "softmax"((K^T Q)/ sqrt(d_h)).
$
Allowing for each $W_V^((i))$ matrix to have size $d times d$ would result in $h d^2$ parameters across all heads. In practice, we would like the parameter count to be comparable to $W_Q$ and $W_K$, so we force $W_V$ to have the low-rank structure

$
    W_V^((i)) = W_(V_l)^((i)) W_(V_r)^((i)),
$
where $W_(V_l)^((i)) in RR^(d times d_h)$ and $W_(V_r)^((i)) in RR^(d_h times d)$. Across all heads, this contributes $2 d d_h$ parameters.

While this factored matrix could be applied per-head, each attention layer instead computes  $W_(V_r)^((i)) A$, which are then concatenated vertically and multiplied by the horizontal concatenation of $W_(V_l)^((i))$ matrices, giving
#footnote[
    This allows the transformation to be expressed as a composition of two learned linear maps, which is computationally more efficient and better structured for parameter optimization than a product of two matrices.
]

$
    Delta E <- ub(
        mat(W_(V_l)^((1)), dots.c, W_(V_l)^((h)), delim: "["),
        W_O
    )
        vec(W_(V_r)^((1)) A, dots.v, W_(V_r)^((h)) A, delim: "["),
$
which is equivalent to computing $Delta E^((i)) <- W_V^((i)) A$ for every head and summing the results. Because of this, we will consider the horizontal concatenation $W_O$ to be a separate matrix that is applied after concatenating the results from each individual head. 
For each head, we will consider the $W_(V_r)^((i))$ matrix to simply be the $W_V^((i))$ matrix applied in the single-headed case.

Adding this to the original embeddings gives the final output for the multi-headed attention,
$
     E_"out" <- E + Delta E.
$

=== Masking

In self-attention, every token can attend to every other token. However, in
many settings including language modelling, we wish to restrict which tokens are allowed to interact. In most generality, if we wish to forbid a token from interacting with another, we can set the corresponding entry in $S$ to $-infinity$, which leads to a corresponding entry in $A$ to be 0. In practice, this is done by adding a masking matrix $M$ with a very large negative integer in the desired entries and $0$ elsewhere to $E$.

In the context of language modelling, this is espeically useful in order to parellelize training of th model. For example, the sequence "the dog sat down" contains four sub-examples:

1. [empty context] $->$ the,
2. the $->$ dog,
3. the dog $->$ sat,
4. the dog sat $->$ down.

An appropriate mask for this purpose needs to prohibit tokens from communicating with tokens beyond their current position. In other words, query $i$ cannot access key $j$ for any $j > i$, meaning that mask must be applied to the strictly lower triangular part.

#let score-table = table(
  columns: 5,
  align: center,
  inset: 5pt,
  stroke: none,

  [], [*The*], [*dog*], [*sat*], [*down*],
  [*The*], [1.1], [-0.2], [-0.8], [-0.5],
  [*dog*], [-$inf$], [2.4], [3.6], [0.7],
  [*sat*], [-$inf$], [$-inf$], [2.9], [1.2],
  [*down*], [-$inf$], [$-inf$], [$-inf$], [2.1],

  table.hline(y: 1),
  table.vline(x: 1),
)

#let weight-table = table(
  columns: 5,
  align: center,
  inset: 5pt,
  stroke: none,

  [], [*The*], [*dog*], [*sat*], [*down*],
  [*The*], [1], [0.07], [0.01], [0.04],
  [*dog*], [0], [0.93], [0.66], [0.14],
  [*sat*], [0], [0], [0.33], [0.24],
  [*down*], [0], [0], [0], [0.58],

    table.hline(y: 1),
    table.vline(x: 1),
)

#align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,

    [
      #score-table
      #align(center)[Masked Scores $S$]
    ],

    [
        #align(horizon)[$ mapsto^"softmax"$]
    ],

    [
      #weight-table
      #align(center)[Weights $A$]
    ],
  )
]

Although essential to include in this overview, the mathematical analysis in future sections does not apply masking. 

== Feed Forward Layer

Once a token has been contextualized using attention, it carries information from across the sequence, but this information is still only combined linearly through a weighted sum. 
The feed forward layer allows for each token to independently introduce nonlinear transformations in its embedding, converting this context into more expressive and higher-level representations. This is done using a _multi-layer perceptron_.

Let $E$ be input embeddings that have been processed by an attention layer. The multi-layer perceptron aims to model how a biological brain might process information using neurons. 
We choose a feed-forward dimension $d_f$, which we will interpret to be the number of neurons in this layer. 

Each neuron processes information using an affine transformation of the input data. This is then fed into a non-linear _activation function_ $sigma$ which aims to model how a neuron might fire. We use $sigma(x) = max(0, x)$, known as ReLU, although there are many other options. The _activation_ of neuron $i$ for some embedding $arrow(E)_j$ is the scalar value given by

$
    n_(i j) <- sigma(arrow(W)_"in"^((i)) dot arrow(E)_j + b_"in"^((i))).
$
Letting $W_"in"$ to be the matrix with weights $arrow(W)_"in"^((j))$ as columns and $b_"in"$ to be the column vector with components $b_"in"^((i))$, this can be expressed in matrix form as

$
    N <- sigma(W_"in" E + b_"in" bold(1)_(d_f)^T),
$
where $sigma$ is applied component-wise, and $bold(1)_d_f in RR^(d_f)$ is the column vector of all ones.

Similar to attention, an update to each embedding is calculated using an affine transformation of the embeding's corresponding neuron activations. In parallel, this gives

$
    Delta E <- W_"out" N + b_"out",
$
which is added to the input to give

$
   E_"out" = E + Delta E.
$

== Unembedding

In practice, embeddings are fed through many iterations of transformer blocks (a multi-headed attention layer followed by a feed forward layer), each with separate parameters. 
After this processing, we wish to convert each embedding $arrow(E)_i$ into a probability distribution over tokens $cal(X)$, indicating the level of confidence that each token in the vocabulary could be the next token at position $i$.

This is done by applying the unembedding matrix $W_U in RR^(|cal(X)|times d)$ to $arrow(E)_i$, followed by a softmax. In matrix form,

$
    Y = "softmax"(W_U E + b_U bold(1)^T),
$
where $"softmax"$ is applied column-wise. Note that each column $arrow(Y)_i$ is the prediction for the next token at each position $i$ in the input sequence. If lower triangular masking is applied, this allows for $n$ different predictions on the $n$ different sub-examples. 
This is useful for model training purposes, but for generation purposes we only use the distribution $arrow(Y)_n$ corresponding to the last input token.

== Summary

== Long-Context Transformers

While the term "long" continuously changes meaning as advances are made in machine learning, long-context transformers are transformers that are designed to support a context length $n$ that is substantially longer than what is seen in common models. Many real-world tasks such as reasoning over long documents, codebases, or dialogues require access to information that may be introduced thousands or even millions of tokens earlier, where simply truncating the context leads to severly degraded performance. Increasing context length effectively augments the models "memory", which allows for richer retrieval of relevant evidence and more coherent global reasoning. 

It is important to distinguish between two types of long-context transformers. The first consists of models explicitly trained on long sequences, which attempt to internalize long-range dependencies during optimization. The second consists of models trained on shorter sequences but designed to generalize at inference time to longer sequences. There also exists hybrids of these two models, which might split training into $80%$ short sequences and $20%$ long sequences, for example.

From a scaling perspective, there are many issues related to naively increasing context length that apply to both types of long-context transformers. First and foremost, attention grows quadratically in context length, since a vanilla attention matrix $A$ is of size $n times n$. Second, attention suffers from large-$n$ scaling issues inherent to the softmax function. The remainder of this report investigates the latter. In general, the results collected throught are moreso properties of the softmax function under specific inputs rather than anything regarding the underlying Transformer architechture or training, and thus we will not differentiate between the two types of long-context Transformers unless it is pertinent.

= The Need for Attention Scaling

== Insufficiency of Standard Softmax

=== Single Head

Consider the standard softmax function applied to a sequence of score vectors $(s^((n)))$ such that $s^((n)) in RR^n$ for all $n$ with uniformly bounded components $m <= s_j^((n)) <= M$ for $m, M in RR$. For a fixed $j$, 

$
    "softmax"(s^((n)))_j = e^(s_j) / (sumkn e^(s_k)) <= e^(M) / (n e^m) = e^(M - m)/n,
$
as well as
$
    "softmax"(s^((n)))_j >= e^(m - M)/n.
$
Then, we have that $"softmax"(s^((n)))_j = Theta(1/n)$ for all $j$ and thus the maximum attention weight $a_"max"$ goes to 0 as $n$ goes to $inf$. 
Heuristically, the entire softmax vector becomes diffuse and "true" attention, in the sense of assigning a non-vanishing proportion of mass to a distringuished set of coordinates, cannot emerge.
Therefore, any nontrivial attention mechanism in the large-$n$ limit must be driven by score fluctuations whose scale grows with $n$, allowing certain coordinates to overcome the $Theta(n)$ growth of the softmax normalizing constant. Because attentions scores are usually designed to be probabilistically $O(1)$, this is naturally a problem when considering large-$n$ inputs.

As an example, consider the sequence of vectors $s^((n)) = (3, -2, ..., -2)$ with first component $3$ and all other components $-2$, noting that these are clearly uniformly bounded. @32 shows that not only does $a_max$ decrease to 0 relatively quickly, but the entropy of the distribution quickly approaches the entropy of the uniform distribution. This is refered to as _rank collapse_, since it is the behaviour obtained when a rank one matrix is used in the attention calculation.

#figure(
    image("entropy_softmax_plot.png", width: 45%),
    caption: [Rank collapse on scores with bounded components.],
)<32>

Even if scores do grow with $n$, the nature of the growth of the scores can lead to another failure mode known as _entropy collapse_. This occurs when the maximum component grows too quickly with respect to the rest of the scores, leading softmax to behave as a "hardmax" for large $n$, outputting a one-hot vector. This collapse is shown in @EC for the sequence $t^((n)) = 1/3 log(n) s^((n))$.

#figure(
    image("entropy_softmax_plot3.png", width: 45%),
    caption: [Entropy collapse on scores with growing components.],
)<EC>

=== Transformer Setting

#stheorem(name: "Long-Context Attention Collapse", source: [@velickovic2024softmax])[
    Let $cal(X) subset.eq RR^d$ be a finite set of token embeddings, and let $X^((n)) in cal(X)^n$ be a matrix of $n$ embeddings. 

    Define query and key matrices $Q^((n)) = phi(X^((n))) in RR^(n times k), quad K^((n)) = kappa(X^((n))) in RR^(n times k),$
    where $phi, kappa$ are continuous functions obtained as compositions of a finite number $L$ of layers of the following two types:
    
    1. A feedforward layer $F_l (Z)_i = f_l (Z_i)$ that is acts row-wise and is continuous,
    2. Self-attention layer $G_l (Z)_i = sumjn a_(i j) v_l (Z_j)$, where $a_(i j) in [0, 1]$ are softmax-normalised attention coefficients and $v$ is a continuous feedforward network.

    Define the score matrix $S^((n)) = Q^((n)) (K^((n)))^T in RR^(n times n)$ and row-wise softmax attention matrix $A^((n)) = "softmax"(S^((n)))$.

    Then, $||A^((n))||_max -> 0$ as $n -> inf$, where $||A^((n))||_"max" = max_(i j) |a_(i j)|$.
]

#proof[
    Since $cal(X) subset.eq RR^d$ is finite, both $cal(X)$ and its convex hull $"conv"(cal(X))$ are compact. Since all feedfoward operators $F_l$ and $v_l$ act row-wise and are continuous, they preserve compactness row-wise. Each output row of a self-attention layer is a convex combination of vectors in a compact set, hence remains in a compact set. By induction over $L$, the finite number of layers, all embeddings lie in a compact set, independent of $n$. In particular, there exists a compact set $C subset.eq RR^k$ such that all rows $Q_i^((n)), thin K_i^((n)) in C$. 
    Hence, there exists an $M > 0$ such that any score has the bound $|S^((n))_(i j)| <= M$. By a similar derivation as before, we obtain
    $
        e^(-2 M)/n <= A^((n))_(i j) <= e^(2 M) / n.
    $
    Since $M$ does not depend on $n$, $||A^((n))||_"max" -> 0$ as $n -> infinity$.
]

While a simplified model, this shows a different fundamental limitation of Transformers. For a fixed trained model, there is no architectural mechanism that guarantees stable, non-degenerate attention behaviour as the sequence length $n$ increases beyond the range seen during training.
Although one might hope that the model could compensate by implicitly scaling scores with sequence length, the architecture does not provide a reliable way to represent or extrapolate $n$. Consequently, any implicit learned rescaling of $W_K$ or $W_Q$ matrices remains uniformly bounded, and cannot in general counteract the linear growth of the softmax normalization as $n -> infinity$.

== Scalable Softmax

An initial instict may be to introduce a fixed parameter $beta > 0$ to softmax,
$
    "softmax"(s)_j = e^(beta s_j)/(sumkn e^(beta s_k)),
$
which suffers from the same issue as standard softmax when scores are bounded, since
$
    e^(beta(m - M))/n <= "softmax"(s)_j <= e^(beta (M - m)) / n.
$
Hence, any non-degenerate scaling $beta$ must depend on $n$, i.e. $beta = beta_n$. The _scalable softmax_ function

$
    "ssmax"(z)_j = (e^((s p_n + b )z_j))/(sumkn e^((s p_n + b) z_k))
$
is proposed in @nakanishi2025scalable,
where $s$ and $b$ are scalar learned parameters that are unique to each attention layer and head, and $p_n$ is a learnable parameter shared across all layers and heads, depending only on the input vector size $n$. They train this scalable softmax on a standard model and find a relationship fitting $p_n tilde log n$. For a given attention layer and head (i.e. fixed constants), this corresponds to $beta_n tilde log n$ as proposed in our original setup.

According to these results, consider the scaling $beta_n = gamma log n$, for some $gamma > 0$, and assume that $s$ has a unique maximum $z_"max" > z_"2nd"$.
On one hand, we have
$
    a_"max"
        = n^(gamma z_"max")/(sumkn n^(gamma z_k)) 
        <= (n^(gamma z_"max"))/((n-1) n^(gamma z_"min") + n^(gamma z_"max"))
        = 1/((n - 1) n^(-gamma(z_"max" - z_"min")) + 1).
$
On the other hand,
$
    a_"max"
        >= (n^(gamma z_"max"))/((n-1) n^(gamma z_"2nd") + n^(gamma z_"max"))
        = 1/((n - 1) n^(-gamma(z_"max" - z_"2nd")) + 1).
$
If $gamma > 1/(z_"max" - z_"min")$, we see that $a_"max" -> 1$, whereas if $gamma < 1/(z_"max" - z_"2nd")$,
then $a_max -> 0$. This is one example showing that even if $beta_n$ is of correct asymptotic order, constant multiples can still induce degenerate behaviour. Thus, $beta_n$ is likely quite fragile in a general sense, which motivates the study of attention scaling beyond the "quick fix" of adding a learnable parameter to the model. 

Furthermore, we observe a phase transition phenomenon in $gamma$. This phenomenon occurs beyond this simple example and appears to be intrinsic to the attention mechanism, as it appears in various different models of attention, as seen in later sections. Precisely, this phase transition separates the behaviour of attention into _subcritical_ and _supercritical_ regimes, $gamma < gamma_1$ and $gamma > gamma_2$ respectively for constants $gamma_1 < gamma_2$, in which the behaviour of attention is undesirable.
For the _critical_ regime $gamma in [gamma_1, gamma_2]$, often a single value $gamma_c = gamma_1 = gamma_2$, attention behaves as desired. In terms of manually tuning this scaling, we see that this must proceed in two stages: determining the order of the scaling $beta_n$ followed by tuning the constant multiple $gamma$.

Returning to the example $s^((n)) = (3, -2, -2, ..., -2)$, we apply the scaling $beta_n = 0.2 log n$. As seen in figure @33, this scaling avoids rank and entropy collapse.

#figure(
    image("entropy_softmax_plot1.png", width: 53%),
    caption: [Non-degenerate behaviour of scaled softmax on $s^((n))$.]
)<33>

This scaling was obtained by the heuristic justification that to avoid both rank and entropy collapse, the maximum element $s_"max"$ and the sum of all other elements must be of the same exponential order. This reasoning extends beyond this simple case to more complex score vectors, and is a first step in ensuring proper scaling. In this case, 

$
    a_"max" = e^(3 beta)/(
            ub(e^(3 beta), "max") + ub((n - 1)e^(-2 beta), "bulk"))
            = (1 + e^(-5 beta + log n) + o(e^(-5 beta + log n)))^(-1),
$
which avoids convergence to either $0$ or $1$ (entropy and rank collapse, respectively) if $beta = 0.2 log n$, which correspond to entropy and rank collapse, respectively.

At this point, the question remains how to determine an effective scaling $beta_n$
 in more general contexts, and whether any universality properties hold. Since transformer inputs lie between deterministic and fully random regimes, a completely general analysis seems intractable at the moment, and it is natural to first study simplified models that capture the essential structure of the attention mechanism before addressing more realistic settings.

= Simplex-Like Geometry <sec:simplex>

The following is an overview of results and proofs found in @chen2025critical with intermediate steps filled in as well as further discussion.

== Setup

Consider a simplified version of unmasked single-headed attention in which the head dimension equals the embedding dimension and $W_K = W_Q = W_V = I_d$. 
For a collection of token embeddings $x_1, ..., x_n in RR^d$, define the normalized vectors $u_i = x_i\/norm(x_i)$. 
The scaled attention scores for the normalized vectors $u_i$ are thus given by 
$
    s_(i j) = beta ip(u_i, u_j),
$
where $ip(dot, dot)$ denotes the standard inner product on $RR^d$, and $beta$ is a scaling factor to be chosen. The corresponding attention matrix $A$ has entries

$
       a_(i j) = e^(s_(i j))/Z_i, "    " Z_i = sum_(k=1)^n e^(s_(i k)).
$
The main object at study is the nonlinear operator $"ATT": RR^d -> RR^d$, defined as
$
    "ATT"(u_i) = sum_(j = 1)^n a_(i j) u_j.
$
Finally, recalling that in transformer architechtures, the attention layer computes an update to the original embeddings through a residual connection, we define the updated embeddings and their normalized versions
$
    x'_i = "ATT"(u_i) + alpha x_i, "   " u'_i = x'_i/norm(x'_i),
$
where $alpha >= 0$ is the strength of the residual connection. Geometrically, $alpha$ interpolates between purely attention-driven dynamics and identity-preserving behaviour.

== Critical Scaling

We are interested in how the attention scaling $beta$ affects the geometry of the token embeddings, namely how pairwise angles evolve under the attention update. 
Equivalently, this amounts to studying the relationship between $ip(u'_i, u'_j)$ and $ip(u_i, u_j)$. 

Consider a further simplified setup in which $ip(u_i, u_j) = p in (0, 1)$ if $i != j$ and $norm(u_i)^2 = q$. The key insight is that the behaviour of attention is governed by the normalization factor $Z_i$, which simplifies to

$
   Z_i = e^beta + (n - 1) e^(p beta) 
$
under these assumptions. Thus, the asymptotic behaviour of $Z_i$ depends on the competition between the self-attention term and the aggregate contribution of attention from all other tokens. Under the scaling $beta = gamma log n$,
$
    e^beta = n^gamma, "    " (n - 1)e^(p beta) asymp n^(1 + p gamma).
$
Hence, the dominant contribution to $Z_i$ depends on the comparison between the exponents $gamma$ and $1 + p gamma$. The critical threshold occurs when $gamma = 1/(1 - p)$. This reveals that there are three regimes under which attention operates. 
In the subcritical regime $gamma < 1/(1 - p)$, attention weights become asymptotically uniform, so each token moves toward the global average. 
In the supercritical regime $gamma > 1/(1 - p)$, the self-attention term dominates, and attention operator converges to the identity map. In the critical regime $gamma = 1/(1 - p)$, both terms are balanced, leading to nontrivial dynamics as desired for an effective attention layer.

In @t31 we generalize this setup beyond the symmetric simplex setting by allowing pairwise inner products to vary within fixed bounds. We show that the attention operator acts asymptotically as a contraction in the subcritical regime and as an identity in the supercritial regime. @c31 provides exact results for the limiting inner product for the simplex case. Although @chen2025critical claims that these results demonstrate that the scaling $beta = Theta(log n)$ is intrinsic to the behaviour of attention mechanisms rather than an artifact of the simplex geometry, we will later see that this in fact not the case.

#stheorem(name: "Phase Transition in Attention Geometry", source: [@chen2025critical])[
    Assume there exist constants $q_1, q_2 > 0$ and $p_1, p_2 in (0, 1)$ such that $q_1 <= norm(x_i)^2 <= q_2$ and $p_1 <= ip(u_i, u_j) <= p_2$ for any $i != j$, with $p_1 = ip(u_i, u_j)$ for some $i, j$. Let $beta = gamma log n$ with $gamma > 0$.

    1. If $gamma < 1/(1 - p_1)$ then there is a constant $epsilon > 0$ depending on $alpha, p_2, q_1, q_2$ such that
    $
        liminf_(n -> infinity) min_(i != j) thin ip(u'_i, u'_j) >= p_1 + epsilon.
    $
    2. If $gamma > 1/(1 - p_2)$, then for any $i$,
    $
        limn ip(u'_i, u'_j) = ip(u_i, u_j).
    $
]<t31>

In the subcritical regime, the value of $epsilon$ could in theory be very small, and thus the theorem should be interpreted as a qualitative phase transition rather than a quantitative improvement guarantee in this regime. In the supercritical regime, we obtain a much stronger result.
Although the randomly sampled vectors found in @fig-comparison do not necessarily satisfy the assumptions of @t31 and the embedding dimension $d = 2$ might lead to weaker concentration results, we still observe a clear separation between the subcritical (left) and supercritical (right) regimes, consistent with the qualitative behaviour described by the theorem.

#figure(
  grid(
    columns: (1fr, 1fr),
    image("underdamped.png", width: 100%),
    image("overdamped.png", width: 100%),
  ),
  caption: [Behaviour of geometric attention on random vectors.]
)<fig-comparison>

We will first prove a series of lemmas before proving @t31.
#lemma[
    For any $i in {1, ..., n}$, we have
    $
        Z_i = cases(
            (1 + o_n (1)) dot (sum_(k != i) e^(s_(i k))) 
            "   " &gamma < 1/(1 - p_1)",",
            (1 + o_n (1)) dot e^(beta)
            "   " &gamma > 1/(1 - p_2).
        )
    $
] <lemma31>

#proof[
    Recall $s_(i k) = beta ip(u_i, u_k)$ and $beta = gamma log n$, giving
    $
        Z_i = sum_(k=1)^n e^(s_(i k)) = n^gamma + sum_(k != i) n^(gamma ip(u_i, u_j))
    $
    If $gamma < 1/(1 - p_1)$, then $gamma - gamma p_1 - 1 < 0$.
    Using $ip(u_i, u_j) >= p_1$,
    $
        (n^gamma)/(sum_(k != i) n^(gamma ip(u_i, u_j)))
        <= n^gamma / ( (n - 1) n^(gamma p_1))
        = n^(gamma - gamma p_1 - 1) dot 1 / (1 - 1\/n)
    $
    which goes to $0$ as $n -> infinity$. Hence,
    $
        Z_i = 
        (1 + (n^gamma)/(sum_(k != i) n^(gamma ip(u_i, u_j))))
        (sum_(k != i) n^(gamma ip(u_i, u_j)))
        =
        (1 + o_n (1))(sum_(k != i) e^(s_(i k))).
    $
    Similarly, if $gamma > 1/(1 - p_2)$, then $gamma p_2 + 1 - gamma < 0$. Using $ip(u_i, u_j) <= p_2$,
    $
        (sum_(k != i) n^(gamma ip(u_i, u_j)))/n^gamma
        <= ((n - 1) n^(gamma p_2))/n^gamma
        = n^(gamma p_2 + 1 - gamma) (1 - 1\/n)
    $
    which goes to 0 as $n -> infinity$. Hence,
    $
        Z_i = (1 + (sum_(k != i) n^(gamma ip(u_i, u_j)))/(n^gamma))(n^gamma)
        = (1 + o_n (1)) dot e^beta.
    $
]

#lemma[
    If $gamma > 1/(1 - p_2)$, then for any $i = {1, ..., n}$,

    $
        "ATT"(u_i) = u_i + o_n (1),
    $
    where $bold(o)_n (1)$ does not depend on $i$.
] <lemma2>

#proof[
    By the previous lemma,

    $
        "ATT"(u_i) = Z_i^(-1) (e^beta u_i + sum_(j != i) e^(s_(i j)) u_j)
        = (1 + o_n (1)) (u_i + e^(-beta) sum_(j != i) e^(s_(i j)) u_j),
    $
    using the fact that $(1 + o_n (1))^(-1) = 1 + o_n (1)$. Since $norm(u_j) = 1$ for all $j$,
    $
       norm(e^(-beta) sum_(j != i) e^(s_(i j)) u_j)
       <= e^(-beta) sum_(j != i) e^(s_(i j)) 
       <= n^(-gamma) dot n^(gamma p_2) (n - 1)
    $
    which goes to $0$ as $n -> infinity$, independent of $i$.
]

#lemma[
    For any $i in {1, ..., n}$, 

    1. If $gamma < 1/(1 - p_1)$,

    $
        norm(x'_i)^2 <=
        alpha^2 norm(x_i)^2 + 2 alpha norm(x_i) p_2 + p_2 + o_n (1).
    $
    2. If $gamma > 1/(1 - p_2)$,
    $
        norm(x'_i)^2 = (alpha norm(x_i) + 1)^2 + o_n (1).
    $
    In both cases, $o_n (1)$ does not depend on $i$.
] <lemma3>

#proof[
    Using the definition of the update rule, expanding the inner product gives
    $
        norm(x'_i)^2 = alpha^2 norm(x_i)^2 + 2 alpha norm(x_i) ip(u_i, "ATT"(u_i)) + norm("ATT"(u_i))^2.
    $
    For the case of $gamma < 1/(1 - p_1)$, Lemma 3.1 gave $Z_i = (1 + o_n (1)) (sum_(j != i) e^(s_(i j)))$. Hence,
    $
        ip(u_i, "ATT"(u_i))
        &=
        ip(u_i, 1/Z_i sum_(j=1)^n e^(s_(i j)) u_j)\
        &= 1/Z_i sum_(j=1)^n e^(s_(i j)) ip(u_i, u_j)\
        &<= 1/Z_i (e^beta + p_2 sum_(j != i) e^(s_(i j)))\
        &= p_2 + o_n (1),
    $
    since $e^beta \/ sum_(j != i) e^(s_(i j)) = o_n (1)$, as shown previously. Similarly,

    $
      norm("ATT"(u_i))^2
      &= ip(
        1/Z_i sum_(k=1)^n e^(s_(i k)) u_k, 
        1/Z_i sum_(l=1)^n e^(s_(i l)) u_l
      ) \
      &= 1/Z_i^2 sum_(k=1)^n sum_(l=1)^n e^(s_(i k) + s_(i l)) ip(u_k, u_l)\
      &<= 1/Z_i^2 (
        e^(2 beta)
        + 2 p_2 e^beta sum_(j != i) e^(s_(i j))
        + p_2 sum_(k != i) sum_(l != i) e^(s_(i k) + s_(i l))
      )\
      &= p_2 + o_n (1).
    $

    For the case of $gamma > 1/(1 - p_2)$, applying @lemma2 directly gives
    $
        norm(x'_i)^2
            &= alpha^2 norm(x_i)^2
            + 2 alpha norm(x_i) ip(u_i, u_i + bold(o)_n (1))
            + ip(u_i + bold(o)_n (1), u_i + bold(o)_n (1))\
            &= alpha^2 norm(x_i)^2 + 2 alpha norm(x_i) + 1 + o_n (1)\
            &= (alpha norm(x_i) + 1)^2 + o_n (1).
    $
]

#lemma[
    For $i, j in {1, ..., n}$ with $i != j$,

    1. If $gamma < 1/(1 - p_1)$,
  
    $
        ip(x'_i, x'_j) >= 
            p_1 (alpha norm(x_i) + 1)(alpha norm(x_j, size: #80%) + 1)
            + o_n (1).
    $
    2. If $gamma > 1/(1 - p_2)$,
    $
        ip(x'_i, x'_j) =
            (alpha norm(x_i) + 1)(alpha norm(x_j, size: #80%) + 1)
            ip(u_i, u_j)
            + o_n (1).
    $
    In both cases, $o_n (1)$ does not depend on $i$.
] <lemma4>

#proof[
    Expand the inner product to give
    $
       ip(x'_i, x'_j) &= 
        alpha^2 norm(x_i) norm(x_j, size: #80%) ip(u_i, u_j)
        + alpha norm(x_i) ip(u_i, "ATT"(u_j))\
        &+ " "alpha norm(x_j, size: #80%) ip(u_j, "ATT"(u_i)) 
        + ip("ATT"(u_i), "ATT"(u_j)).
    $
    If $gamma < 1/(1 - p_1)$, by following a similar procedure as in @lemma3 we can instead bound the terms in question from below, i.e.
    $
        ip(u_i, "ATT"(u_j))
        &=
        ip(u_i, 1/Z_i sum_(j=1)^n e^(s_(j k)) u_k)\
        &>= 1/Z_i (p_1 e^beta + p_1 sum_(j != i) e^(s_(i j)))\
        &= p_1 + o_n (1),
    $
    and
    $
      ip("ATT"(u_i), "ATT"(u_j))
      &= ip(
        1/Z_i sum_(k=1)^n e^(s_(i k)) u_k, 
        1/Z_i sum_(l=1)^n e^(s_(j l)) u_l
      )\ 
      &>= 1/(Z_i Z_j) (
        p_1 e^(2 beta)
        + p_1 e^beta sum_(k != i) e^(s_(i k))
        + p_1 e^beta sum_(l != j) e^(s_(j l))
        + p_1 sum_(k != i) sum_(l != i) e^(s_(i k) + s_(j l))
      )\
      &= p_1 + o_n (1).
    $
    Hence,
    $
        ip(x'_i, x'_j) &>=
            alpha^2 norm(x_i) norm(x_j, size: #80%) p_1
            + alpha (norm(x_i) + norm(x_j, size: #80%)) p_1
            + p_1
            + o_n (1)\
            &= p_1 (alpha norm(x_i) + 1)(alpha norm(x_j, size: #80%) + 1) + o_n (1).
    $
    If $gamma > 1/(1 - p_2)$, @lemma2 can be applied to obtain
    $
        ip(x'_i, x'_j) &= 
            alpha^2 norm(x_i) norm(x_j, size: #80%) ip(u_i, u_j)
            + alpha norm(x_i) ip(u_i, u_j + bold(o)_n (1))\
            &+ " "alpha norm(x_j) ip(u_j, u_i + bold(o)_n (1))
            + ip(u_i + bold(o)_n (1), u_j + bold(o)_n (1))\
            &= (alpha norm(x_i) + 1)(alpha norm(x_j, size: #80%) + 1)
                ip(u_i, u_j) + o_n (1).
    $

]

We now prove @t31.

#proof([of @t31])[
  First, consider the case where $gamma < 1/(1 - p_1)$. We can further simplify the result in @lemma3 by completing the square, giving
  $
    norm(x'_i)^2 &<= 
        alpha^2 norm(x_i)^2 
        + 2 alpha norm(x_i) p_2
        + p_2 + o_n (1)\
    &= alpha^2 norm(x_i)^2
        + 2 alpha norm(x_i) + 1
        - 2 alpha norm(x_i) - 1
        + 2 alpha norm(x_i) p_2
        + p_2 + o_n (1)\
    &= (alpha norm(x_i) + 1)^2 
        - (1 - p_2) (2 alpha norm(x_i) + 1)
        + o_n (1)\
    &<= (alpha norm(x_i) + 1)^2
        - (1 - p_2) (2 alpha sqrt(q_1) + 1)
        + o_n (1),
  $
  where the last inequality follows from the fact that $norm(x_i) >= sqrt(q_1)$. Thus, there is some constant $delta > 0$ depending on $alpha, p_2, q_1, q_2$ such that

$
    1/norm(x_i) >= (1 + delta)/(alpha norm(x_i) + 1) + o_n (1).
$
An equivalent formulation of the result of @lemma4 is the equation

$
    ip(u'_i, u'_j) 
        >= ((alpha norm(x_i) + 1)(alpha norm(x_j, size: #50%) + 1))
        / (norm(x_i) norm(x_j, size: #50%)) p_1
        + o_n (1),
$
and hence

$
    ip(u'_i, u'_j) >= p_1 (1 + delta)^2 + o_n (1) = p_1 + epsilon + o_n (1),
$
where $epsilon = p_1 (2 delta + delta^2)$. Since this was for any $i != j$, it follows that
$
    liminf_(n -> infinity) min_(i != j) thin ip(u'_i, u'_j) >= p_1 + epsilon.
$
For the case where $gamma > 1/(1 - p_2)$, we formulate the result of @lemma4 as
$
    ip(u'_i, u'_j) = 
        ((alpha norm(x_i) + 1)(alpha norm(x_j, size: #80%) + 1))/(norm(x_i) norm(x_j, size: #80%)) ip(u_i, u_j)
        + o_n (1)
$
and substitute the result of @lemma3 to immediately obtain
$
    ip(u'_i, u'_j) = ip(u_i, u_j) + o_n (1).
$
Hence,
$
    limn ip(u'_i, u'_j) = ip(u_i, u_j).
$
  
]

#corollary("Phase Transition, Simplex Case")[
    Assume there exist constants $q > 0$, $p in (0, 1)$ such that $norm(x_i)^2 = q$ and $ip(u_i, u_j) = p$ for any $i != j$. Then,

    $
        limn ip(u'_i, u'_j) = cases(
            (p(alpha sqrt(q) + 1)^2)/(alpha^2 q + 2 alpha sqrt(q) p + p)
                "    " &gamma < 1/(1 - p)",",
            (p(alpha sqrt(q) + 1)^2)/(alpha^2 q + alpha sqrt(q) (1 + p) + (1 + 3 p)/4) 
                "    " &gamma = 1/(1 - p)",",
            p
                "    " & gamma > 1/(1 - p).
        )
    $
]<c31>

#proof[
    This corresponds to the special case of @t31 where $p_1 = p_2$.

    If $gamma < 1/(1 - p)$, the steps of the proof of @lemma3 follow with equality, giving
    $
        norm(x'_i)^2 = alpha^2 q + 2 alpha sqrt(q) p + p + o_n (1).
    $
    Similar reasoning can be used with @lemma4 to obtain
    $
        ip(x'_i, x'_j) = p(alpha sqrt(q) + 1)^2 + o_n (1).
    $
    Hence, 
    $
        limn ip(u'_i, u'_j) = (p(alpha sqrt(q) + 1)^2)/(alpha^2 q + 2 alpha sqrt(q) p + p).
    $
    If $gamma = 1/(1 - p)$, we first observe
    $
        (sum_(j != i) e^(s_(i j))) / (e^beta)
            = ((n - 1) n^(gamma p))/(n^gamma)
            = n^(gamma (p - 1) + 1) - n^(gamma(p - 1))
            = 1 + o_n (1),
    $
    thus $Z_i = (2 + o_n (1)) e^beta$. Similar calculations as in @lemma3 yield
    $
       ip(u_i, "ATT"(u_i)) = (1 + p)/2 + o_n (1),\
       ip("ATT"(u_i), "ATT"(u_i)) = (1 + 3p)/4 + o_n (1).
    $
    Similar calculations as in @lemma4 give
    $
        ip(u_i, "ATT"(u_j)) = p + o_n (1),\
        ip("ATT"(u_i), "ATT"(u_j)) = p + o_n (1).
    $
    By substituting these results into the inner product expansions, we obtain
    $
        ip(u'_i, u'_j) = ip(x'_i, x'_j) / (norm(x'_i) norm(x'_j, size: #80%))
        = (p(alpha sqrt(q) + 1)^2)/(alpha^2 q + alpha sqrt(q) (1 + p) + (1 + 3p)/4)
        + o_n (1).
    $

    If $gamma > 1/(1 - p)$, @t31 directly gives
    $
        limn ip(u_i, u_j) = ip(u_i, u_j) = p.
    $
]

Since $p < 1$, the trailing term in the denominators of the $gamma >= 1/(1 - p)$ cases show that we have $limn ip(u'_i, u'_j) > p$. Thus, even when scaled appropriately, attention is still inherently a contractive operator.

#remark[
    In the absence of residual connections, i.e. $alpha = 0$, we have

    $
        limn ip(u'_i, u'_j) = cases(
            1 "    " &gamma < 1/(1 - p)",",
            (4p)/(1 + 3p) "    " &gamma = 1/(1 - p)",",
            p "    " &gamma > 1/(1 - p).
        )
    $
]
The phase transition is especially clear in this case. In the subcritical regime, all token embeddings collapse onto the same direction, which implies that attention is behaving like an averaging operator, $"ATT"(u_i) = 1/n sum_j u_j$. In the supercritical regime, the geometry of the embeddings is asymptotically unchanged and attention behaves like the identity map.
In some sense, this provides a reason for why residual connections are crucical in avoiding this rank collapse phenomenon.

== Backpropagation

#stheorem(name: "Phase Transition in Attention Gradient", source: [@chen2025critical])[
    Assume there exist constants $q_1, q_2 > 0$ and $p_1, p_2 in (0, 1)$ such that $q_1 <= norm(x_i)^2 <= q_2$ and $p_1 <= ip(u_i, u_j) <= p_2$ for any $i != j$, with $p_1 = ip(u_i, u_j)$ for some $i, j$. Let $beta = gamma log n$ with $gamma > 0$.

    1. If $gamma < 1/(1 - p_1)$,
    $
        1/(n d) norm(nabla_X X')^2 <= (4 gamma^2 (log n)^2)/(q_1 d) + o_n (1).
    $
    2. If $gamma > 1/(1 - p_2)$,
    $
        1/(n d) norm(nabla_X X')^2 >= 1/q_2 (1 - 1/d) + o_n (1).
    $
]<t32>

#lemma[
    For any $i, k in {1, ..., n}$ and $u, w in {1, ..., d}$, 
    $
        pddv((N(x_k))_(w), (x_i)_u)
            = drd(i, k) (drd(w, u) norm(x_k)^2 - (x_k)_w (x_k)_u)/(norm(x_k)^3)
    $
]

#proof[
    By the quotient rule,
    $
        pddv((N(x_k))_(w), (x_i)_u)
            &= pddv(((x_k)_w dot norm(x_k)^(-1)), (x_i)_u)
            = drd(i, k) 
                (drd(w, u) norm(x_k) - (x_k)_w dot (x_k)_u / norm(x_k))/norm(x_k)^2
            = drd(i, k) (drd(w, u) norm(x_k)^2 - (x_k)_w (x_k)_u)/(norm(x_k)^3).
    $
]

#lemma[
    For any $k, j in {1, ..., n}$ and $w, v in {1, ..., d}$,
    $
        pddv(("ATT"(u_j))_v, (u_k)_w)
        &= [
            (drd(k, j) beta (sumo(m, n) e^(beta ip(u_j, u_m)) (u_m)_w (u_m)_v)
            + e^(beta ip(u_j, u_k)) (beta (u_j)_w (u_k)_v + drd(w, v))) dot Z_j\
        &- " " (drd(k, j) beta (sumo(l, n) e^(beta ip(u_j, u_l)) (u_l)_w) 
            + beta e^(beta ip(u_j, u_k)) (u_j)_w) 
            dot (sumo(m, n) e^(beta ip(u_j, u_m)) (u_m)_v)
        ] dot Z_j^(-2)

    $
]

#proof[
    Recall 
    $
        ("ATT"(u_j))_v = 
            (sumo(m, n) e^(beta ip(u_j, u_m)) (u_m)_v)/(sumo(l, n) e^(beta ip(u_j, u_l)))
            := (y_j)_v/Z_j
    $
    The derivative of the numerator,
    $
        pddv((y_j)_v, (u_k)_w) 
            &= sumo(m, n) [(pdv((u_k)_w) e^(beta ip(u_j, u_m))) (u_m)_v
                + e^(beta ip(u_j, u_m)) pdv((u_k)_w) (u_m)_v]\
            &= sumo(m, n) [beta(drd(k, j) (u_m)_w + drd(k, m) (u_j)_w)
                    e^(beta ip(u_j, u_m)) (u_m)_v
                + e^(beta ip(u_j, u_m)) (drd(k, m) drd(w, v))]\
            &= (beta drd(k, j) sumo(m, n) e^(beta ip(u_j, u_m)) (u_m)_w (u_m)_v)
                + e^(beta ip(u_j, u_k)) (beta (u_j)_w (u_k)_v + drd(w, v)).
    $
    The derivative of the denominator,
    $
        pddv(Z_j, (u_k)_w) 
        &= sumo(l, n) beta(drd(k, j) (u_l)_w + drd(k, l) (u_j)_w) 
            e^(beta ip(u_j, u_l))
        &= (drd(k, j) beta sumo(l, n) e^(beta ip(u_j, u_l)) (u_l)_w)
            + beta e^(beta ip(u_j, u_k)) (u_j)_w.
    $
    Applying the quotient rule yields the desired result.
]

#lemma[
    The Jacobian of the attention operator has the form
    $
        (pddv((("ATT"(N(x_j)))_v), (x_i)_u))_(u, v = 1)^d
        = norm(x_i)^(-1\/2)
            [(bf(R)_1 + bf(R)_2) Z_j - (bf(U_1) + bf(U_2)) tp bf(V)_j] dot Z_j^(-2),
    $
    where
    $
        &bf(R)_1 = drd(i, j) (bf(W)_j - u_i tp (bf(W)_j u_j)), 
        &"   " 
        &bf(R)_2 = e^(beta ip(u_j, u_i)) 
            ((- u_j + beta bf(P)_(u_i) u_j) tp u_i + I_d),\
        &bf(U)_1 = drd(i, j) beta (bf(P)_(u_i) bf(V)_j),
        &"   " 
        &bf(U)_2 = beta e^(beta ip(u_j, u_i)) (bf(P)_(u_i) u_j),\
        &bf(V)_j = sumo(m, n) e^(beta ip(u_j, u_m)) u_m,
        &"   " 
        &bf(W)_j = sumo(m, n) e^(beta ip(u_j, u_m)) u_m tp u_m,\
        &bf(P)_x y = y - ip(y, x) x.
    $
]

#proof[
    Let
    $
        J_(w v) = pddv(("ATT"(N(x_j)))_v, (N(x_k))_w)
        = pddv(("ATT"(u_j))_v, (u_k)_w),
    $
    which was computed in the previous lemma.
    By the multivariate chain rule, we have
    $
        pddv((("ATT"(N(x_j)))_v), (x_i)_u)
        &= sumkn sumo(w, d) J_(w v) dot pddv((N(x_k)_(w)), (x_i)_u)\
        &= sumo(w, d) J_(w v) dot
            (drd(w, u) norm(x_i)^2 - (x_i)_w (x_i)_u)/norm(x_i)^3\
        &= norm(x_i)^(-1) (J_(u v) - (u_i)_u sumo(w, d) J_(w v) (u_i)_w).
    $
    We compute
    $
        &sumo(w, d) J_(w v) (u_i)_w\
        &= [
            (drd(i, j) beta (sumo(m, n) e^(beta ip(u_j, u_m)) ip(u_m, u_i) (u_m)_v)
            + e^(beta ip(u_j, u_i)) ((beta ip(u_j, u_i) + 1)(u_i)_v) dot Z_j\
        &- " " (drd(i, j) beta (sumo(l, n) e^(beta ip(u_j, u_l)) ip(u_l, u_i)) 
            + beta e^(beta ip(u_j, u_i)) ip(u_j, u_i)) 
            dot (sumo(m, n) e^(beta ip(u_j, u_m)) (u_m)_v)] dot Z_j^(-2),
    $ 
    which then gives
    $
        pddv((("ATT"(N(x_j)))_v), (x_i)_u)
        &= [
            (drd(i, j) beta (
                sumo(m, n) 
                    e^(beta ip(u_j, u_m)) 
                    ((u_m)_u (u_m)_v - ip(u_m, u_i) (u_m)_v (u_i)_u
                ))\
            &+ " "e^(beta ip(u_j, u_k)) 
                (
                    beta (u_j)_u (u_k)_v 
                    + drd(u, v)
                    - (beta ip(u_j, u_i) + 1) (u_i)_v (u_i)_u
                )
            )
                dot Z_j\
        &- " " (drd(i, j) beta (
                sumo(l, n) 
                    e^(beta ip(u_j, u_l)) 
                    ((u_l)_u - ip(u_l, u_i) (u_i)_u)
                )\
        &+ " " beta e^(beta ip(u_j, u_k)) ((u_j)_u - ip(u_j, u_i) (u_i)_u)) 
        dot (sumo(m, n) e^(beta ip(u_j, u_m)) (u_m)_v)
        ]\ 
        &dot Z_j^(-2) dot norm(x_i)^(-1).
    $
    When written in matrix form (i.e. the transposed Jacobian), this becomes
    $
        &lrs([
        [drd(i,j) beta (bf(W)_j - u_i tp (W_j u_i))
        + e^(beta ip(u_j, u_k)) (
            beta u_j tp u_i + I_d 
            - (beta ip(u_j, u_i) + 1) thin u_i tp u_i
        )] dot Z_j\
        &- [drd(i, j) beta (
           bf(V)_j - ip(bf(V)_j, u_i) u_i)
           + beta e^(beta ip(u_j, u_i)) (u_j - ip(u_j, u_i) u_i)
           )
        ] tp bf(V)_j
        ]) dot Z_j^(-2) dot norm(x_i)^(-1)\
        &= lrs([
            (bf(R)_1
            + e^(beta ip(u_j, u_i))
                ((-u_i + beta bf(P)_(u_i) u_j) tp u_i + I_d)) dot Z_j\
            &- [drd(i, j) beta (bf(P)_(u_i) bf(V)_j) 
            + beta e^(beta ip(u_j, u_i)) (bf(P)_(u_i) u_j)] tp bf(V)_j

        ]) dot Z_(j)^(-2) dot norm(x_i)^(-1)\
        &= lrs([
            (bf(R)_1 + bf(R)_2) dot Z_j
            + (bf(U)_1 + bf(U)_2) tp bf(V)_j

        ]) dot Z_(j)^(-2) dot norm(x_i)^(-1),
    $
    where the intermediate steps follow from basic properties of the tensor product.
]

We now prove @t31.

#proof([of @t31])[
    Throughout this proof, we use the fact that $norm(x tp y)_F = norm(x)_2 norm(y)_2$, as well as $norm(A)_F^2 = tr(A^T A)$. The subscripts will be dropped since we are exclusively using the Frobenius norm for matrices and 2-norm for vectors. First consider the case when $gamma > 1/(1 - p_2)$.

    1. When $i != j$, $bf(R)_1 Z_j^(-1) = 0$. When $i = j$, we obtain (using the form of $bf(R)_1$ before simplification in the previous lemma)
    $
        norm(bf(R)_1) Z_j^(-1)
        &= norm(
            beta sum_(m = 1)^n 
            e^(beta ip(u_j, u_m)) (u_m tp u_m - ip(u_m, u_i) u_i tp u_m)
        ) dot Z_j^(-1)\
        &= norm(
            beta sum_(m != j) 
            e^(beta ip(u_j, u_m)) (u_m tp u_m - ip(u_m, u_i) u_i tp u_m)
        ) dot Z_j^(-1)\
        &<= beta [sum_(m != j)
            e^(beta p_2) (norm(u_m)^2 + norm(u_m)^2 norm(u_i)^2)]
            dot Z_j^(-1)\
        &= 2 beta (n - 1) e^(beta p_2) dot Z_j^(-1)\
        &<= 2 gamma log(n) dot n^(gamma p_2 + 1 - gamma) (1 + o_n (1))\
    $
    where the last inequality follows from @lemma31. Since $gamma p_2 + 1 - gamma < 0$ by assumption, this value goes to $0$ as $n -> infinity$.

    2. When $i = j$, we notice that $bf(P)_(u_i) u_j = 0$, hence 
    $
        bf(R)_2 Z_j^(-1) = e^beta Z_j^(-1) (- u_i tp u_i + I_d)
        = (I_d - u_i tp u_i)(1 + o_n (1))
    $
    When $i != j$, naively bounding gives
    $
        norm(bf(R)_2) Z_j^(-1) 
            &<= e^(beta p_2) Z_j^(-1) (norm(u_i)^2
                + beta norm(bf(P)_(u_i) u_j, size: #50%) norm(u_i)
                + norm(I_d))\
            &<= n^(gamma (p_2 - 1)) (1 + 2 gamma log n + sqrt(d)) (1 + o_n (1)),
    $
    which goes to $0$ as $n -> infinity$ since $gamma > 0$ and $p_2 - 1 < 0$. 

    3. When $i = j$, we have that $bf(U)_2 = 0$ since $bf(P)_(u_i) u_i = 0$, so
    $
        &norm((bf(U)_1 + bf(U)_2) tp bf(V)_j) dot Z_j^(-2)
            = Z_j^(-2) beta norm(bf(P)_(u_i) bf(V)_j) norm(bf(V)_j)\
            &<= Z_j^(-2) beta 
                (sum_(m != j) e^(beta ip(u_j, u_m)) norm(bf(P)_(u_i) u_m))
                dot (e^(beta) + sum_(m != j) e^(beta ip(u_j, u_m)) 
                    norm(u_m))\
            &<= 2 e^(-2 beta) beta (n e^(beta p_2)) (e^beta + n e^(beta p_2)) (1 + o_n (1))\
            &<= 2 gamma log n dot n^(gamma (p_2 - 1) + 1) 
                dot (1 + n^(gamma (p_2 - 1) + 1)) (1 + o_n (1)),
    $ 
    which goes to $0$ as $n -> infinity$ since $gamma(p_2 - 1) + 1 < 0$.
    When $i != j$, $bf(U_1) = 0$, thus
    $
        &norm((bf(U)_1 + bf(U)_2) tp bf(V)_j) dot Z_j^(-2)
        = Z_j^(-2) beta e^(beta ip(u_j, u_i)) norm(bf(P)_(u_i) u_j) norm(bf(V)_j)\
        &<= 2 Z_j^(-2) beta e^(beta p_2) 
            sumo(m,n) e^(beta ip(u_j, u_m)) norm(u_m)\
        &<= 2 beta e^(-2 beta) e^(beta p_2) n e^(beta) (1 + o_n (1))\
        &<= 2 gamma log(n) n^(gamma(p_2 - 1) + 1) (1 + o_n (1)),
    $
    which goes to $0$ as $n -> infinity$ since $gamma(p_2 - 1) + 1 < 0$.
    Combining all cases for $gamma > 1/(1 - p_2)$, we have that
    $
        (pddv((("ATT"(N(x_j)))_v), (x_i)_u))_(u, v = 1)^d
        &<= delta_(i j)/norm(x_i) (I_d - u_i tp u_i) (1 + o_n (1)) \
        &= drd(i, j)/norm(x_i) (I_d - u_i tp u_i) + bf(o)_n (1) + o_n (1) I_d.
    $
    Noticing the block-diagonal structure of $nabla_X X'$ and that $(u_i tp u_i)^2 = u_i tp u_i$, 
    $
        1/(n d) norm(nabla_X X')^2 &= 1/(n d) (sum_(i=1)^n 1/norm(x_i) norm(I_d - u_i tp u_i)^2) + o_n (1)\
        &>= 1/(n d q_2) (sum_(i = 1)^n tr(I_d - u_i tp u_i)) + o_n (1)\
        &= 1/(n d q_2) (sumin d - tr(u_i u_i^T)) + o_n (1)\
        &= 1/q_2 (1 - 1/d) + o_n (1).
    $

    Next, consider the case when $gamma < 1/(1 - p_2)$.

    1. Once again, if $i != j$, $bf(R_1) Z_j^(-1) = 0$. When $i = j$, we first note that $norm(bf(P)_(u_i) u_m) <= 1$, since $bf(P)_(u_i) u_m$ gives the component of $u_m$ orthogonal to $u_i$. Thus,
    
    $
        bf(R)_1 Z_j^(-1)

        &= norm(
           beta
           sum_(m = 1)^n 
           e^(beta ip(u_j, u_m)) (u_m tp u_m - ip(u_m, u_i) u_i tp u_m)
        ) dot Z_j^(-1)\

        &= beta norm(
            sumo(m, n) 
            e^(beta ip(u_j, u_m)) (bf(P)_(u_i) u_m) tp u_m
        ) dot Z_j^(-1)\

        &<= beta Z_j^(-1) sumo(m,n) e^(beta ip(u_j, u_m)) norm(bf(P)_(u_i) u_m) norm(u_m)\
        &<= beta
    $
    2. As computed earlier, $norm(I_d - u_i tp u_i) = sqrt(d - 1)$, giving
    
    $
        norm(bf(R)_2) Z_j^(-1) 
            &<= Z_j^(-1) e^(beta ip(u_j, u_i))
                norm((-u_i + beta bf(P)_(u_i) u_j) tp u_i + I_d)\

            &<= Z_j^(-1) e^(beta ip(u_j, u_i)) (norm(I_d - u_i tp u_i) + beta norm(bf(P)_(u_i) u_j) norm(u_i))\

            &<= Z_j^(-1) e^(beta ip(u_j, u_i)) (sqrt(d - 1) + beta).
    $

    3. Note that $norm(bf(V)_j) <= Z_j$, and $norm(bf(P)_(u_i) bf(V)_j) <= norm(bf(V)_j) = Z_j$.

    $
        norm((bf(U)_1 + bf(U)_2) tp bf(V)_j) dot Z_j^(-2)

        &<= (norm(bf(U)_1) + norm(bf(U)_2)) dot Z_j^(-1)\

        &<= Z_j^(-1) drd(i, j) beta norm(bf(P)_(u_i) bf(V)_j) 
            + beta e^(beta ip(u_j, u_i)) norm(bf(P)_(u_i) u_j) dot Z_j^(-1)\
        &<= beta (drd(i, j) + e^(beta ip(u_j, u_i)) Z_j^(-1)).
    $
    All together, this gives
    $
        &norm((pddv((("ATT"(N(x_j)))_v), (x_i)_u))_(u, v = 1)^d)\

        &<= 1/norm(x_i) (
            drd(i, j) beta 
            + Z_j^(-1) e^(beta ip(u_j, u_i)) (sqrt(d - 1) + beta)
            + beta(drd(i, j) + e^(beta ip(u_j, u_i)) Z_j^(-1))
        )\
        
        &= 1/norm(x_i) (2 beta drd(i, j) 
            + (2 beta + sqrt(d)) e^(beta ip(u_j, u_i)) Z_j^(-1)).\
    $
    Using the properties of $Z_j$ as shown in @lemma31,
    $
        &1/(n d) norm(nabla_X X')^2 \

        &<= 1/(n d) sum_(i, j = 1)^n 
            1/norm(x_i) (4 beta^2 drd(i, j) 
                + 4 beta drd(i, j) (2 beta + sqrt(d))
                    e^(beta ip(u_j, u_i))/Z_j
                + (2 beta + sqrt(d))^2
                    dot e^(2 beta ip(u_j, u_i))/Z_j^(2))\

        &<= 1/(n d q_1) (
            n dot 4 beta^2
            + O((log n)^2) (sumjn e^(beta)/Z_j)
            + O((log n)^2) (
                sum_(i,j=1)^n 
                    (e^(2 beta ip(u_j, u_i)))/Z_j^2
            )
        )\

        &<= ((1 + o_n (1)))/(n d q_1) (
            4 n beta^2
            + O((log n)^2) n^(gamma + 1)/n^(gamma p_1 + 1)
            + O((log n)^2) dot sumjn (
                n^gamma/(n^(gamma p_1 + 1)) 
                dot sumin e^(beta ip(u_j, u_i))/Z_j
            )
            )\

        &= ((1 + o_n (1)))/(d q_1) lrs((
            4 beta^2
            + O((log n)^2) n^(gamma(1 - p_1) - 1)
            + O((log n)^2) n^(gamma(1 - p_1) - 1)
            )
        )\

        &= 4 (gamma^2 (log n)^2)/(d q_1) + o_n (1),
    $
    since $gamma(1 - p_1) - 1 < 0$.
]

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

= Attention at Initialization

A natural next step after studying the equicorrelated setting is to consider the fully random case, where the inputs no longer share a common correlation structure and instead behave like independent high-dimensional noise.
This serves as a baseline model in which any structure in the attention scores comes purely from random fluctuations, most importantly in the case of a newly initialized, pre-trained model. The purpose of this regime is not to improve upon the equicorrelated analysis, but to provide a contrasting basline.

== Independent Entries

To identify a potential scaling for the full model, we first restrict ourselves to the simpler independent case. Consider a score vector $s = (s_1, ..., s_n)$ with independent and identically distributed entries $s_i tilde cal(N)(0, 1)$.
If $s_m$ is the maximum score, the maximum attention weight is given by
$
    a_m = e^(beta s_m) / (sumkn e^(beta s_k)) = (1 + (sum_(k != m) e^(beta s_k))/e^(s_m))^(-1)
$
For large $n$, we expect
$
    sum_(k != m) e^(beta s_k) tilde n EE[e^(beta s_1)] = n e^(beta^2\/2) = exp(log n + beta^2 \/ 2).
$
It can also be shown that $s_m tilde sqrt(2 log n)$ @leadbetter1983extremes and hence $e^(beta s_m) tilde e^(beta sqrt(2 log n))$. A scaling for which $a_m in (0, 1)$ must then have
$
    log n + beta^2/2 = beta sqrt(2 log n),
$
which yields $beta_n = sqrt(2 log n)$ after solving. Although a rough asymptotic argument, this identifies $beta_n asymp sqrt(log n)$ as the critical scaling factor under this i.i.d. assumption.

== Correlated Entries

We now generalize to a model that resembles the attention mechanism more closely. Consider the $n times n$ score matrix
obtained from the multiplication

$
    S = [((X W_Q) (X W_K)^T)/sqrt(d)],
$
where $X in RR^(n times d)$ is a random matrix with i.i.d. $cal(N)(0, 1)$ components, and $W_Q, W_K in RR^(d times d)$ are random matrices with i.i.d. $cal(N)(0, 1\/d)$ entries.

In this context, $S$ is a random object representing what score values may be produced by a randomly initialized model prior to training. For this reason, the score scaling $beta$ may equivalently be viewed as a scaling factor for the variance of i.i.d. $cal(N)(0, beta\/d)$ entries in the $W_Q$ and $W_K$ matrices. This analysis is therefore also particularly useful in the effort of counteracting pathological behaviour at the beginning of training, which could in the worst case lead to an untrainable model.
Since the methods used in this section are closely related to those of statistical physics, we will rely on asymptotic approximations and high-probability concentration results, as opposed to the exact results found in other sections.

#definition[
    Given attention weights $a_i in RR^n$, define the _inverse participation ratio_
    $
        Y^((2))_(i) = sum_(j = 1)^n a_(i j)^2.
    $
]
The inverse participation ratio heuristically satisfies
$
    Y^((2))_(i) = 1 / ("effective support size of" a_i),
$
since $Y^((2))_(i) = 1$ if $a_i$ has a single coordinate with value $1$, and $Y^((2))_i = 1/n$ if $a_i$ is uniform. We are specifically interested in a scaling $beta$ that avoids these two cases in the large-$n$ limit.
#stheorem(name: "Phase Transition for IPR", source: [@giorlandino2025failure])[
    Under the scaling $beta_n = gamma sqrt(log n)$ where $gamma > 0$ is a constant, the inverse participation ratio approximately satisfies
    $
        limn EE[Y_i^((2)) (beta)]
            = cases(
                0"," quad &gamma < sqrt(2)",",
                1 - sqrt(2)/gamma"," quad &gamma > sqrt(2),
            )
    $
    given that the embedding dimension $d$ is sufficiently large.
    Thus, under this model, the critical scaling is of order $beta asymp sqrt(log n)$ with phase transition at $beta = sqrt(2 log n)$.
]<theorem-correlated>

#lemma("Concentration of Token Similarity")[
    For two rows $x_i$, $x_j$ of the embedding matrix $X$, define the _token similarity_ $q_(i j) = ip(x_i, x_j)\/d$. Then,
    $
        q_(i j) ->^p
            cases(
                1"," quad &i = j",",
                0"," quad &i != j.
            )
    $
    In particular, we may assume approximate deterministic equality for large $d$, such as those used in modern language models.
]<concentration>

#proof[
    For $i = j$, we have
    $
        q_(i i) = 1/d sumo(k, d) (x_i)_k ^2.
    $
    By the Law of Large Numbers, $q_(i i) ->^p EE[(x_i)_1^2] = 1$, since $(x_i)_k tilde^("i.i.d.") cal(N)(0, 1).$ For $i != j$,
    $
        EE[q_(i j)] = 1/d sumo(k, d) EE[(x_i)_k (x_j)_k] = 0,
    $
    and
    $
        "Var"(q_(i j)) = 1/d dot EE[(x_i)_k^2 (x_j)_k^2] = 1/d.
    $
    By Chebyshev's Inequality, $q_(i j) -> 0$. Moreover, sharper concentration results (see @vershynin2026) imply that these quantities are tightly concentrated with high probability for large $d$.
]


#lemma("Stein's Lemma")[
    Let $Z = (Z_1, ..., Z_n)^T tilde cal(N)(0, V)$ and $g: RR^n -> RR$ such that $g in C^1$ and $EE[norm(g(Z))], EE[norm(nabla g(Z))] < infinity$. Then,
    $
        EE[Z_i thin g(Z)] =  sum_(j=1)^n "Cov"(Z_i, Z_j) dot EE[pdv(Z_j) g(Z)]
    $
]

#proof[
    The pdf of $Z$ has the form $p(z) = C exp[-1/2 z^T V^(-1) z]$. Thus,
    $
        nabla p(z) &= - V^(-1) z p(z)\
        => z p(z) &= - V nabla p(z)\
        => z_i p(z) &= - sum_(j=1)^n V_(i j) pdv(z_j) p(z).
    $
    Substituting the above equation into the calculation of expectation,
    $
        EE[Z_i g(Z)] = lint(RR^n) z_i g(z) p(z) thin d z = - sumjn V_(i j) lint(RR^n) g(z) pdv(z_j) p(z) thin d z.
    $
    We then apply integration by parts along the dimension of $z_j$, noting that $g(z)$ vanishes when $z_j -> plus.minus infinity$ by assumption of finite expectation. This gives
    $
        EE[Z_i g(Z)] = sumjn V_(i j) lint(RR^n) pdv(z_j) g(z) p(z) thin d z
        = sumjn "Cov"(Z_i, Z_j) dot EE[pdv(z_j) g(Z)].
        
    $
]

#lemma("Derivative of Scaled Softmax")[
    Let $S in RR^(n times n)$. For any $i, j, k in {1, ..., n}$,
    $
        pddv(a_(i j), s_(i k)) = pdv(s_(i k)) ((e^(h beta s_(i j)))/(sum_(l = 1)^n e^(h beta s_(i l))))
        = h beta a_(i j) (delta_(j k) - a_(i k)).
    $
]

#proof[
    The derivative of the numerator is
    $
       pdv(s_(i k)) e^(h beta s_(i j)) = h beta e^(h beta s_(i j)) delta_(j k).
    $
    The derivative of the denominator $Z_i$ is 
    $
       pdv(s_(i k)) Z_i = h beta e^(h beta s^(i k)).
    $
    By quotient rule,
    $
        pdv(s_(i k)) ((e^(h beta s_(i j)))/(sum_(l = 1)^n e^(h beta s_(i l))))
        = (h beta e^(h beta s_(i j)) delta_(j k) Z  - e^(h beta s_(i j)) h beta e^(h beta s^(i k))) / Z^2
        = h beta a_(i j) (delta_(j k) - a_(i k)).
    $
]

#lemma[
    For large $d$, we have
    $
        s_(i j) tilde cal(N)(0, 1),
    $
    with covariance structure
    $
        "Cov"(s_(i j), s_(p q)) = q_(i p) q_(j q).
    $
]

#proof[
    By definition, 
    
    $
        s_(i j) &= 1/sqrt(d) sum_(k=1)^d (W_Q x_i)_k (W_k x_j)_k\
        &= 1/sqrt(d) sum_(k=1)^d
            sum_(a, b = 1)^d (W_Q)_(k a) (x_i)_a (W_K)_(k b) (x_j)_b\
        &:= 1/sqrt(d) sum_(k=1)^d D_k.
    $
    First, $EE[D_k] = 0$ since all entries of $W_K$ and $W_Q$ have expectation $0$. As for variance,
     
    $
        EE[D_k^2] &= EE[(sum_(a = 1)^d (W_Q)_(k a) (x_i)_a)^2] dot EE[(sum_(b=1)^d (W_K)_(k b) (x_j)_b)^2]\
        &= (sum_(a=1)^d EE[(W_Q)_(k a)^2] (x_i)_a^2) (sum_(b=1)^d EE[(W_K)_(k a)^2] (x_j)_b^2)\
        &= 1/d^2 norm(x_i)^2 norm(x_j, size: #80%)^2\
    $
    We have that $norm(x_k)^2/d ->^p 1$ by @concentration, hence by the Central Limit Theorem, $s_(i j) ->^d cal(N)(0, 1)$. 
    A similar calculation for covariance yields

    $
        "Cov"(s_(i j), s_(p q)) &= EE[s_(i j) s_(p q)]\
        &= 1/d sum_(k, a, b, l, alpha, beta = 1)^d
            (x_i)_a (x_j)_b (x_p)_alpha (x_q)_beta EE[(W_Q)_(k a) (W_Q)_(l alpha)] EE[(W_K)_(k b) (W_K)_(l beta)]\
        &= 1/d sum_(k, a, b = 1)^d
            (x_i)_a (x_j)_b (x_p)_a (x_q)_b EE[(W_Q)_(k a)^2] EE[(W_K)_(k a)^2]\
        &= 1/d^3 sum_(k=1)^d ip(x_i, x_p) ip(x_j, x_q)\
        &= q_(i p) q_(j q).
    $
]



#lemma[
    Define the function
    $
        Phi_n (beta, h) = EE[log Z_i (beta, h)], "    " Z_i (beta, h) = sumo(j, n) e^(h beta s_(i j)).
    $
    Then,
    $
        limn EE[Y^((2))_i] = 1 - lim_(h -> 1) limn pdv(h) 1/beta^2 Phi_n (beta, h).
    $
]

#proof[
    We first note that we can safely swap derivative and expectation in the case of 
    $
        pdv(h) Phi_n (beta, h) = EE[pdv(h) log Z_i (beta, h)]
            = EE[(sumjn beta s_(i j) e^(h beta s_(i j)))/(sumkn e^(h beta s_(i k)))]
            = beta sumjn EE[s_(i j) a_(i j)],
    $
    where we temporarily use the notation $a_(i j) = exp(h beta s_(i j)) \/ Z_i (beta, h)$.
    Since this is a sum of elements of the form $EE[s_(i j) g(s_i)]$ where $s_i in RR^n$ is normally distributed, we apply Stein's Lemma to obtain
    $
        pdv(h) Phi_n (beta, h) &=
            beta sumjn sumkn "Cov"(s_(i j), s_(i k)) EE[pddv(a_(i j), s_(i k))]\
            &= beta^2 sumjn sumkn q_(i i) q_(j k) EE[h beta a_(i j) (delta_(j k) - a_(i k))].
    $
    Define $Y_(i, h)^((2)) = sum_(j = 1)^n a_(i j)^2$. By @concentration, $q_(i i) ->^p 1$ and $q_(i j) ->^p 0$ for $i != j$. This gives
    $ 
        pdv(h) Phi_n (beta, h) 
            &approx beta^2 h sumjn sumkn
                delta_(j k) 
                EE[a_(i j) delta_(j k) - a_(i j) a_(i k)]\
            &= beta^2 h sumjn
                EE[a_(i j)
                - a_(i j)^2
                ]\
            &= beta^2 h (1 - EE[Y_(i, h)^((2))]).
    $
    Rearranging and taking limits gives
    $
        limn EE[Y_(i)^((2))] = 
        1 - lim_(h -> 1) limn pdv(h) 1/beta^2 Phi_n (beta, h).
    $
]

We are now left with computing $Phi_n (beta, h) = EE[log Z_i (beta, h)]$. This is a problem similar to the well-studied Random Energy Model from statistical physics, for which methods exist. In particular, we use the replica method, which is based on the applicaiton of the limit

$
    EE[log Z] = lim_(t -> 0^+) (EE[Z^t] - 1)/t.
$

In general, the replica method is not fully mathematically rigorous, especially in its use of analytic continuation and assumptions about the structure of overlap matrices. Despite this, it is able to provide accurate predictions in a wide range of disordered systems and high-dimensional probabilistic models.


#proof([of @theorem-correlated])[
    First, we consider the replicated system
    $
        EE[Z_i^t (beta, h)] &= EE[(sumkn e^(h beta s_(i k)))]^t
        = sum_(k_1, ..., k_t = 1)^n EE[exp(h beta sumo(a, t) s_(i k_a))].
    $
    Using the moment-generating function of the multivariate normal distribution,
    $
        EE[Z_i^t (beta, h)] 
            &= sum_(k_1, ..., k_t = 1)^n 
                exp((h^2 beta^2)/2 sum_(a, b = 1)^t EE[s_(i k_a) s_(i k_b)])\
            &= sum_(k_1, ..., k_t = 1)^n
                exp((h^2 beta^2)/2 
                    sum_(a, b = 1)^t delta_(k_a k_b)
                )
    $
    Define the empirical overlap matrix $Q$ with entries $Q_(a b) = delta_(k_a k_b)$. 
    Among all $n^t$ possible configurations of replica states, there are 
    $
        S(Q) = sum_(k_1, ..., k_t = 1)^n product_(a, b = 1)^t ind(Q_(a b) = delta_(k_a k_b))
    $
    with a specific overlap matrix $Q$. Thus,
    $
        EE[Z_i^t (beta, h)] = 
            sum_Q S(Q)
                exp((h^2 beta^2)/2 
                    sum_(a, b = 1)^t Q_(a b)
                )
            = sum_Q exp(log S(Q) + (h^2 beta^2)/2 sum_(a, b = 1)^t Q_(a b)).
    $
    For large $n$, this sum is dominated by overlap structure $Q$ that maximizes the exponent, in acordance with the Laplace principle. To evaluate this dominant contribution, we restrict attention to a class of structured overlap matrices, which is the _1-RSB Ansatz_ step in the Replica method. 

    By a symmetric argument, we assume that for this maximizing $Q$, the $t$ states across the replicas are divided into $t / x$ groups of $x$ elements each, where replicas within the same group occupy the same state, i.e. $k_a = k_b$ if and only if replicas $a$ and $b$ are in the same group. 

    Although $x$ will ulitimately be extended to real values via analytic continuation, we temporarily assume that relevent quantities are integers in order to analyze the system combinatorically.
    For any maximizing $Q$ under this assumption,
    $
        S(Q) = n dot (n - 1) thin dots.c thin (n - t/x + 1) dot (t!)/((x!)^(t\/x) (t\/x!)),
    $
    so $log S(Q) approx t/x log n$ for large $n$, since $t, x << n$. Also,
    $
        sum_(a, b = 1)^t Q_(a b) = sum_("groups") x^2 = t x,
    $ 
    which gives after Taylor expansion
    $
        (EE[Z^t_i (beta, h)] - 1)/t = 1/x log n + (h^2 beta^2)/2 x + O(t^2).
    $
    Determining the value of $x$ corresponding to this maximizing $Q$ reduces, in this representation, to a minimization over $x$, the reason for which lies beyond the scope of this derivation. In the analytic continuation with $t -> 0^+$, we minimize over $x in (0, 1)$, giving
    $
        x = min{1/(h beta) sqrt(2 log n), 1}.
    $
    Using this,
    $
        Phi_n (beta, h) = lim_(t -> 0^+) [(EE[Z^t_i (beta, h)] - 1)/t] = 
            cases(
                log n + (h^2 beta^2)/2"," quad &beta < sqrt(2 log n)/h",",
                h beta sqrt(2 log n)"," quad &beta > sqrt(2 log n)/h"."
            )
    $
    We now assume $beta_n = gamma sqrt(log n)$. Thus,
    $
        pdv(h) 1/beta^2 Phi_n (beta, h)
            = cases(
                h"," quad &gamma < sqrt(2),
                (h sqrt(2))/gamma"," quad &gamma > sqrt(2).
            )
    $

    By the previous lemma,
    $
        limn EE[Y_i^((2)) (beta)]
            = cases(
                0"," quad &gamma < sqrt(2)",",
                1 - sqrt(2)/gamma"," quad &gamma > sqrt(2).
            )
    $
]

#corollary("Update Average Cosine Similarity")[
    Define the _updated token matrix_
    $
        Y = A X W_V,
    $
    where $A = "softmax"(S)$, applied row-wise, and $W_V in RR^(d times d)$ is a random matrix with i.i.d. $cal(N)(0, 1\/d)$ entries. For sufficiently large embedding dimension $d$, the updated token similarity matrix approximately satisfies
    $
        limn EE[(Y Y^T)/d] = cases(
            0"," quad &gamma < sqrt(2)",",
            (1 - sqrt(2)/gamma) I_d"," quad &gamma > sqrt(2).
        )
    $
    In other words, $norm(y_i)^2 \/d$

    ...
]
#proof[
    First observe that $EE[W_V W_V^T] = I_d$ by a simple calculation and for large $d$, $X X^T approx I_n$ by @concentration.
    Since $W_V$ and $a_i$ are indepdendent,
    $
        EE[ip(y_i, y_j)/d] 
            &= 1/d dot EE[(a_i X W_V) (a_j X W_V)^T]\
            &= 1/d dot EE[a_i X thin EE[W_V W_V^T] X^T a_j^T]\
            &approx EE[a_i a_j^T]\
            = delta_(i j) Y^(2)_i
    $

]


= Deterministic Score Theory

This section contains results found in @hayase2026.

== Scaling Order

We now concern ourselves with a fixed score vector $s in RR^n$. 
The following analysis is independent of the attention mechanism, in the sense that the scores are taken as given, where the specific token vectors and weight matrices used to obtain these scores are not of concern.
Let $s^*$ be the maximum of such scores, and define the _gaps_ $Delta_j = s^* - s_j$ from each score to the maximum.
We refer to scores as _competitors_, as they are "competing" for the attention weight that is to be distributed by softmax.
Thus, the attention weight of any given competitor can be rewritten as
$
    a_j (beta) = e^(- beta Delta_j) / (sumkn e^(- beta Delta_k)),
$
which is equivalent to scaled softmax applied to the new score vector $Delta$ with maximum $0$. We define $Z(beta) = sumjn exp(- beta Delta_j)$ as in previous sections, while noting that it is defined on gap scores and not raw scores in this context.


#definition("CGF")[
    The _cumulative gap-counting function_ (CGF) is the number of competitors with gap at most $t$ from the maximum, 
    $
        N(t) = sum_(j=1)^n ind(Delta_j <= t).
    $
]

Now, $Z(beta)$ can be rewritten as the Laplace transform of the CGF, since

$
    Z(beta) 
        = sumjn e^(-beta Delta_j)
        = sumjn beta int(0, inf) ind(Delta_j <= t) e^(-beta t) thin d t
        = beta int(0, inf) e^(-beta t) N(t) thin d t.
$
This is clearly well-defined on finite $n$ since $N(t) <= n$. However, we immediately observe that if $N(t)$ is not of exponential order, the attention weights tend toward uniformity.

#definition[
    The _upper tail accumulation scale_ $Lambda$ is the smallest exponent for which $N(t) <= exp(Lambda t)$ for all $t > 0$. Equivalently,
    $
        Lambda = sup_(t > 0) (log N(t)) / t.
    $
    When $Lambda < inf$, a gap $Delta_j$ is called a _contact gap_ if 
    $N(Delta_j) = e^(Lambda Delta_j)$. Let $Delta_Lambda$ be the largest of all such contact gaps and define the _contact accumulation exponent_ 
    $
        alpha = (log N(Delta_Lambda)) / (log n) in [0, 1], 
    $
    which satisfies $N(Delta_Lambda) = n^alpha$.
]

#lemma[
    The Shannon entropy $H(beta) = - sum_(j=1)^n a_j (beta) log (a_j (beta))$ satisfies
    $
        H(beta) = log Z(beta) - beta (Z'(beta))/Z(beta)
    $
]<lemma:shannon>


#proof[
    Recall that $sumjn a_j (beta) = 1$. Thus,
    $
        H(beta) 
            = - sumjn a_j (beta) (-beta Delta_j - log Z(beta))
            = log Z(beta) + sumjn beta Delta_j (e^(-beta Delta_j))/(Z (beta))
            = log Z(beta) - beta (Z'(beta)) / Z(beta).
    $
]

#stheorem(name: "Sequence Collapse Criterion", source: [@hayase2026])[
    For each $n >= 2$, let $s^((n)) in RR^n$ be a score vector of length $n$ with corresponding upper tail accumulation scale $Lambda_n$. For any positive sequence $(beta_n)$,

    1. If $beta_n \/ Lambda_n -> 0$, then _top-two collapse_ holds, i.e.
    $
        limn D_n (beta_n) = 0,
    $
    where $D_n (beta_n)$ is the difference between the largest and second largest attention weights $a^((n)) (beta_n)$.

    2. If $beta_n \/ Lambda_n -> infinity$, then _entropy collapse_ holds, i.e.
    $
        limn H_n (beta_n) = 0,
    $
    where is the Shannon entropy of the attention weights $a^((n)) (beta_n)$.
]

#proof[
    Let $r_n = beta_n \/ Lambda_n$.

    #text[1.] Let $Delta^((n))_"min"$ be the smallest non-zero gap, which necessarily has $Delta^((n))_"min" <= Delta^((n))_Lambda$. The corresponding difference in attention weights can be bounded using $1 - e^(-x) <= x$, giving
    $
        D_n (beta_n) 
        = (1 - exp(-beta_n Delta^((n))_"min")) / Z(beta_n)
        <= (beta_n Delta_"min"^((n))) / Z(beta_n)
        <= (beta_n Delta^((n))_Lambda) / Z(beta_n).
    $
    By definition, we have that $log (N(Delta^((n))_Lambda)) = Lambda_n Delta_Lambda^((n)) = beta_n Delta_Lambda^((n)) \/ r_n$. Substituting into and rearranging the definition of $alpha_n$ gives
    $beta_n Delta^((n))_Lambda = r_n alpha_n log n$.
    Since $e^(- beta_n t)$ is monotone on the $N(Delta^((n))_Lambda)$ gaps with $Delta_j <= Delta^((n))_Lambda$, we have that
    
    $
        Z(beta_n) &>= 
            N(Delta^((n))_Lambda) 
            exp(-beta_n Delta^((n))_Lambda)\
            &= exp(beta_n Delta_Lambda^((n)) \/ r_n) n^(r_n alpha_n)\
            &= n^(alpha_n (1 - r_n)).
    $
    Thus,
    
    $
        D_n (beta_n) 
            <= (beta_n Delta_Lambda^((n))) / n^(alpha_n (1 - r_n))
            = r_n alpha_n log n dot n^(- alpha_n (1 - r_n)).
    $
    Since $t e^(-t) <= e^(-1)$ for all $t in RR$, setting $t = (1 - r_n) dot alpha_n log n$ yields
    $
        D_n (beta_n) <= r_n / (1 - r_n) dot e^(-1),
    $
    which goes to $0$ as $n -> infinity$ since $r_n -> 0$ by assumption.

    #text[2.] First observe that $Z(beta_n) >= 1$, since the gap between the maximum score and itself is always $0$. Using the previously established integral form for $Z(beta_n)$ as well as the fact that $N(t) <= exp(Lambda^((n)) t)$ by definition,
    $
        Z(beta_n) 
            = beta_n int(0, inf) e^(-beta_n t) N(t) thin d t
            <= beta_n int(0, inf) e^(-(beta_n - Lambda_n) t) thin d t
            = beta_n / (beta_n - Lambda_n)
            = (r_n) / (r_n - 1),
    $
    which goes to $1$ as $n -> infinity$ since $r_n -> inf$. Thus, $Z(beta_n) -> 1$. 

    For any $n$, differentiating the integral form with respect to $beta$ gives
    
    $
        Z'(beta) 
            = int(0, inf) e^(-beta t) N(t) thin d t
            + beta d/(d beta) (int(0, inf) e^(-beta t) N(t) thin d t)
            = beta^(-1) Z (beta) - beta int(0, inf) t e^(-beta t) N(t) thin d t,
    $
    which implies that
    $
        - beta Z' (beta) 
            &= beta^2 int(0, inf) t e^(-beta t) N(t) thin d t 
                - Z(beta)\
            &<= beta^2 int(0, inf) t e^(-(beta - Lambda_n) t) thin d t
                - Z(beta)\
            &= (beta / (beta - Lambda_n))^2 - (beta / (beta - Lambda_n)).
    $
    Since $(beta_n \/ (beta_n - Lambda_n)) -> 1$ and 
    $-beta Z' (beta) = beta sumjn Delta_j exp(- beta Delta_j) >= 0$, 
    we have that $- beta Z'(beta) -> 0$. 
    By @lemma:shannon,
    $
        limn H(beta_n) 
            = limn [log Z(beta_n) + 1/Z(beta_n) (- beta (Z' (beta)))]
            = 0.
    $
]

This establishes $Lambda_n$ as the order of the critical scaling for $s^((n))$.
Denote $a_n asymp b_n$ if there exists constants $c, C > 0$ such that for large enough $n$, $c b_n <= a_n <= C b_n$.

#corollary("Critical Scaling Exponent")[
    Let $xi_Lambda > 0$. If $Lambda_n asymp (log n)^(xi_Lambda)$, then $xi_Lambda$ is unique and any non-collapsing scaling $beta_n$ must have $beta_n asymp (log n)^(xi_Lambda)$.

    Furthermore, if $alpha_n asymp (log n)^(xi_alpha)$ and $Delta_Lambda^((n)) asymp (log n)^(xi_Delta)$, then
    $
        xi_Lambda = xi_alpha - xi_Delta + 1.
    $
]<c511>

#proof[
    If $Lambda_n asymp (log n)^(xi')$ for $xi' > 0$, then $1 asymp (log n)^(xi_Lambda - xi')$, which is only possible when $xi_Lambda = xi'$. For a non-collapsing scaling, we must have $beta_n \/ Lambda_n asymp 1$, which is only possible if $beta_n asymp (log n)^(xi_Lambda)$.

    Since $Lambda_n = (alpha_n log n)\/Delta_Lambda^((n))$, 
    $
        Lambda_n asymp ((log n)^(xi_alpha) (log n))/((log n)^(xi_Delta))
        = (log n)^(xi_alpha - xi_Delta + 1),
    $
    and by uniqueness of this exponent must mean that $xi_Lambda = xi_alpha - xi_Delta + 1$.
]

Consider the Simplex case from @sec:simplex. For any row $i$ of the $n times n$ similarity matrix,

$
    a_(i j) = cases(q", " i = j",", p", " i != j.)
$
The gap counting function is given by
$
    N(t) = cases(1", " &t < q - p",", n", " &t >= q - p)
$
which immediately gives $Lambda_n = (q-p)^(-1) log(n)$, $alpha_n = 1$, $Delta_Lambda^((n)) = q - p$. Thus,
$xi_Lambda = 1$, $xi_alpha = 0$, and $xi_Delta = 0$, which satisfy the relation derived in @c511. Furthermore, we see that the non-collapsing scaling has $beta_n asymp log n$, which validates @t31.

== Rank-Collapse Boundary

#proposition("Ties at the maximum")[
    Let $s in RR^n$. If $N(0) >= 2$, then $Lambda = inf$ and for every $beta > 0$, 
    $
        norm(a (beta))_inf <= 1/N(0), quad H(beta) >= log N(0).
    $
    In particular, for a sequence of score vectors $(s^((n)))$ with $s^((n)) in RR^n$ and $N(0) >= 2$ for infinitely many $n$, no sequence $X_n$ exists such that $beta_n / X_n -> inf$ implies that $H(beta_n) -> 0$.
]
#proof[
    Since $N(0) >= 2$ and $N$ is non-decreasing, we have that
    $
        Lambda = sup_(t > 0) (log N(t))/t >= sup_(t > 0) (log 2) / t = inf.
    $
    The top $N(0)$ post-softmax weights are equal and have total mass at most 1, thus $norm(a(beta))_inf <= 1\/N(0)$. This gives entropy bound
    $
        H(beta) >= sumjn a_j (beta) log(N(0)) = log N(0) >= log 2,
    $
    since post-softmax weights sum to 1. The conclusion for the sequence $(s^((n)))$ follows from the fact that $H(beta) >= log 2$ for infinitely many $n$.
]

#definition[
    Given a score vector $s in RR^n$, define the _rank free energy_ $cal(F) (beta) = log Z(beta)$ and the _maximum post-softmax gap_ $cal(G)(beta) = max_(j,k) |a_j (beta) - a_k (beta)|$.
]

#proposition("Rank-Collapse Criterion")[
    Let $(s^((n)))$ be a sequence of score vectors such that $s^((n)) in RR^n$. For any positive sequence $(beta_n)$, the following are equivalent:
    1. $cal(F) (beta_n) -> infinity$,
    2. $cal(G) (beta_n) -> 0$,
    3. $max_j |a_j^((n)) (beta_n) - 1\/n| -> 0$.
]<rank-collapse-criterion>

#proof[
    For simplicity, we omit the argument $beta$ from the post-softmax weights $a^((n))_j (beta)$.
    The maximum value of $s$ has $Delta_j = 0$, so 
    $
        max_j a^((n))_j = 1 / (Z (beta)) = e^(-cal(F) (beta)).
    $
    Since $cal(G) (beta) <= max_j a^((n))_j$, $cal(F) (beta_n) -> inf$ implies that $cal(G) (beta_n) -> 0$.

    Conversely, by a basic property of the arithmetic mean, we have the bound $min_j a^((n))_j (beta) <= 1/n$, which gives
    $
        max_j a^((n))_j = max_j a^((n))_j - min_k a^((n))_k + min_k a^((n))_k <= cal(G) (beta) + 1/n.
    $
    Thus, $cal(G) (beta_n) -> 0$ implies that $max_j a_j^((n)) -> 0$, which is equivalent to $cal(F) (beta) -> infinity$.

    Finally, we observe that $max_j a_j^((n)) -> 0$ is equivalent to $max_j |a_j^((n)) (beta_n) - 1\/n| -> 0$, since $0 <= a^((n))_k <= max_j a^((n))_j$ and $1\/n -> 0$.
]

== Resolved Accumulation

#definition("Rank Boundary")[
    For $0 <= r <= log n$, define
    $
        B(r) = sup{beta >= 0 : cal(F) (beta) >= r} = sup{beta >= 0 : Z (beta) >= e^r}.
    $

]

#definition("Resolved Accumulation Scale")[
    For $r >= 0$ and a fixed $s in RR^n$, define
    $
        Lambda^((r)) = sup_(t > 0 : log N(t) > r) (log N(t) - r)/t,
    $
    with convention $sup nothing = 0$.
]

#proposition[
    Let $(r_n)$ be a positive sequence such that $r_n -> infinity$. If $beta_n < Lambda_n^((r_n))$ eventually, then $cal(G)(beta_n) -> 0$.
]

#proof[
    By assumption, for large enough $n$ we have $beta_n < Lambda_n^((r_n))$. Hence, there is some $t_n$ such that
    $
        beta_n < (log N(t_n) - r_n)/t_n,
    $
    since $Lambda_n^((r_n))$ is defined in terms of a supremum. Rearranging,
    $
         N (t_n) e^(-beta_n t_n) > e^(r_n).
    $
    Since $N(t_n)$ competitors each contribute at least $e^(-beta_n t_n)$ to $Z(beta_n)$, we have that
    $
        Z(beta_n) > e^(r_n),
    $
    which gives that $cal(F)(beta_n) -> inf$, and thus $cal(G)(beta_n) -> 0$ by @rank-collapse-criterion.
]

#proposition[
    Define the Laplace envelope 
    $
        S(beta) = sup_(t > 0) lrs((log N(t) - beta t)).
    $
    For every $beta > 0$,
    $
        S(beta) <= cal(F)(beta) <= S(beta) + log(1 + log n).
    $
]

#proof[
    Fix $t > 0$. The $N(t)$ tokens with $Delta_j <= t$ each contribute at least $e^(-beta t)$, thus
    $
        N(t) e^(-beta t) <= Z(beta),
    $
    and hence $log N(t) - beta t <= cal(F) (beta)$. After taking supremums, this gives $S(beta) <= cal(F) (beta)$.

    Consider the ordered gaps $0 = Delta_1 <= Delta_2 <= ... <= Delta_n$. Since $j <= N(Delta_j)$, 
    $
        log j - beta Delta_j <= log N(Delta_j) - beta Delta_j <= S(beta),
    $
    by definition of $S$. Thus,
    $
        e^(-beta Delta_j) <= S(beta)/j,
    $
    which when summed over $j$ gives
    $
        Z(beta) <= e^(S(beta)) <= S(beta) sumjn 1/j <= e^(S(beta)) (1 + log n),
    $
    by a classic bound on the harmonic numbers. Taking logarithms on both sides of the inequality yields the desired upper bound for $cal(F)$.
]

#corollary[
    If $beta_n \/ Lambda_n -> 0$ and the _contact-count entropy_ $C_n = log N(Delta^((n))_Lambda) = Lambda_n Delta^((n))_Lambda$ satisfies $C_n -> inf$, then $cal(G) (beta_n) -> 0$.
]

#proof[
    Set $r_n = beta_n \/ Lambda_n$. By the previous proposition,
    $
        Z(beta_n) >= N(Delta_n) e^(-beta_n Delta^((n))_Lambda) = e^(C_n - r_n C_n) = e^((1 - r_n) C_n),
    $
    and thus $cal(F) (beta_n) >= (1 - r_n)C_n$. Since $r_n -> 0$ and $C_n -> inf$, we have that $cal(F) -> inf$, which is equivalent to $cal(G) (beta_n) -> 0$ by @rank-collapse-criterion.
]

#pagebreak()
#bibliography("citations.bib", title: "References")

