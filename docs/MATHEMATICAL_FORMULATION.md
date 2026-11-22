# Mathematical Formulations and Theoretical Foundations
## Sign Language Recognition and Translation Assistant

---

## 1. Problem Formulation and Kinematic Coordinate Topology

A sign language gesture is a spatiotemporal sequence of body configurations. Let the input sequence $\mathbf{X}$ be represented as a discrete-time sequence of normalized 3D anatomical skeletal landmarks:
$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times D}$$
where:
- $T = 30$ denotes the uniform temporal window length ($\approx 1.0\text{ second}$ at $30\text{ FPS}$).
- $D = 201$ features per frame, consisting of:
  - Upper Pose: $25$ landmarks $\times 3$ coordinates $(x, y, z) = 75$ values.
  - Left Hand: $21$ landmarks $\times 3$ coordinates $(x, y, z) = 63$ values.
  - Right Hand: $21$ landmarks $\times 3$ coordinates $(x, y, z) = 63$ values.
  - Total: $75 + 63 + 63 = 201$ features.

The objective is to learn a parameterized mapping $f_\Theta: \mathbb{R}^{T \times D} \to \Delta^{C-1}$, where $\Delta^{C-1}$ is the probability simplex over $C$ sign classes:
$$\hat{\mathbf{y}} = \text{Softmax}(f_\Theta(\mathbf{X})) \in [0, 1]^C, \quad \sum_{k=1}^C \hat{y}_k = 1$$

---

## 2. Invariant Spatial Normalization

To ensure invariant predictions regardless of the signer's distance to the camera and physical placement within the video frame, coordinates undergo affine normalization:

1. **Torso Centering (Translation Invariance)**:
   Let $\mathbf{p}_{11}(t) \in \mathbb{R}^3$ and $\mathbf{p}_{12}(t) \in \mathbb{R}^3$ denote the 3D coordinates of the left and right shoulders at time step $t$. The mid-shoulder centroid is defined as:
   $$\mathbf{p}_{\text{mid}}(t) = \frac{\mathbf{p}_{11}(t) + \mathbf{p}_{12}(t)}{2}$$
   Each landmark coordinate $\mathbf{p}_i(t)$ is centered relative to this anatomical origin:
   $$\mathbf{p}'_i(t) = \mathbf{p}_i(t) - \mathbf{p}_{\text{mid}}(t)$$

2. **Torso Scaling (Scale Invariance)**:
   The characteristic scale factor $s$ is the Euclidean distance between shoulders:
   $$s = \|\mathbf{p}_{11} - \mathbf{p}_{12}\|_2 + \epsilon$$
   where $\epsilon = 10^{-6}$ prevents numerical division by zero. The normalized coordinates are:
   $$\tilde{\mathbf{p}}_i(t) = \frac{\mathbf{p}'_i(t)}{s}$$

---

## 3. Bidirectional Recurrent Neural Network (BiLSTM)

The temporal backbone employs a two-layer Bidirectional Long Short-Term Memory network to capture both past (causal) and future (anti-causal) gesture context.

### 3.1 LSTM Cell Equations (Forward Pass)
For an input $\mathbf{z}_t \in \mathbb{R}^d$ at frame $t$, the forward LSTM unit updates its internal cell state $\mathbf{C}_t \in \mathbb{R}^H$ and hidden state $\mathbf{h}_t \in \mathbb{R}^H$ via five non-linear gating equations:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{z}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f) \quad \text{(Forget Gate)}$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{z}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i) \quad \text{(Input Gate)}$$
$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c \mathbf{z}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c) \quad \text{(Candidate Cell State)}$$
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad \text{(Cell State Update)}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{z}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o) \quad \text{(Output Gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(Hidden State)}$$

where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid activation function.
- $\odot$ denotes the Hadamard (element-wise) product.
- $\mathbf{W}_* \in \mathbb{R}^{H \times d}$ and $\mathbf{U}_* \in \mathbb{R}^{H \times H}$ are input and recurrent weight matrices.

### 3.2 Bidirectional Representation
The forward pass unrolls sequentially from $t=1 \to T$, generating $[\overrightarrow{\mathbf{h}}_1, \dots, \overrightarrow{\mathbf{h}}_T]$, while the backward pass unrolls from $t=T \to 1$, generating $[\overleftarrow{\mathbf{h}}_1, \dots, \overleftarrow{\mathbf{h}}_T]$. The complete temporal hidden state is the channel-wise concatenation:
$$\mathbf{h}_t^{\text{bi}} = \left[ \overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t \right] \in \mathbb{R}^{2H}$$

---

## 4. Temporal Additive Self-Attention Mechanism

Fixed-pooling (e.g., taking the final hidden state $\mathbf{h}_T^{\text{bi}}$) forces the network to retain all gesture information until the final frame, which introduces bottlenecking and recency bias. In sign language, key inflection frames (e.g., peak hand shape or point of contact) convey the primary semantics, while beginning and ending frames represent resting transitions.

We introduce an **Additive (Bahdanau-style) Temporal Attention** layer:

1. **Alignment Energy Score**:
   $$\mathbf{e}_t = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{h}_t^{\text{bi}} + \mathbf{b}_a)$$
   where $\mathbf{W}_a \in \mathbb{R}^{A \times 2H}$ projects the hidden state to attention dimension $A=128$, and $\mathbf{v}_a \in \mathbb{R}^A$ is a learnable context vector.

2. **Normalized Attention Distribution**:
   $$\alpha_t = \frac{\exp(\mathbf{e}_t)}{\sum_{\tau=1}^T \exp(\mathbf{e}_\tau)}, \quad \sum_{t=1}^T \alpha_t = 1, \quad \alpha_t \in (0, 1)$$

3. **Context Vector Pooling**:
   $$\mathbf{c} = \sum_{t=1}^T \alpha_t \mathbf{h}_t^{\text{bi}} \in \mathbb{R}^{2H}$$
   The resulting context vector $\mathbf{c}$ represents an attention-weighted summary of the entire gesture.

---

## 5. Backpropagation Through Time (BPTT) & Gradient Dynamics

### 5.1 Recurrent Gradient Chain Rule
Let $\mathcal{L}$ denote the scalar loss. The total derivative of $\mathcal{L}$ with respect to the hidden state at time $t$ is partitioned into its direct contribution to the attention context $\mathbf{c}$ and its indirect contribution through future time steps:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}} \frac{\partial \mathbf{c}}{\partial \mathbf{h}_t} + \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}$$

Expanding across $T - t$ time steps yields:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \sum_{\tau=t}^T \frac{\partial \mathcal{L}}{\partial \mathbf{c}} \frac{\partial \mathbf{c}}{\partial \mathbf{h}_\tau} \left( \prod_{j=t}^{\tau-1} \frac{\partial \mathbf{h}_{j+1}}{\partial \mathbf{h}_j} \right)$$

### 5.2 Resolution of Vanishing Gradients via Constant Error Carousel
In vanilla RNNs, $\frac{\partial \mathbf{h}_{j+1}}{\partial \mathbf{h}_j} = \mathbf{W}^\top \text{diag}(1 - \mathbf{h}_{j+1}^2)$. Repeated matrix multiplications by $\mathbf{W}$ cause exponential decay (vanishing gradient) when eigenvalues $\lambda_i < 1$, or exponential growth (exploding gradient) when $\lambda_i > 1$.

In the LSTM cell:
$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t$$
Because this relationship is **additive** rather than purely multiplicative, when the network learns to saturate the forget gate $\mathbf{f}_t \approx \mathbf{1}$, the error propagates back through time virtually undiminished ($\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} \approx \mathbf{I}$). This forms an unobstructed **Constant Error Carousel (CEC)**.

### 5.3 Gradient Clipping
To mitigate gradient explosion from unexpected spikes, global gradient norm clipping is enforced:
$$\mathbf{g} \leftarrow \begin{cases} 
\mathbf{g} & \text{if } \|\mathbf{g}\|_2 \le \gamma \\
\frac{\gamma}{\|\mathbf{g}\|_2} \mathbf{g} & \text{otherwise}
\end{cases}$$
where $\gamma = 1.0$.

---

## 6. Regularization and Optimization Formulation

### 6.1 Label Smoothed Cross-Entropy Loss
Given ground truth label $y \in \{0, \dots, C-1\}$, label smoothing prevents overconfidence by replacing the Dirac delta distribution $\delta_{k, y}$ with a mixture distribution:
$$q(k) = (1 - \epsilon)\delta_{k, y} + \frac{\epsilon}{C}$$
The regularized cross-entropy loss is:
$$\mathcal{L}_{\text{CE}}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{k=1}^C q(k) \log \hat{y}_k = (1 - \epsilon)(-\log \hat{y}_y) - \frac{\epsilon}{C} \sum_{k=1}^C \log \hat{y}_k$$
with smoothing factor $\epsilon = 0.1$.

### 6.2 AdamW Optimizer (Decoupled Weight Decay)
Parameters $\Theta$ are updated using decoupled weight decay to prevent $L_2$ regularization from degrading adaptive momentum estimates:
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
$$\Theta_{t+1} = \Theta_t - \eta_t \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \Theta_t \right)$$
with $\beta_1 = 0.9, \beta_2 = 0.999, \lambda = 10^{-4}$.

### 6.3 Cosine Annealing with Warm Restarts
The learning rate $\eta_t$ follows:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left( 1 + \cos\left( \frac{T_{\text{cur}}}{T_i}\pi \right) \right)$$
where $T_0 = 10$ and $T_{\text{mult}} = 2$.
