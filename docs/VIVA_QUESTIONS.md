# Deep Learning Defense & Viva Questions
## Sign Language Recognition and Translation Assistant

---

### Category A: Mathematical & Theoretical Foundations

#### Q1: What is the mathematical formulation of an LSTM cell and what purpose does each gate serve?
**Answer:**
An LSTM cell processes input $\mathbf{x}_t$ and recurrent state $\mathbf{h}_{t-1}$ through three gates and an internal cell state $\mathbf{C}_t$:
1. **Forget Gate $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$**: Decides what fraction of historical memory $\mathbf{C}_{t-1}$ to retain ($0 = \text{discard}, 1 = \text{keep}$).
2. **Input Gate $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$**: Regulates which elements of the candidate state update should enter the cell state.
3. **Candidate Cell State $\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$**: Creates a new candidate vector in $[-1, 1]$.
4. **Cell State $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$**: Linearly integrates memory via the Constant Error Carousel.
5. **Output Gate $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$**: Controls what components of $\mathbf{C}_t$ are emitted as hidden state $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$.

---

#### Q2: How does the LSTM architecture solve the vanishing gradient problem inherent in vanilla RNNs?
**Answer:**
In a standard RNN, $\mathbf{h}_t = \tanh(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t)$. Backpropagation over $k$ steps involves the product of Jacobian matrices $\prod_{j=t-k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$. Because matrix multiplication by $\mathbf{W}^\top$ compounds exponentially, eigenvalues $|\lambda| < 1$ cause the gradient to vanish rapidly.  
In an LSTM, the cell state update is **additive**:
$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t$$
If the network learns to keep $\mathbf{f}_t \approx 1$, the gradient flows back through arbitrary temporal lengths without multiplicative decay. This is known as the **Constant Error Carousel (CEC)**.

---

#### Q3: How does the Temporal Attention mechanism work mathematically, and why is it preferred over pooling or using only the final hidden state $\mathbf{h}_T$?
**Answer:**
Given hidden states $\mathbf{h}_1, \dots, \mathbf{h}_T \in \mathbb{R}^{2H}$:
1. We compute an energy score for each frame: $e_t = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a)$.
2. We normalize scores using Softmax: $\alpha_t = \frac{\exp(e_t)}{\sum_{\tau=1}^T \exp(e_\tau)}$.
3. We compute the context vector: $\mathbf{c} = \sum_{t=1}^T \alpha_t \mathbf{h}_t$.  
Using only $\mathbf{h}_T$ forces the model to compress all information through a temporal bottleneck, suffering from recency bias. Simple average pooling weights irrelevant transition/resting frames equally with key expressive frames. Attention learns to assign high $\alpha_t$ to discriminative inflection frames (e.g., peak hand shape or contact point).

---

#### Q4: What is Label Smoothing, and what is the mathematical intuition for using it here?
**Answer:**
Standard Cross-Entropy uses one-hot targets $y_k \in \{0, 1\}$, encouraging logit $z_y \gg z_k$, pushing weights toward infinity and resulting in overconfident predictions. Label smoothing replaces the one-hot target with:
$$q(k) = (1 - \epsilon)\mathbb{I}(k = y) + \frac{\epsilon}{C}$$
The loss becomes:
$$\mathcal{L} = (1 - \epsilon)(-\log \hat{y}_y) - \frac{\epsilon}{C}\sum_{k=1}^C \log \hat{y}_k$$
In sign language, gestures have kinematic overlap (e.g., identical finger positions with different palm orientations). Label smoothing regularizes logit margins, preventing the model from over-saturating on near-ambiguous gestures.

---

#### Q5: What is the difference between Adam and AdamW, and why is AdamW used?
**Answer:**
In standard $L_2$ regularization with Adam, weight decay is added directly to the gradient before computing moving averages $m_t$ and $v_t$. This causes parameters with large historical gradients to receive smaller weight decay, coupling learning rate adaptation with weight penalty.  
**AdamW (Loshchilov & Hutter)** decouples weight decay by subtracting $\eta_t \lambda \Theta_t$ directly from the weights after the gradient step:
$$\Theta_{t+1} = \Theta_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \eta_t \lambda \Theta_t$$
This ensures pure exponential weight decay regardless of gradient magnitude.

---

### Category B: Data, Preprocessing & Leakage Prevention

#### Q6: How do you prevent data leakage during train/val/test splitting in video gesture recognition?
**Answer:**
We employ a **Signer-Disjoint (Subject-Independent) Split**. If frames or samples from the same person appear in both train and test sets, the model learns person-specific visual shortcuts (e.g., body proportions, arm span, signing velocity) rather than the invariant gesture topology. In our protocol, Signers 0–3 are strictly in Train, Signer 4 in Validation, and Signer 5 in Test. All normalization statistics are computed strictly on the training partition.

---

#### Q7: Why is shoulder-relative spatial normalization necessary?
**Answer:**
Raw MediaPipe coordinates are bound to the camera's image coordinate frame $[0.0, 1.0]$. If a signer stands off-center or at a different distance, coordinates shift and scale. By translating points relative to the mid-shoulder coordinate $\frac{\mathbf{p}_{11} + \mathbf{p}_{12}}{2}$ and dividing by shoulder distance $\|\mathbf{p}_{11} - \mathbf{p}_{12}\|_2$, our representations become **translation-invariant** and **scale-invariant**.

---

#### Q8: How do you handle videos with varying temporal frame lengths?
**Answer:**
Raw gesture clips vary in duration (e.g., 20 to 50 frames). We apply 1D linear temporal interpolation across the temporal axis to resample every sample to a standard sequence length $T = 30$ frames ($\approx 1.0\text{ second}$).

---

### Category C: Architecture, Evaluation & Implementation

#### Q9: Why use 1D-CNN as the baseline rather than a simple Multi-Layer Perceptron (MLP)?
**Answer:**
An MLP flattens the input $(30 \times 201 = 6030\text{ dims})$, discarding temporal sequence order and spatial joint locality while introducing an explosion of parameters. A 1D-CNN with shared convolutional kernels over time represents a rigorous, spatiotemporally aware baseline that learns local multi-frame joint correlations with translation equivariance.

---

#### Q10: How does the real-time application prevent flickering and erratic predictions between sign transitions?
**Answer:**
We implement a **Temporal Debouncer with Consensus Thresholding**:
1. The sliding window produces a prediction at every incoming frame ($30\text{ FPS}$).
2. A prediction is only considered if confidence exceeds $\tau = 0.75$.
3. The debouncer requires $K = 5$ consecutive identical frames before officially committing the gesture to the translation transcript.
4. Once committed, a cooldown period ($15\text{ frames}$) suppresses duplicate triggers during the return-to-rest phase.

---

#### Q11: What evaluation metrics are reported, and why is accuracy alone insufficient?
**Answer:**
In addition to Top-1 and Top-5 accuracy, we report **Macro-Averaged Precision, Recall, and F1-score**, along with a **Normalized Confusion Matrix**. Accuracy can mask severe class-specific dropouts in minority classes. Macro-F1 equally weights all vocabulary classes regardless of class frequency, ensuring balanced discriminability across the entire lexicon.
