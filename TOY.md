# Toy GRPO Influence Sandbox

The toy example is a minimal, end-to-end implementation of Group Relative Policy Optimization (GRPO) designed to be mathematically transparent. It uses a tiny autoregressive model to demonstrate how specific training examples (helpful vs. harmful) affect the model's performance on a test prompt.

## 1. Model Architecture
**Type:** Autoregressive Logistic Regression
**Parameters:** $\approx 22$ trainable parameters.

The model generates a sequence of **2 tokens**. Each token is binary (0 or 1).
- **First Token:** A linear layer takes a 3D input vector $z$ and produces logits for the first token.
- **Second Token:** A second linear layer takes the concatenation of $z$ and the value of the *first* token to produce logits for the second token.

This structure allows the model to learn conditional dependencies (e.g., "if the first token is 1, the second should be 0").

## 2. The Dataset ("The Sandbox")
The dataset is hardcoded in `build_user_plan_sandbox` to create a clear "influence story":

| Name | Input $z$ | Target Sequence | Role / Expected Influence |
| :--- | :--- | :--- | :--- |
| **helpful_feature_a** | `(1, 0, 0)` | `(1, 0)` | **Helpful:** Shares Feature A with the test set. |
| **harmful_shared_noise**| `(0, 1, 1)` | `(0, 1)` | **Harmful:** Shares "Noise" (Feature C) but has a conflicting target. |
| **neutral_feature_b** | `(0, 1, 0)` | `(0, 1)` | **Neutral:** Irrelevant to the test set. |
| **TEST EXAMPLE** | `(1, 0, 1)` | `(1, 0)` | The evaluation target. |

**The Logic:**
- The test example has Feature A (`z[0]=1`) and Feature C (`z[2]=1`).
- `helpful_feature_a` teaches the model that Feature A implies target `(1, 0)`.
- `harmful_shared_noise` teaches the model that Feature C implies target `(0, 1)`.
- Since the test set target is `(1, 0)`, the "noise" example actively pulls the model in the wrong direction.

## 3. Training: Toy GRPO
The model is trained using a simplified version of the repo's GRPO logic:
1.  **Rollout:** For a prompt $z$, the model generates all 4 possible sequences `(0,0), (0,1), (1,0), (1,1)`.
2.  **Reward:** A sequence gets a reward of `1.0` if it matches the target exactly, `0.0` otherwise.
3.  **Advantage:** The reward is mean-centered across the 4 sequences to calculate the GRPO advantage.
4.  **Update:** The model is updated using the standard GRPO policy gradient with clipping ($\epsilon$) and optional KL divergence.

## 4. Influence Calculation
The sandbox computes how each training step affected the model's probability of success on the Test Example.

### Geometry Features ($X$)
The Fisher Information is calculated using the **Policy Score**:
$$x = \nabla_\theta \log \pi_\theta(\text{rollouts})$$
This captures the curvature of the policy manifold—how the distribution of generated sequences changes as parameters move.

### Historical Trajectory
Because RL training is non-stationary, the sandbox tracks the model's state at every step. It offers two modes for the Fisher Matrix ($F$):
- **`active_only`:** $F$ only includes the curvature of the specific example being trained at each step.
- **`all_samples`:** $F$ is the average curvature of the entire dataset, evaluated at every point along the training path.

### Solvers
- **Full:** Inverts the $D \times D$ matrix (where $D$ is the parameter count).
- **Woodbury:** Uses the Woodbury Matrix Identity to invert an $n \times n$ matrix (where $n$ is the number of samples), which is the primary "trick" used in the main repo for scaling to large models.

## 5. Success Metrics
The sandbox validates that the Influence Function is working by comparing:
1.  **Predicted Influence:** The score calculated using the Fisher Inverse.
2.  **Actual Influence:** The script performing a "counterfactual" update (stepping the model with a specific training example) and measuring the real change in test reward.

When the Fisher matrix is accurate, the **Predicted** and **Actual** lines on the generated plots should closely align.
