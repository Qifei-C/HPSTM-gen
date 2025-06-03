# HPSTM-gen

**HPSTM-Gen** extends the Human Pose Smoothing with Transformer and Manifold Model (HPSTM) into a *generative* model for human pose trajectories.
Unlike traditional discriminative models that only denoise or smooth existing pose sequences, HPSTM-Gen learns the full distribution of physically plausible trajectories. This enables the model to sample new, diverse, and anatomically valid human motions conditioned on visual, language, or context cues.

Key features:

* Diffusion/Flow-matching based generative training
* Transformer backbone with global temporal attention
* Explicit kinematic constraints via differentiable forward kinematics (FK)
* Optional multi-modal context (vision/language/robot state)
* Physically plausible sampling and uncertainty estimation

## Method

### 1. **Model Architecture**

HPSTM-Gen retains the core architecture of the original HPSTM:

* **Input:** A sequence of (possibly noisy) 3D joint positions, optionally with context (e.g., language/vision embeddings)
* **Backbone:** Encoder-decoder Transformer, capturing long-range temporal dependencies
* **Output:**

  * Refined/sampled 3D pose sequences
  * (Optional) Per-joint covariance estimates
  * All outputs are mapped via a differentiable forward-kinematics layer, enforcing manifold (bone-length) constraints

### 2. **Generative Training with Flow Matching**

Instead of only learning to denoise trajectories, HPSTM-Gen is trained to *model the full data distribution* using flow matching (or optionally score-based diffusion).

#### **A. Noise Injection (Data Corruption)**

For each training sample:

* Sample a noise level $\tau \in [0, 1]$
* Sample Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$
* Mix the clean pose sequence $\mathbf{X}_\text{data}$ with noise:

  $\mathbf{X}_\tau = \tau \cdot \mathbf{X}_\text{data} + (1 - \tau) \cdot \epsilon$
  
* The model receives $\mathbf{X}_\tau$ (and optional context) as input

#### **B. Flow Matching Loss**

The model learns to predict the "denoising direction"—the vector field that maps noisy samples back to the data manifold.

$\mathcal{L}_\text{FM} = \mathbb{E}_{\mathbf{X}_\text{data}, \tau, \epsilon}\left[ \left\| f_\theta(\mathbf{X}_\tau, \text{context}) - (\epsilon - \mathbf{X}_\text{data}) \right\|^2 \right]$

* $f_\theta$: HPSTM-Gen's prediction of the denoising vector, given the noisy input
* The output is passed through the FK layer to enforce anatomical plausibility

#### **C. Auxiliary Losses (Optional but recommended)**

* **Bone Length Consistency**: Penalize deviation from canonical bone lengths
* **Smoothness**: Temporal velocity and acceleration losses for physically realistic motion
* **Negative Log-Likelihood (NLL)**: If predicting covariance, add NLL of predicted distributions

#### **D. Full Training Objective**

$\mathcal{L} = \mathcal{L}_\text{FM} + \lambda_\text{bone} \mathcal{L}_\text{bone} + \lambda_\text{vel} \mathcal{L}_\text{vel} + \lambda_\text{accel} \mathcal{L}_\text{accel} + \lambda_\text{NLL} \mathcal{L}_\text{NLL}$

All terms can be weighted based on task priorities.

### 3. **Sampling (Inference) Procedure**

To generate a new pose sequence:

1. **Initialize** with a pure Gaussian noise sequence: $\mathbf{X}_0 \sim \mathcal{N}(0, I)$
2. **Iteratively Denoise**: For a chosen number of steps, repeatedly input the current sequence into HPSTM-Gen (optionally with context), and update using the predicted denoising vector:

   $\mathbf{X}_{k+1} = \mathbf{X}_k + \alpha_k f_\theta(\mathbf{X}_k, \text{context})$

   where $\alpha_k$ is the step size at iteration $k$
3. **Manifold Projection**: After each step, pass the output through the FK layer to ensure bone-length and anatomical constraints.
4. **Obtain Final Output**: The last sequence is a physically plausible, context-conditioned motion sampled from the learned distribution.

This process is analogous to diffusion/score-based generative models in vision, but tailored to structured, constrained motion data.

### 4. **Implementation Notes**

* You can leverage existing diffusion model frameworks (e.g., DiffusionPolicy, π0, Score-based Models) and plug in the HPSTM backbone.
* The FK layer and bone-length losses are essential for anatomical validity—do not skip them!
* Sampling step sizes, number of iterations, and noise schedule can be tuned based on downstream requirements.

## Getting Started

1. **Requirements**

   * PyTorch >= 1.10
   * numpy, tqdm, etc.
   * (Optional) Visualization toolkit (e.g., matplotlib, open3d)

2. **Dataset**

   * Human pose sequences (e.g., AMASS, 3DPW, or your own motion capture)
   * (Optional) Context data: language instructions, video/image embeddings, etc.

3. **Training**

   * Use `train.py` (example to be provided)
   * Specify model config (Transformer depth/width, FK parameters, loss weights)
   * Train with both clean and noise-corrupted data as above

4. **Sampling**

   * Use `sample.py` to generate new motions given a context
   * Visualize and/or retarget to a robot arm via your pose-mapping module


## Acknowledgments

This work extends the HPSTM model ([link to your repo](https://github.com/Qifei-C/HPSTM))
Inspired by π0, Diffusion Policy, and recent advances in generative robotics.

## Contact

Questions or issues?
Open an issue or email [qifei@seas.upenn.edu](mailto:qifei@seas.upenn.edu)
