# DUST: Discrete Unit Sequence Transformer for Ultra-Wideband Non-Line-of-Sight Identification



<img width="1813" height="761" alt="figure1" src="https://github.com/user-attachments/assets/4d4f9515-9dd2-4afd-b6f9-31b2384f076d" />



The overall architecture of the proposed DUST framework, which consists of three modules: 1) Discretization for CIR  signal, where a composite kernel adaptively determines the values of quantization intervals to denoise the CIR amplitude sequence, followed by exponential nonlinear mapping to obtain a discrete index unit sequence; 2) Dual-domain feature fusion, which integrates time-domain and frequency-domain features extracted from the discrete sequence to capture both local and global signal characteristics; and 3) Hybrid Transformer and SVM, where Transformer blocks, composed of transformer encoders, extract the fused features and an SVM classifier produces the final NLOS/LOS identification results.


Dataset:https://log-a-tec.eu/hw-uwb.html

Code:
The code is coming soon
