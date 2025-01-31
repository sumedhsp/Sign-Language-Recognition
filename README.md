
# 🧑‍🎓 Computer Vision - Final Project (CS-GY 6643)

## 🔍 Sign Language Recognition Using I3D and Transformers

This project focuses on **word-level sign language recognition** using deep learning techniques, combining **Inflated 3D ConvNets (I3D)** for feature extraction and **Transformers** for temporal modeling. We leverage the **WLASL dataset**, one of the largest publicly available ASL datasets, to train and evaluate our model. 

---

## 📌 Project Overview

Sign languages are complex visual languages used by the Deaf and hard-of-hearing communities. Our goal is to develop an **AI-driven system** to **automatically recognize sign language gestures from video data**.

### 🔹 **Motivation**
- Traditional **Hidden Markov Models (HMMs) and Dynamic Time Warping (DTW)** struggle with complex sign language patterns.
- **I3D models** excel at capturing **spatiotemporal** relationships in videos.
- **Transformers** are powerful at modeling **long-range dependencies** in sequential data.
- Our **hybrid approach** combines **I3D and Transformers** to improve sign language recognition accuracy.

---

## 📊 Dataset: **WLASL**
We use the **Word-Level American Sign Language (WLASL) dataset**, which contains:
- 🏷 **2000 ASL words**  
- 🎥 **21,000 videos**  
- 🎭 **100+ signers**  
- 📂 **WLASL subsets:** `WLASL100`, `WLASL300`, `WLASL1000`, `WLASL2000`

📜 **Dataset Citation**:  
- Dongxu et al. (2020) - [WLASL Dataset](https://github.com/dxli94/WLASL)

---

## 🏗 Model Architecture

### **🔹 Feature Extraction: I3D Model**
- **Pretrained I3D model** extracts spatiotemporal features from video sequences.
- **3D convolutions** allow it to model motion patterns in signing gestures.
- Output: **1024-dimensional feature vector** per video.

### **🔹 Temporal Modeling: Transformer**
- A **6-layer Transformer encoder** captures sequential dependencies across frames.
- **Multi-head Self-Attention** helps the model focus on important frames.
- **Final classifier** predicts the most likely ASL word from 2000 possible classes.

![image](https://github.com/user-attachments/assets/7d95dd44-0d27-4274-a08b-5b9ce9774d0b)


### **🔹 Alternative Approach: Vision Transformer (ViT)**
- We experimented with **Vision Transformers (ViT)** for feature extraction.
- **Challenges:** Requires **high computational power** and **larger datasets**.
- Results showed **overfitting on smaller subsets** due to dataset limitations.

---

## 📈 Results & Evaluation

### 🔹 **Evaluation Metrics**
- ✅ **Accuracy (Top-1, Top-5, Top-10)**
- 🎯 **Precision, Recall, F1-Score**
- 🔍 **Confusion Matrix**
- ⚖ **Balanced Accuracy** (to handle class imbalance)

### 🔹 **Performance Comparison**
| Model | WLASL100 (Top-1) | WLASL300 (Top-1) | WLASL1000 (Top-1) | WLASL2000 (Top-1) |
|-------|------------------|------------------|-------------------|------------------|
| **Baseline I3D** | 65.89% | 56.14% | 47.33% | 32.48% |
| **Proposed I3D + Transformer** | **74.47%** | **57.86%** | **45.13%** | **34.66%** |

✅ **Key Findings**:
- Our **hybrid model outperforms the baseline I3D** on smaller datasets.
- **Class imbalance issues** affect performance on larger datasets (`WLASL2000`).
- **Future improvements**: Weighted sampling, additional fine-tuning.

---

## 🔧 Improvements Implemented
| Enhancement | Description |
|-------------|-------------|
| **Scaling to Full WLASL2000 Dataset** | Trained on the **entire dataset** to improve generalization. |
| **Vision Transformer (ViT) Integration** | Tested ViT for spatial feature extraction but faced compute limitations. |
| **Class Imbalance Handling** | Used **Weighted Random Sampler** and **Weighted Cross-Entropy Loss** to balance underrepresented classes. |

---

## Project Reproducibility 

### Prerequisites
1. **Python Environment**:
   - Ensure you have Python 3.x installed.
   - Install the required dependencies using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

2. **Download I3D Weights**:
   - Download the appropriate I3D model weights from [this link](https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view?usp=sharing).
   - Place the downloaded weights file (.pt) in the `code/` directory.

3. **Download Dataset**:
   - Download the dataset from [this link](https://drive.google.com/file/d/1UKESPYEvFsrrQByrl9mWny_JJdoMXKfk/view?usp=drive_link). This dataset was obtained directly from the author of the [original paper](https://github.com/dxli94/WLASL) so please use this for educational purposes only. To download the data for running the project on your own, please follow the original author's github repo for more instructions.
   - Create a folder named `data` inside the `code/` directory and paste the dataset into it:
     ```
     code/
     ├── data/
     │   ├── [dataset files here]


### Running the Project
#### Training
- To train the model, use one of the training scripts. For example:
  `python train_vanilla_v1_2000.py`

#### Testing
- To evaluate the model, use one of the testing scripts. For example:
  `python test_vanilla_v1_2000.py`


### Notes

- Ensure the preprocess/ and configfiles/ folders contain necessary configuration files for the model and preprocessing pipeline.

- If you encounter any issues, verify that all dependencies are installed and that the dataset and weights are correctly placed.

---

## 🤝 Acknowledgments

🙏 **Baseline Repository:**  
This project was built upon the excellent work by **Dongxu et al.** (2020)  
📌 **GitHub Repo:** [WLASL Baseline](https://github.com/dxli94/WLASL)


👨‍💻 **Contributors:**
- **Akshat Shaha** (`as16655@nyu.edu`) - (https://github.com/akshatshaha)
- **Ashutosh Kumar** (`ak10514@nyu.edu`) - (https://github.com/ashutoshkumar03)
- **Pranav Mohril** (`pm3727@nyu.edu`) - (https://github.com/TechQuazar)
- **Sumedh Parvatikar** (`sp7479@nyu.edu`) - (https://github.com/sumedhsp)


We completed the project under the guidance of Prof. Erdem Varol.

---

🚀 **Future Work**
- Fine-tuning the **Transformer model** with **larger batch sizes** and **longer training durations**.
- **Hyperparameter tuning** to further enhance generalization.
- **Exploring multi-modal learning** with **pose-based inputs**.


💡 **If you're interested in sign language recognition, feel free to explore and contribute!** 🌟

For more information, kindly refer the project report.
