# **Understanding Neural Tangent Kernel: Key Theories and Experimental Insights**  

📄 **Published Paper**: [Access on HAL](https://normandie-univ.hal.science/hal-04784111)  

---

This repository accompanies the published paper and provides the complete **codebase and experimental setup** used to explore the **Neural Tangent Kernel (NTK)**.  
It highlights both the **theoretical foundations** of NTK and its **practical implications**, focusing on its behavior during training and its influence on the loss dynamics.

---

## 🧠 **Overview**

The experiments and analyses presented here investigate several key properties of the NTK, including:
- Its constancy during training across architectures.
- The relationship between NTK eigenvalues and loss bounds.
- The deterministic behavior of NTK in wide neural networks.
- Applications of NTK to **Neural Architecture Search (NAS)**.

---

## 📚 **Datasets**

The following datasets are used in this study:

- **MNIST** – for classification experiments.  
- **Boston Housing** – for regression experiments.  
- **NAS-Bench-201** – for architecture search experiments.

---

## 📁 **Repository Structure**

| Folder | Description |
|--------|-------------|
| `constant/` | Explores the **constant nature of the NTK** during training. |
| `loss_bound/` | Investigates how the **loss is bounded** by quantities related to the NTK’s eigenvalues. |
| `deterministic/` | Verifies the **deterministic behavior** of NTKs in wide neural networks. |
| `neural_archi_search/` | Contains experiments related to **Neural Architecture Search** using NTK metrics. |

---

## ⚙️ **Setup Instructions**

Follow these steps to set up the environment and reproduce the experiments:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vilhess/NTK.git
cd NTK
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Clone the XAutoDL Submodule

```bash
cd neural_archi_search
git clone --recurse-submodules https://github.com/D-X-Y/AutoDL-Projects.git XAutoDL
mv XAutoDL/xautodl/ ./
rm -rf XAutoDL
cd ..
```

### 4️⃣ Download the Datasets

#### CIFAR-10 (for NAS experiments)

```bash
wget -P data/cifar/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf data/cifar/cifar-10-python.tar.gz -C data/cifar/
rm data/cifar/cifar-10-python.tar.gz
```

#### NAS-Bench-201

1. Download the NAS-Bench-201 file from [Google Drive](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view)  
2. Place the downloaded file in:  
   ```bash
   neural_archi_search/nb201_api/
   ```

---

