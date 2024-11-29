# **Understanding Neural Tangent Kernel: Key Theories and Experimental Insights**  
**Authored by Samy VILHES**  

📄 **Published Paper**: [Access on HAL](https://normandie-univ.hal.science/hal-04784111)  

---

This repository accompanies the published paper and provides the code and experimental setups that delve into the **Neural Tangent Kernel (NTK)**, highlighting both its theoretical foundations and practical applications. The experiments and analysis focus on understanding key properties of NTK during training, as well as its influence on loss behavior.

## **Datasets**  
The following datasets are utilized in this study:  
- **MNIST Dataset**: For classification tasks.  
- **Boston Housing Dataset**: For regression tasks.  

## **Repository Structure**  

- **`constant/`**: This folder contains code and experiments that explore the **constant nature of the NTK** during training.  
- **`loss_bound/`**: This folder investigates how the **loss is bounded by a quantity depending of the eigenvalues of the NTK**, providing further theoretical insights.