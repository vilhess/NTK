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

## **Setup Instructions**  
To set up the repository, clone it along with its submodules using the following commands:
1) Clone the repository
2) Install requirements
```bash
pip install -r requirements.txt
```

3) Clone the XAutoDL submodule

```
git clone --recurse-submodules https://github.com/D-X-Y/AutoDL-Projects.git XAutoDL
mv XAutoDL/xautodl/ ./
rm -rf XAutoDL
```

4) Download datasets
```bash
wget -P data/cifar/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf data/cifar/cifar-10-python.tar.gz -C data/cifar/
rm data/cifar/cifar-10-python.tar.gz
```