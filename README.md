# UFDA-Universal-Federated-Domain-Adaptation-with-Practical-Assumptions--accepted by AAAI 2024

This repository provides the implementation for our paper: [UFDA: Universal Federated Domain Adaptation with Practical Assumptions](https://ojs.aaai.org/index.php/AAAI/article/view/29311)

## Model Review:
![framework](resources/Model.png)

## Setup

**Install Package Dependencies**

We need users to declare a base path to store the dataset as well as the log of the training procedure. The directory structure should be
```
base_path
│       
└───data
│   │   Office-31
│       │   amazon
│       │   dslr
|       |   webcam
│   │   OfficeHome
│       │   ...
│   │   VisDA2017+ImageCLEF-DA
```
Our framework now supports four multi-source domain adaptation datasets: ```Office-Home, Office-31, and VisDA2017+ImageCLEF-DA```.

**Training**  
We provide the config files with the format `.yaml`. To perform the FUFDA: Universal Federated Domain Adaptation with Practical Assumptions on the specific dataset (e.g., Office-31), please use the following commands:
 
```python
python main_new.py --config train-config-office311.yaml --dist-url 'tcp://localhost:13110' --loss_weight 0.01 --loss_penalty 0.00 --prot_start 5
```

**Citation**

If you use this code, please cite:
```python
@inproceedings{liu2024ufda,
  title={UFDA: Universal Federated Domain Adaptation with Practical Assumptions},
  author={Liu, Xinhui and Chen, Zhenghao and Zhou, Luping and Xu, Dong and Xi, Wei and Bai, Gairui and Zhao, Yihan and Zhao, Jizhong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={12},
  pages={14026--14034},
  year={2024}
}
```

