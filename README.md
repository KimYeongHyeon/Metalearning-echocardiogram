# Metalearning-echocardiogram

## Overview

This repository contains the code for the paper "Quantification of Left Ventricular Mass in Multiple Views of Echocardiograms using Model-agnostic Meta Learning in a Few-Shot Setting".
![Figure1](https://github.com/KimYeongHyeon/Metalearning-echocardiogram/blob/main/Figure%201.png?raw=true)

## Environments
We tested with:
- torch==1.12.0
- torchvision==0.13.1
- learn2learn==0.1.7
- segmentation-models-pytorch==0.3.0
- timm==0.4.12

## Meta-train

To train the meta-learning model, run the following command:
```shell
python train.py data.shot={5,10,20,30} data.target={2CH, 4CH, PLAX, PSAX} model.algorithm={MAML, MetaCurvature, MetaSGD} model.network={unet, DeepLabV3Plus}
```

This will train the model using the meta-training dataset.
The model is then automatically created in the following folder
```shell
outputs-MAML_FO_True_AN_True-{data.target}-{data.shot}-lightning_logs-{version_0,1,2,...}
```

## Meta-test
To evaluate the meta-learning model on the meta-testing dataset, run the following command:
```shell
python test.py --shot={5, 10, 20, 30} --target={2CH, 4CH, PLAX, PSAX} --version={0,1,2,..} --algorithm={MAML, MetaCurvature, MetaSGD} --network={unet, DeepLabV3Plus}
```

This will evaluate the model's performance on the meta-testing dataset and output the results.

## Inference

To perform inference on a new echocardiogram image, run the following command:
```shell
python inference.py --target={2CH, 4CH, PLAX, PSAX} --version={0,1,2,..} --network={unet, DeepLabV3Plus} --algorithm={MAML, MetaCurvature, MetaSGD}
```

[model download](https://drive.google.com/drive/folders/1xXmYt1wmqtiqmpLlJc3sjbWxK7ogRP5v?usp=share_link)

## Dataset

This study used three publicly available echocardiography datasets, each processed with additional labeling for our experiments.  
**The results in this repository can only be reproduced using the labeled versions** provided through our controlled-access link, not the raw datasets alone.

### 1. EchoNet-LVH (PLAX view)
- **Source:** Stanford University, EchoNet Project  
- **Original URL:** [https://echonet.github.io/dataset/echonet-lvh](https://echonet.github.io/dataset/echonet-lvh)  
- **License/Access:** Publicly available for research use upon agreement to the dataset license.  
- **Modifications in this study:** Added custom labels specific to our task.  
- **Access to labeled version:** Please request via the Google Drive link below.

### 2. TMED-2 (PSAX view)
- **Source:** Tufts Medical Center  
- **Original URL:** [https://tmed.cs.tufts.edu/echonet/](https://tmed.cs.tufts.edu/echonet/) *(replace with actual)*  
- **License/Access:** Access requires a request to the data owner.  
- **Modifications in this study:** Added custom labels specific to our task.  
- **Access to labeled version:** Please request via the Google Drive link below.

### 3. CAMUS (A2C and A4C views, 1-, 5-, and 10-shot scenarios)
- **Source:** University Hospital of St Etienne, France  
- **Original URL:** [https://www.creatis.insa-lyon.fr/Challenge/camus](https://www.creatis.insa-lyon.fr/Challenge/camus)  
- **License/Access:** Free for research use after registration and acceptance of terms.  
- **Modifications in this study:** Added custom labels specific to our task.  
- **Access to labeled version:** Please request via the Google Drive link below.

---

### Access to Labeled Versions
The labeled datasets used in this study are derived from the above publicly available datasets, with additional annotations created by our team.

To request access to the labeled datasets:

1. Visit the following link:  
   [Request Access to Labeled Datasets](https://drive.google.com/file/d/1MuIyXgZOkx1WwM0rT5qn4JeA8gbKMRDl/view?usp=sharing)
2. Click **"Request Access"**.
3. Provide your **name, affiliation, intended use, and confirmation that you have legitimate access to the original datasets**.
4. Your request will be reviewed, and access will be granted to qualified researchers.
5. Upon approval, you will receive download access to the labeled dataset package.

---

### Important Notes
- This repository **does not redistribute the original datasets** without modification.
- Access will only be granted to researchers who confirm they already have (or are eligible to obtain) access to the original datasets.
- Use of the labeled datasets is subject to the terms of both the **original dataset licenses** and our **custom labeling license agreement**.

---

### Citation
If you use the labeled datasets from this repository, please cite both the original dataset sources and our work.

## Citation

Feel free to modify and use this code for your own research. If you use this code, please cite the original paper.

## Contact

Please let me know if you have any questions or need further assistance.


