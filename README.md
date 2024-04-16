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
python meta_train.py
```
This will train the model using the meta-training dataset.

## Meta-test
To evaluate the meta-learning model on the meta-testing dataset, run the following command:
```shell
python meta_test.py
```

This will evaluate the model's performance on the meta-testing dataset and output the results.

## Inference

To perform inference on a new echocardiogram image, run the following command:
```shell
python inference.py --image_path <path_to_image>
```

Replace `<path_to_image>` with the path to the echocardiogram image you wish to analyze. The script will output the left ventricular chamber quantification results.

[model download](https://drive.google.com/drive/folders/1xXmYt1wmqtiqmpLlJc3sjbWxK7ogRP5v?usp=share_link)

## Sample Images

Some sample echocardiogram images are included in the `sample_images` directory. You can use these images to test the inference script or to visualize the model's predictions.

## Dataset

To access the dataset, please visit the following URL: [data download](https://drive.google.com/file/d/1MuIyXgZOkx1WwM0rT5qn4JeA8gbKMRDl/view?usp=sharing). Please note that you will need to request download permissions to access the data. 
Once your request is approved, you will be able to download the datset.

## Citation

Feel free to modify and use this code for your own research. If you use this code, please cite the original paper.

## Contact

Please let me know if you have any questions or need further assistance.


