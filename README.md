# Metalearning-echocardiogram

## Overview

This repository contains the code for the paper "Few-shot Quantification of Left Ventricular Chamber in the Standard Echocardiogram via Model-agnostic Meta Learning."

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

## Citation

Feel free to modify and use this code for your own research. If you use this code, please cite the original paper.

## Contact

Please let me know if you have any questions or need further assistance.


