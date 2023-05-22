# Image Captioning using Few-Shot GPT

This repository contains code for generating text paragraph for the Flickr8k images using a few-shot GPT model. The model is fine-tuned on the image and caption data from the Flickr8k dataset to generate more captions for the images. The generated captions are used to generate a text paragraph.

## Problem Statement

The problem is to generate captions for the images in the Flickr8k dataset and use them to create a text paragraph. The dataset can be downloaded from [this link](https://www.kaggle.com/datasets/adityajn105/flickr8k).

## How to Run

Follow the steps below to run the code:

1. Download the dataset from [this link](https://www.kaggle.com/datasets/adityajn105/flickr8k) and extract it.

2. Rename the extracted folder to "Data".

3. Install the dependencies by running the following command:

conda create --name new_environment_name --file requirements.txt

4. To train the model from scratch, run `train.py` to generate the weights (model). Alternatively, you can download the pretrained weights from [this Google Drive link](https://drive.google.com/drive/folders/1_VZB0SDfqN6c0QAUnG3Y3RoENAtF8Pii?usp=share_link).

5. Create your ChatGPT API key by visiting [this link](https://platform.openai.com/account/api-keys). Save and paste the API key in a file named `api_key.txt`.

6. For inference, i.e., to generate a paragraph given an image, run `infer.py`.

7. The sample output can be seen in `output.txt`.

Feel free to explore and modify the code to suit your needs!
