# Medical-VQA

## 1. Dataset
We used Path VQA Dataset from Hugging Face which consist of pathology images, such as histopathology slides, accompanied by questions pertaining to tissue analysis, disease identification, etc. and their relevant answers.
The dataset contains a total of 5,004 images and 32,795 question-answer pairs, with no fixed number of questions per image. The dataset has different kinds of questions that aren’t limited to just Yes or No questions and images are each different sized in the dataset.
## 2. Dataset Preprocessing
### 1. Imports:
- Imported necessary libraries for data handling, image processing, numerical operations, plotting, and deep learning using PyTorch and Hugging Face.
### 2. Dataset Loading:
- Loaded the "path-vqa" dataset from the Hugging Face Hub using the `load_dataset` function.
### 3. Answer Mapping:
- Created mappings (`id_to_ans` and `ans_to_id`) to associate each unique answer with a unique index. This helps in converting answers into numerical labels for the classification task.
### 4. Custom Dataset Class:
- Defined a `QADataset` class inheriting from `torch.utils.data.Dataset`:
- Initialization: Stored the dataset and processor.
- Length Method: Returned the number of samples in the dataset.
- Get Item Method: Processed each sample by resizing images, encoding questions, and converting answers to one-hot encoded vectors.
### 5. Processor Initialization:
- Initialized a processor (`ViltProcessor`) from Hugging Face's transformers library to handle the encoding of images and text.
### 6. Dataset Instances:
- Created instances of `QADataset` for both the training and test datasets, using the processor to handle image and text processing.
### 7. Custom Collate Function:
- Defined a custom `collate_fn` function to batch process the dataset items, handling the padding and stacking of images and text.
### 8. DataLoader Creation:
- Created `DataLoader` instances for training and test datasets, utilizing the custom collate function to facilitate batching, shuffling, and efficient data loading during model training and evaluation.

## 3. Model Architecture
The model architecture is a variant of the ViLT (Vision and Language Transformer) model adapted for question answering tasks.
A. ViltEmbeddings:
1. text_embeddings:
This submodule deals with embedding textual inputs. It consists of:
1.
word_embeddings: Embedding layer for converting word tokens into fixed-size vectors.
2.
position_embeddings: Embedding layer for representing the positional encoding of words.
3.
token_type_embeddings: Embedding layer for token type information, such as segment IDs in tasks like question answering.
4.
LayerNorm: Layer normalization to normalize embeddings.
5.
dropout: Dropout layer for regularization.
2. patch_embeddings:
This submodule deals with embedding image patches. It consists of:
1.
projection: Convolutional layer for projecting image patches into the same embedding dimension as text embeddings.
2.
token_type_embeddings: Embedding layer for token type information.
B. ViltEncoder:
This module consists of a stack of ViltLayers. Each ViltLayer includes:
1. ViltAttention: Multi-head self-attention mechanism.
1.
ViltSelfAttention: Self-attention layer.
2.
query, key, value: Linear layers for projecting input into query, key, and value spaces.
3.
dropout: Dropout layer for regularization.
4.
output: Layer for combining attention scores with input embeddings.
5.
dense: Linear layer for output transformation.
6.
dropout: Dropout layer for regularization.
2. ViltIntermediate: Intermediate feedforward layer.
1.
dense: Linear layer for feedforward transformation.
2.
intermediate_act_fn: Activation function, often GELU.
3. ViltOutput: Output layer of the ViltLayer.
1.
dense: Linear layer for output transformation.
2.
dropout: Dropout layer for regularization.
4. layernorm_before, layernorm_after:
Layer normalization before and after each sublayer.
C. ViltPooler:
Pooling layer for generating a fixed-size representation of the sequence.
1.
dense: Linear layer for output transformation.
2.
activation: Activation function, often Tanh.
D. Classifier:
Sequential module consisting of linear layers and normalization layers.
1.
Linear: Linear layer for mapping features to a higher-dimensional space.
2.
LayerNorm: Layer normalization.
3.
GELU: Activation function, often Gaussian Error Linear Unit.
4.
Linear: Final linear layer for classification, outputting scores for different classes.
This architecture is designed to process both textual inputs (such as questions) and image patches, integrating information from both modalities to perform tasks like question answering. The ViltEmbeddings module handles the input embeddings from the processor, the ViltEncoder processes the embeddings through a stack of transformer layers, and the ViltPooler generates a fixed-size representation of the sequence. Finally, the Classifier module provides the classification scores.
## 4. Training:
A Pretrained weights of VILT were applied to it which had been pretrained on the VQAv2 dataset the has 265,016 images different images from datasets like COCO and At least 3 questions (5.4 questions on average) per image with 10 ground truth answers per question
The model’s last layer of classification was replaced with a custom layer of size 4101 neuron which represents the number of possible answers in the dataset.
The Model was trained on 3 epochs on the training data, Adam with a learning rate of 5e-5 was used as the optimizer for the model, finally binary cross entropy is applied by the transformer as it is a multiclassification model, The model achieved 32% accuracy on testing set while it had a validation of 4.76 and training loss of 4.84.
Fig 2: shows the accuracy of the model on the training set
Fig 3: shows the losses of the model on the training and validation for all the three epochs.
Fig 4: Graph of losses on the training and validation for all the three epochs.
4. Model Inference:
Fig 6: Inference on sample data from the test set.
Fig 7: Inference on a sample data from the test set.
## 5. Web Application:
The web application’s frontend is built on top of React ( a framework of JavaScript ) paired with backend in Flask, Python.
Website:
Fig 8:Website Page.
The interface is simple with a file upload and question input fields. Upon the upload of the file, the API request is sent to the backend, which runs inference function and returns the predicted output. The image and the predicted answers is then output on the screen.
Fig 9: Prediction Result.
Fig 10: API Request.
Fig 11: API Response.
## 6. Conclusion and Improvements
The Model failed to achieve the accuracies we were hoping to achieve as such it isn’t recommend trying this model for the dataset, The accuracy achieved was not at all satisfactory, moreover a GPT2 model was tried which failed to have any improvements on the accuracy but rather and had even worse performance on the dataset.
In the future we can look to improve the model by doing the following things:
1.
Increasing the size of our dataset
2.
Training a model that has already been pretrained on a more similar dataset.
