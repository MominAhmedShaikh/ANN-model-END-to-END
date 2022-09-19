![This is an image](https://media1.giphy.com/media/wp0PWXANZck7BHr0TF/giphy.gif)
# Sequencial model (EDA, Hyperparameters tuning and much more )

In this Binary Classification, we predict the Status of candidate being selected (1) or not (0) after interview process based on various factors. In order to learn about selection features we will make use of one pf the Deep Learning techniques, the Artificial Neural Networks (ANN).From the millions of interview records we have randomly selected around 10K samples. Moreover, we will use popular Python libraries such as Tensorflow, Kerastuner, imblear and Machine Learning techniques such as Adam Optimizer to train the ANN model and predict the Status.

My goal was to build a full fledge sequencial model from Data preprocessing to model building with hyperparameters tuning and evaluating the model with required metrics.

In future we will add more methods, findings and techniques to build more optimized ANN model.

- Dataset - [Link](https://github.com/MominAhmedShaikh/Artificial-neural-netwrok-ANN-/tree/main/dataset)
- Python Code - [Link](https://github.com/MominAhmedShaikh/Artificial-neural-netwrok-ANN-/blob/main/ANN_model.ipynb)
## Required Installations


```bash
  !pip install -U keras-tuner -q  # for hyperparameter tuning
  !pip install -U imblearn -q     # for handling imbalance data
  !pip install -U scikeras -q     # Gridsearch for optimal epochs, batch size and much more
```
    
## Models made

- Base model - Random Paramters Utilized

- Reduced model - No. of Neurons Reduced

- Regularized model - Added L2 Regularizer

- Dropout model - With Dropout Layers

- Combined model - Combined Base model with L2 Regularizer, Added Dropout Layers.

- Tuner model - Build with kerastuner library



## Model setting

The combination of the Rectifier and Sigmoid activation functions is quite popular and this exact combination will be used in this model as well, Given that the output variable is binary we use cost function Binary Cross Entroopy. Following topics and technical are covered in the paper and in the rest of the files:

Activation function for Hidden Layer: ReLU

Activation function for Output Layer: Sigmoid

Optimization method: Adam Optimizer

Cost function: Binary Cross Entropy

Number of epochs: 
 - Base 100 without EarlyStopping
 - 10 - 20 with EarlyStopping

Batch size: 20

## ETL diagram

<img width="342" alt="BlankMap 3@2x" src="https://user-images.githubusercontent.com/105166921/190937526-2f468295-19f9-4e6b-b29d-8232ae4177c9.png">




## Metrics on imbalanced distribution of values


| Metric / Model| Baseline Model | Reduced Model | L2 Regularized model | Dropout Layers Model | Combined Model |
| --------------| ---------------| ------------  | --------------------| -----------------    |------------- |
| Accuracy      |0.69            |0.70           | 0.72                 | 0.67                 | 0.71           |
|Precision_score|0.68            |0.69           | 0.73                 | 0.67                 | 0.72           |
|Recall_score   |0.71            |0.71           | 0.69                 | 0.69                 | 0.69           |




## Notebook link
It is highly recommended to open in notebook in google colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MominAhmedShaikh/Artificial-neural-netwrok-ANN-/blob/main/ANN_model.ipynb)

## 🔗 Connect with me
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/momin-ahmed-shaikh/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/iammomin139)


## Authors

- [@MominAhmedShaikh](https://github.com/MominAhmedShaikh)


## Feedback

If you have any feedback, please reach out to us at mominahmedshaikh@gmail.com
