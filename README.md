# COMP6721 Summer2024 course project 
Image classification on five venues with regarding of three different models 

## Contributors
| Group    | Student Name | Student ID |
|----------|--------------|------------|
| CloseAI  | Yang Cao     | 26654029   |
|          | Hongwu Li    | 40280054   |
|          | Bo Shi       | 40292839   |

## Dataset

In terms of dataset, we use a subset of the [MIT Places365 dataset](https://www.kaggle.com/datasets/mittalshubham/images256/data) which contains 5150 pictures of 5 classes out of 365 classes from the original dataset. (For the original MIT Places365 dataset, see: [The Original Dataset](https://www.kaggle.com/datasets/mittalshubham/images256/data)) This dataset consists of 365 classes of different scene-centric pictures that all contain representative features of various locations. All the images are JPEG color images and resized to 256x256 resolution. With consideration of robustness and fitness, the five classes chosen from the original dataset are general, representative locations with specific features. The five classes are:

- **homeOffice** (1346/5150)
- **hotelRoom** (1524/5150)
- **church** (797/5150)
- **museum** (760/5150)
- **supermarket** (1466/5150)

For better performance from the classifier, every sampled image is taken from real life, including people or all possible surroundings, so that the model could possibly be more robust. In our dataset, the quantities of all five classes are balanced, with each class containing 700 images, and are divided into training and testing images in an 8:2 ratio. Our dataset is finally presented as follows:

- **homeOffice** (700/3500)
- **hotelRoom** (700/3500)
- **church** (700/3500)
- **museum** (700/3500)
- **supermarket** (700/3500)

## Methodology

We will use supervised decision tree, semi-supervised decision tree, and supervised CNN as our methods to solve the problem. Common steps for all methods are data loading and data preprocessing.

### Data Loading:

- Load the image dataset and associated labels.
- Split the dataset into training, testing, and validation image sets.

### Data Preprocessing:

- Resize images to a consistent size.
- Normalization of images and encoding labels into numerical format.

### Model 1: Supervised Decision Tree

This method is used for both classification and regression tasks [2], representing decisions and their possible outcomes in a tree structure. It operates by initially extracting features from images, such as color histograms, textures, or embeddings from pre-trained CNNs. The algorithm is then trained using these features and their corresponding labels to formulate decision rules. When making predictions, new images are classified by navigating through the tree based on their features.

### Model 2: Semi-supervised Decision Tree

Semi-supervised learning utilizes both labeled and unlabeled data for training, enhancing the performance of models like Decision Trees since we will need to train multiple Decision Trees iteratively taking into account the labeled data and more confident predicted labels. Initially, the Decision Tree is trained on a labeled subset of images. Then, it predicts labels for the unlabeled images which is known as pseudo-labeling. The labeled and pseudo-labeled data are combined, and the model is retrained. Finally, the retrained model is used to classify images [3].

### Model 3: Supervised CNN

Convolutional Neural Networks (CNNs) are deep learning models tailored for grid-like data like images, learning spatial hierarchies through convolutional layers [4]. Initially, data undergoes augmentation for diversity. Then, a custom or pre-trained CNN architecture is chosen. The model trains on raw pixel values, extracting features via convolutional, pooling, and fully connected layers. Lastly, images are classified through the trained CNN [5]. This method will be implemented using PyTorch library.

### Metrics

Metrics are quantitative measures used to assess the performance of the models. We will ensure that all methods will be evaluated using the same metrics listed following.

1. Accuracy
2. Precision
3. Recall
4. F1-Score
5. Confusion matrix

### Comparison and Analysis

By comparing these methods using the following strategies, we can determine which approach best solves the problem of classifying images into venue labels considering both performance and practical constraints.

- **Standardized Metrics:** Compare accuracy, precision, recall, F1-Score for all three methods.
- **Performance Visualization:** Compare confusion matrix for all three methods to visualize wrong classifications.
- **Computational Complexity:** Consider training time and computational resources required for each method.
- **Optimization:** Focus on models’ performance through hyperparameter tuning. For Decision Trees, parameters like depth, number of branches, and pruning options need to be adjusted, while for the CNN, modifications such as adding or removing pooling layers and altering the number of convolutional layers are explored.

## References

1. MIT Places365 dataset. Available at: [https://www.kaggle.com/datasets/mittalshubham/images256/data](https://www.kaggle.com/datasets/mittalshubham/images256/data)
2. Decision Trees for classification and regression. Available at: [https://scikit-learn.org/stable/modules/tree.html](https://www.kaggle.com/datasets/mittalshubham/images256/data)
3. Semi-supervised learning with pseudo-labeling. Available at: [https://arxiv.org/abs/1908.02983](https://arxiv.org/abs/1908.02983)
4. Convolutional Neural Networks. Available at: [https://www.deeplearningbook.org/contents/convnets.html](https://www.deeplearningbook.org/contents/convnets.html)
5. Implementing CNNs using PyTorch. Available at: [https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
