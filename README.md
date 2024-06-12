# COMP6721 Summer2024 course project 
Image classification on five venues with regarding of three different models 

## Contributors
| Group    | Student Name | Student ID |
|----------|--------------|------------|
| CloseAI  | Yang Cao     | 26654029   |
|          | Hongwu Li    | 40280054   |
|          | Bo Shi       | 40292839   |

## Dataset

In terms of dataset, we use a subset of the [MIT Places365 dataset](https://www.kaggle.com/datasets/mittalshubham/images256/data) which contains 5893 pictures of 5 classes out of 365 classes from the original dataset. (For the original MIT Places365 dataset, see: [The Original Dataset](https://www.kaggle.com/datasets/mittalshubham/images256/data)) This dataset consists of 365 classes of different scene-centric pictures that all contain representative features of various locations. All the images are JPEG color images and resized to 256x256 resolution. With consideration of robustness and fitness, the five classes chosen from the original dataset are general, representative locations with specific features. The five classes are:

- **homeOffice** (1346/5893)
- **hotelRoom** (1524/5893)
- **church** (798/5893)
- **museum** (761/5893)
- **supermarket** (1466/5893)

We also randomly extract 700 images from each of the class to make a blanced dataset to do a comparison analysis against the original dataset. For both datasets, images are divided into training and testing images in an 8:2 ratio. The balanced dataset is prenented as follows:

- **homeOffice** (700/3500)
- **hotelRoom** (700/3500)
- **church** (700/3500)
- **museum** (700/3500)
- **supermarket** (700/3500)


## Installing the required dependencies and modules

```
pip install -r requirements.txt
```

### How to train model

1. **Clone the repository:**

```
git clone git@github.com:itshisher/COMP6721_Summer2024_Course_Project.git
```

2. **Setting Up an Environment:**

```
pip install torch
```
```
pip install torchvision
```

3. **Launch Jupyter Notebook:**

```
jupyter notebook
```

4. **Open and run the notebooks:**
   - Navigate to the directory called **COMP6721_Summer2024_Course_Project** in the Jupyter Notebook interface.
   - Choose one of the two folders (CNN, Desicion Tree) to continue. 
   - Select and run the notebook under the folder by clicking the "Run" button or using "Shift + Enter".
  
5. **Saved models:**
   - Saved models from **best_model_fold_1.pth** to **best_model_fold_10.pth** are runned by the file **Supervised_CNN_OriginalDataset_Transform_BS64_Epoch20_10foldCV_0.74.ipynb**
   - The saved model **best_model.pth** is runned by the file **Supervised_CNN_OriginalDataset_Transform_BS64_Epoch20_0.71.ipynb**
   - All models are available at the following link:
   ```
   https://drive.google.com/drive/folders/1NfSKp3qYk4uu6YsoAc2_uo4fIcsXbaWg?usp=sharing
   ```

**Notes:**

- All libraries used are mentiond at the beginning of each jupyter notebook file.
- The tested datasets are available under the main directory, which are called **original_dataset** and **balanced_dataset**.
- The project proposal and reports for two phases can be viewed under the main directory as well. 


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
