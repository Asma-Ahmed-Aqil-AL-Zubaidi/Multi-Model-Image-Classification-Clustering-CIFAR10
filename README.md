# Multi-Model-Image-Classification-Clustering-CIFAR10
A complete deep learning pipeline for image classification and clustering on CIFAR-10 using custom CNNs, transfer learning (VGG16), and unsupervised learning (Autoencoder + KMeans), with Grad-CAM explainability.

##  Models Used
- Custom CNN (from scratch)
- Advanced CNN with BatchNorm & Dropout
- VGG16 Transfer Learning
- Autoencoder for Unsupervised Feature Extraction
- Grad-CAM for Visual Explainability

##  Dataset
CIFAR-10: 60,000 32×32 color images across 10 classes  
- Training: 50,000 images  
- Testing: 10,000 images

##  Main Features
-  Data exploration, cleaning, normalization, and augmentation  
-  Training and comparison of multiple CNN models  
-  Hyperparameter tuning using `ReduceLROnPlateau`  
-  Metrics: Accuracy, Precision, Recall, F1  
-  Grad-CAM for visual model interpretation  
-  KMeans clustering + t-SNE visualization

##  Results
| Metric     | Score   |
|------------|---------|
| Accuracy   | 0.8442  |
| Precision  | 0.8498  |
| Recall     | 0.8442  |
| F1-Score   | 0.8423  |

##  Visuals
- Confusion matrix
- Grad-CAM heatmaps
- Most confident correct & incorrect predictions
- Clustering with t-SNE

##  File Structure
- `DL2.ipynb`: Full code for the pipeline
- `models/`: Trained model files
- `results/`: CSVs, plots, and Grad-CAM outputs
- `requirements.txt`: Required libraries
- `README.md`: This file

##  Libraries Used
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn
- Optuna (if used in extended notebook)


