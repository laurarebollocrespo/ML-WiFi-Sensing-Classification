# TODO List for ML WiFi Sensing Classification


1) **Setup & Exploratory Data Analysis (EDA)**
*Where: notebooks/01_EDA_and_Setup.ipynb*
    - [ ] Load the Data: Import train.csv (12,888 samples) and test_nolabels.csv (3,223 samples).
    - [ ] Create a Local Validation Split: Use an 80/20 or 90/10 split on train.csv so you can test models locally.
    - [ ] Verify Class Balance: Plot a histogram of the position label (0 to 9). The brief states classes are well-balanced (~1,500 per class). Verify this.
    - [ ] Analyze Metadata: Plot the relationship between the estimated angle of arrival (aoa) and the two signal strengths (rssi_1, rssi_2). See if these three features alone offer any predictive power.
    - [ ] Visualize Raw CSI: Pick a sample from Position 0 and a sample from Position 5. Plot the In-phase and Quadrature values across the 64 subcarriers for both antennas to see if they look distinct.  
    - [ ] Unsupervised Reality Check: Run K-Means Clustering on the raw dataset (ignoring the labels). Set $k=10$. Use a Silhouette Score or the Elbow Method to see if the data naturally clusters into the 10 physical locations.  
    - [ ] Outlier Detection: Run DBSCAN to find "noisy" samples (e.g., moments when someone walked in front of the tracking unit). Consider dropping these outliers from your training set to improve model stability.

2) **Feature Engineering & Preprocessing**

    *Where: src/features.py*
    - [ ] Calculate Amplitude: For all 64 subcarriers across both antennas, calculate the magnitude: $\text{Amplitude} = \sqrt{I^2 + Q^2}$.
    - [ ] Calculate Phase: For all 64 subcarriers across both antennas, calculate the phase angle: $\text{Phase} = \arctan(Q/I)$.
    - [ ] Handle Collinearity: Calculate a Correlation Matrix. If adjacent subcarriers are highly correlated (e.g., subcarrier 10 and 11 have a correlation $> 0.95$), note this. You may need to use PCA (Principal Component Analysis) or L1/L2 Regularization later to handle this multicollinearity.  
    - [ ] Standardization (Crucial Step): Scale all features (Amplitude, Phase, rssi, aoa) so they have a mean of 0 and a variance of 1. If you skip this, models relying on distance (KNN, SVM, Neural Networks) will fail completely. 

3) **The Fast Baselines**
    
    *Goal: Establish a minimum F1-score using simple models. If a complex model cannot beat these, it is not worth using.*
    *Where: src/models.py*

    - [ ] Softmax Regression: Train this multi-class extension of Logistic Regression. It is fast and provides a solid linear baseline.  
    - [ ] Linear Discriminant Analysis (LDA): Train this generative classifier. It assumes all 10 positions share the same covariance matrix.  
    - [ ] K-Nearest Neighbors (KNN): Train a KNN model on your standardized features. Use cross-validation to find the optimal $K$ value (e.g., testing $K=3$ through $K=15$).  
    - [ ] Submit to Kaggle: Pick your best model so far. Generate predictions for test_nolabels.csv. Ensure the output is a 2-column CSV (ID, position) exactly 3,223 rows long. Submit it to ensure your pipeline formatting is correct.   


4) **The Heavy HittersGoal**

    *Train complex models capable of finding non-linear decision boundaries in your high-dimensional CSI data.*
    
    *Where: src/models.py*
    - [ ] Support Vector Machine (SVM): Train an SVM using a Radial Basis Function (RBF) Kernel. This maps the features into a higher-dimensional space to find non-linear boundaries.  Task: Use Grid Search to validate the $C$ (regularization) and $\gamma$ (kernel width) hyperparameters to balance the bias-variance tradeoff. 
    - [ ] Multilayer Artificial Neural Network (MLNN): Build a neural network with 1 or 2 hidden layers using non-linear activation functions (like ReLU). This model can learn deep representations of the CSI amplitude graphs.
    - [ ] Classification Trees (CART): Train a single decision tree. Note its performance and how quickly it overfits the data (high variance). You will use this to understand why Phase 5 is necessary.
5) **Ensembles**
    
    *Goal: Combine multiple weak learners to drastically reduce variance and bias, achieving the highest possible F1-score.*
    
    *Where: src/models.py and notebooks/02_Ensemble_Tuning.ipynb*

    - [ ] Random Forest: Train this ensemble of decision trees. It uses "bagging" (bootstrap aggregating) and random feature selection to reduce the high variance you saw in the single CART model.  
    - [ ] Gradient Boosting: Train an XGBoost or LightGBM model. This builds sequential trees, where each new tree tries to correct the residual errors of the previous trees. This is often the strongest standalone model for tabular data.  
    - [ ] Stacking (The Final Push): Take the predictions from your tuned Random Forest, your SVM, and your Gradient Booster. Feed those predictions as new features into a final "Blender" model (like Softmax Regression) to make the ultimate decision.  

6) **Other Models** 

    - [ ] 2D Convolutional Neural Networks (CNNs)The Concept: Standard models (like Random Forest) treat your 256 raw CSI features as independent variables. However, we know they are physically related: they represent 64 adjacent subcarriers across 2 antennas. You can reshape your 256 flat features into a $4 \times 64$ matrix (Antenna 1 In-phase, Antenna 1 Quadrature, Antenna 2 In-phase, Antenna 2 Quadrature).  Why it works: By feeding this matrix into a CNN, the model treats the CSI data like an image, using convolutional filters to learn spatial patterns and frequency distortions caused by the room's physical environment. CNNs are the current backbone of most modern CSI localization research.

    - [ ] Decision Fusion CNNs (DF-CNN)The Concept: Instead of feeding both antennas into one network, you build two parallel CNNs. Antenna 1 goes into Model A, and Antenna 2 goes into Model B. The network processes them independently and then fuses their outputs at the very end to make a final decision.  Why it works: Recent studies show that processing CSI channels separately prevents the network from overlooking valuable, channel-specific multipath reflections, achieving significantly higher accuracy than traditional single-input approaches.  


# **MODELS TO TRY** (*From Class*)

### **Parametrized Gaussian**
- [ ] Linear Regression  
- [ ] LDA  
- [ ] QDA  

### **Non-parametric**
- [ ] Parzen Windows  
- [ ] KNN  

### **Unknown**
- [ ] Logistic Regression  
- [ ] Softmax Regression  
- [ ] NN  
- [ ] SVM (different kernels)  

### **Ensembles**
- [ ] Random Forest  
- [ ] AdaBoost  
- [ ] Gradient Boosting  
- [ ] Stacking  

### **Unsupervised Learning Models**
- [ ] Gaussian Mixture Models via EM  
- [ ] K‑Means Clustering  
- [ ] Fuzzy K‑Means  
- [ ] DBSCAN  
- [ ] HDBSCAN  
