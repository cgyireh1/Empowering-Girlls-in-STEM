# Empowering-Girls-in-STEM

## Project Overview

### Objective
The primary objective of this project is to analyze factors influencing girls' pursuit of STEM (Science, Technology, Engineering, and Mathematics) careers by using a machine learning model to predict the likelihood of students pursuing STEM based on various demographic, educational, and extracurricular factors.

### Dataset Used
The dataset used in this project is the **Student Performance Data** from Keggle. It contains 2392 samples and 16 features, including:

- **Demographic Information**: Age, Gender, Ethnicity
- **Educational Factors**: Parental Education, GPA, Study Time, Absences
- **Extracurricular Activities**: Tutoring, Sports, Music, Volunteering
- **Target Variable**: GPA_Pass (indicating whether the student has a GPA greater than 2.5)

### Key Findings
1. **Model Performance**:
   - **Simple Model**: 
     - Test Loss: 0.5713
     - Test Accuracy: 92.48%
   - **Optimized Model**: 
     - Test Loss: 0.1631
     - Test Accuracy: 94.99%
   
2. **F1 Score and Specificity**:
   - F1 Score: 0.9143, indicating a good balance between precision and recall.
   - Specificity: 0.9684, showing the model's effectiveness at correctly identifying students who are not pursuing STEM.

3. **Confusion Matrix**: The confusion matrices for both models show the distribution of true and predicted classes, providing insight into the model's performance.

### Instructions for Running the Notebook
1. **Loading the Notebook**:
   - Open the `Empowering-Girls-in-STEM.ipynb` file in Colab or any compatible environment.

2. **Running the Notebook**:
   - Execute each cell sequentially to preprocess the data, train the models, and evaluate their performance respectively.

3. **Loading Saved Models**:
   - To load the trained models, use the following code:
     ```python
     from tensorflow import keras

     # Load the models
     model = keras.models.load_model('model.keras')
     optimized_model = keras.models.load_model('optimized_model.keras')
     ```

---

## Optimization Techniques Used

### 1. Early Stopping
- **Principle**: Early stopping is a regularization technique used to prevent overfitting during training. It monitors the model's performance on the validation dataset and halts training when performance stops improving.
- **Implementation**: As shown in cell 11 and as below.👇
  ```python
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  ```
  - **Parameters Used**:
    - `monitor` is the quantity to be monitored and in this project, validation loss was the one being monitored, **(val_loss)**.
    - `patience` defines how many epochs to wait for an improvement in the monitored metric before stopping. A patience value of 5 in this project means that if the validation loss does not improve for 5 consecutive epochs, training is stopped.
   
    - Using the early stopping helps to maintain a balance between training a model adequately and preventing it from learning noise in the training data and the patience was set to 5 based on observations from initial runs, where performance typically stabilized after a few epochs.

### 2. Regularization
- **L1 and L2 Regularization**:
  - **L1 Regularization**: It adds the absolute value of the weights to the loss function. This can lead to sparse models, effectively selecting important features while reducing the impact of less important ones.
    ```python
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1(0.01))
    ```
  - **L2 Regularization**: It adds the squared value of the weights to the loss function. This helps to keep all weights small and reduces model complexity.
    ```python
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
    ```
  -The regularization strength (`0.01`) was chosen based on experimentation to balance the model complexity and its performance.

  In summary, Regularization was used to prevent overfitting and help in feature selection.

### 3. Dropout
- **Principle**: Dropout is another regularization technique that randomly sets a fraction of input units to 0 during training, preventing overfitting.
- **Implementation**:
  ```python
  layers.Dropout(0.2)
  ```
  - **Rate**: The dropout rate (`0.2`) was chosen based on experiments showing improved validation performance without significant loss in training accuracy to provide a reduction in overfitting.

### Conclusion

The combination of these optimization techniques has resulted in improved model performance, as evidenced by the significant reduction in loss and increase in accuracy in the optimized model.
