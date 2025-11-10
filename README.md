# Accident Severity Prediction in the UK

This project investigates the use of machine learning to classify the severity of road accidents in the UK. The dataset includes reported incidents from 2019 along with conditions such as weather, junction type, surface conditions and vehicle information.

The work compares several traditional machine learning models with a deep neural network. The aim was to understand whether a neural network could capture non-linear patterns more effectively than standard classifiers.

## Dataset
File: `OPTION1_uk_road_accident_2019_coursework_final.csv`

After cleaning and preprocessing, the dataset contains:
- 31,647 accident records
- Target variable: `accident_severity` (Slight, Serious, Fatal)
- Mixed categorical and numerical features
- Class imbalance across severity levels

## Preprocessing Steps
- Removed duplicated accident entries
- Handled missing categorical and numerical values
- Normalised numerical features
- One-hot encoded categorical features
- Split data into train, validation and test sets using stratified sampling
- Reduced dimensionality using PCA (kept 90% variance → 11 components)

## Models Trained
| Model | Balanced Accuracy |
|------|------------------|
| Logistic Regression | 75.78% |
| K-Nearest Neighbour | 72.57% |
| Decision Tree | 74.73% |
| Support Vector Machine | 74.90% |
| Random Forest | 77.05% |
| **Deep Neural Network (MLP)** | **80.65%** |

The deep neural network achieved the best balanced accuracy, indicating stronger performance in separating the three severity classes.

## Deep Neural Network Architecture
- Input: PCA-reduced features (11 inputs)
- Hidden layers: 2 fully-connected layers
- Activation: ReLU
- Dropout used to reduce overfitting
- Optimiser: Adam
- Loss: Sparse Categorical Cross-Entropy

## Repository Contents
| File | Description |
|------|-------------|
| `code_comp1804.ipynb` | Jupyter notebook containing model training and evaluation |
| `OPTION1_uk_road_accident_2019_coursework_final.csv` | Dataset used for modelling |
| `report_comp1804-option1.docx` | Full report including analysis, results and references |

## Key Outcome
The deep neural network performed best, suggesting that severity classification has non-linear feature interactions that are difficult for simpler models to fully capture.

## Full Report
The complete written analysis is available here:

**[Download the full report](report_comp1804-option1.docx)**
