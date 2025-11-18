# Classification-Biased Apparent Brain Age for the Prediction of Alzheimer's Disease

### Keywords
Alzheimer's disease, brain age, magnetic resonance imaging, machine learning, predictive and descriptive models, explainable artificial intelligence

### Authors
- Ali Varzandian,
- Miguel Angel Sanchez Razo,
- Michael Richard Sanders,
- Akhila Atmakuru and
- Giuseppe Di Fatta 

# Abstract
Machine Learning methods are often adopted to infer useful biomarkers for the early diagnosis of many neurodegenerative diseases and, in general, of neuroanatomical ageing. Some of these methods estimate the subject age from morphological brain data, which is then indicated as “brain age”. The difference between such a predicted brain age and the actual chronological age of a subject can be used as an indication of a pathological deviation from normal brain ageing. An important use of the brain age model as biomarker is the prediction of Alzheimer's disease (AD) from structural Magnetic Resonance Imaging (MRI). Many different machine learning approaches have been applied to this specific predictive task, some of which have achieved high accuracy at the expense of the descriptiveness of the model. This work investigates an appropriate combination of data science techniques and linear models to provide, at the same time, high accuracy and good descriptiveness. The proposed method is based on a data workflow that include typical data science methods, such as outliers detection, feature selection, linear regression, and logistic regression. In particular, a novel inductive bias is introduced in the regression model, which is aimed at improving the accuracy and the specificity of the classification task. The method is compared to other machine learning approaches for AD classification based on morphological brain data with and without the use of the brain age, including Support Vector Machines and Deep Neural Networks. This study adopts brain MRI scans of 1, 901 subjects which have been acquired from three repositories (ADNI, AIBL, and IXI). A predictive model based only on the proposed apparent brain age and the chronological age has an accuracy of 88% and 92%, respectively, for male and female subjects, in a repeated cross-validation analysis, thus achieving a comparable or superior performance than state of the art machine learning methods. The advantage of the proposed method is that it maintains the morphological semantics of the input space throughout the regression and classification tasks. The accurate predictive model is also highly descriptive and can be used to generate potentially useful insights on the predictions.

### Link to the paper
DOI:
https://doi.org/10.3389/fnins.2021.673120
