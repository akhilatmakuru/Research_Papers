# Sensitivity Analysis for Feature Importance in Predicting Alzheimer’s Disease

### Authors 
- Akhila Atmakuru,
- Giuseppe Di Fatta,
- Giuseppe Nicosia,
- Ali Varzandian and
- Atta Badii 

### Abstract
Artificial Intelligence (AI) classifier models based on Deep Neural Networks (DNN) have demonstrated superior performance in medical diagnostics. However, DNN models are regarded as “black boxes” as they are not intrinsically interpretable and, thus, are reluctantly considered for deployment in healthcare and other safety-critical domains. In such domains explainability is considered a fundamental requisite to foster trust and acceptability of automatic decision-making processes based on data-driven machine learning models. To overcome this limitation, DNN models require additional and careful post-processing analysis and evaluation to generate suitable explainability of their predictions. This paper analyses a DNN model developed for predicting Alzheimer’s Disease to generate and assess explainability analysis of the predictions based on feature importance scores computed using sensitivity analysis techniques. In this study, a high dimensional dataset was obtained from Magnetic Resonance Imaging of the brain for healthy subjects and for Alzheimer’s Disease patients. The dataset was annotated with two labels, Alzheimer’s Disease (AD) and Cognitively Normal (CN), which were used to build and test a DNN model for binary classification. Three Global Sensitivity Analysis (G-SA) methodologies (Sobol, Morris, and FAST) as well as the SHapley Additive exPlanations (SHAP) were used to compute feature importance scores. The results from these methods were evaluated for their usefulness to explain the classification behaviour of the DNN model. The feature importance scores from sensitivity analysis methods were assessed and combined based on similarity for robustness. The results indicated that features related to specific brain regions (e.g., the hippocampal sub-regions, the temporal horn of the lateral ventricle) can be considered very important in predicting Alzheimer’s Disease. The findings are consistent with earlier results from the relevant specialised literature on Alzheimer’s Disease. The proposed explainability approach can facilitate the adoption of black-box classifiers, such as DNN, in medical and other application domains.

### Link to the paper
DOI:
https://doi.org/10.1007/978-3-031-53966-4_33
