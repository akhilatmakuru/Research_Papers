# Transfer Learning for the Cognitive Staging Prediction in Alzheimer’s Disease

### Authors 
Akhila Atmakuru, Atta Badii & Giuseppe Di Fatta 

### Abstract
Alzheimer’s Disease (AD) is a life-threatening neurodegenerative disease with far-reaching global implications. Deep neural network-based techniques are being utilized to predict and diagnose AD at every stage. The Mini-Mental State Examination (MMSE) scores are essential for monitoring the onset and progression of the disease since they serve as numerical assessments of cognitive function. The present paper describes a novel multi-step algorithm for predicting MMSE scores in patients with AD. Initially, a regression analysis was carried out to predict the patient’s age on a combined MCI dataset created by including individuals with mild cognitive impairment (MCI), early cognitive impairment (EMCI), and late cognitive impairment (LMCI). Subsequently, transfer learning techniques were applied to incorporate knowledge from the regression model into an autoencoder. The autoencoder extracted significant features from the combined MCI dataset, creating meaningful encoded representations. A regression analysis was employed to predict MMSE scores based on the encoded features, and subsequently classify patients into two categories Mild and Moderate based on cognitive status. This strategy achieved an accuracy of approximately 73.26% with a 3.92% standard deviation. For comparison, a simple regression model without employing Transfer Learning and Auto Encoders was implemented. This simple regression model gave an accuracy of 61.08% with 2.21% Standard Deviation, indicating an enhancement of approx. 12.18% accuracy due to transferred knowledge. This comparison illustrated the effectiveness of the proposed methodology. Cross-validation techniques were used to analyze the stability and applicability of the approach, confirming consistency and performance across subsets of the dataset. The results highlight the potential of transfer learning and autoencoder-based feature extraction to improve predicting MMSE scores for AD patients.

### Link to the paper
DOI:
https://doi.org/10.1007/978-3-031-82487-6_13
