# tensorflow-learn
ML/AI experiments using tensorflow

The original WISDM dataset can be downloaded from http://www.cis.fordham.edu/wisdm/dataset.php

Included in the data/ directory, is the transformed dataset.  Every 200 sequential original readings are massaged into a single record of 46 features, which are described in the WISDM_ar_v1.1_trans_about.txt.

The transformed dataset is in arff format: data are in comma-separated, lines beginning for '@' describe attributes and sections.
Missing features in a sample are represented by '?'.  As such, this arff file needs cleaning during the import phase or offline.

The code here plays with different predictors:
- random forest via tensorflow

To be tried:
- scikit's Random Forest regressor
