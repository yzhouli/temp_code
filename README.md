# Description

Source code corresponding to the paper "Z. Yang, Y. Pang, Haoyang Zhang, Hongbo Yin, and Yunpeng Xiao, “User to Audio: A Multi-feature Fusion Model for Spammer Detection Inspired by Audio Classification Technology,” AAAI 2025.

# Datasets

**Twitter:**  A publicly available dataset [1] was used as training data for the model on the Twitter platform. Since the dataset is publicly available, we do not provide the source dataset. Subsequently, a processed version of the dataset that takes up little memory (original) and a version of the dataset that can be trained directly are provided.

**Weibo:** Because official datasets for studying spammers on the Weibo platform are scarce and user information is non-recourseable. Therefore, we constructed the fully open-source Weibo dataset. A detailed description of the *collection process* and *user label definitions* for this dataset can be found in the file **Supplement_for_Reproducibility.md**.

In particular, we provide processing files between dataset versions: **datasets/preprocess.py**, **datasets/config.py**, and **datasets/text_clear.py**.

[1] Yang, C.; Harkreader, R.; Zhang, J.; Shin, S.; and Gu, G. 2012. Analyzing spammers’ social networks for fun and profit: a case study of cyber criminal ecosystem on twitter. In Proceedings of the 21st international conference on World Wide Web, 71–80.

# Spammer Detection

**Source Code:** Source code for the user audioization model. The file of **spammer_detection/run.py**  is the main entry file for the model training and prediction process.

**Settings and BaseLine Methods:** We provide a series of Python library versions for model replication in the file **Supplement_for_Reproducibility.md**. Similarly, *detailed experimental setups and descriptions* of other baseline methods used in the manuscript are likewise provided.
