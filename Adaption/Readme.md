# Domain adaption

# Training Statistics
* `python view_data.py` -> watch domain image & target image
* `python main.py` -> training & inference
* lambda 根據[論文](https://arxiv.org/pdf/1505.07818.pdf) page.21調整
* epochs = 2000

# Result
* pass boss baseline
`Result/`
# Todo
1. recurrent training
2. 將原domain上的圖像做 `style transfer` 後以三個模型[DANN,MCD,MSDA]所預測出target domain中相同的答案為 label 當做target domain 的 Pseudo label,並在之後對其做Semi-Supervised
Learning
# Reference
* https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/hw/HW11/HW11.pdf
* https://drive.google.com/file/d/11uNDcz7_eMS8dMQxvnWsbrdguu9k4c-c/view
* https://drive.google.com/file/d/1xIkSs8HAShdcfV1E0NEnf4JDbL7POZTf/view

