## Student Evaluation Data

In this project, we try to reproduce the results obtained by this [paper](https://github.com/abishekarun/Student-Evaluation-dataset/blob/master/LA_EdMining_SanghoSuh.pdf) on [Turkiye Student evaluation dataset](http://archive.ics.uci.edu/ml/datasets/turkiye+student+evaluation). This project was also used to explore different regression models in matlab.

The final matlab file is [here](https://github.com/abishekarun/Student-Evaluation-dataset/blob/master/student_evaluation.m) for this project and rendered html file is [here](http://htmlpreview.github.io/?https://github.com/abishekarun/Student-Evaluation-dataset/blob/master/html/student_evaluation.html).

I also tried to replicate the results using weka explorer and experimenter. I got the exact same results using Weka explorer but due to different splits in experimenter, the results were not the same. You can find the benchmark results using weka [here](https://github.com/abishekarun/Student-Evaluation-dataset/blob/master/weka_benchmark_results.csv) and the improved results after considering only four columns [here](https://github.com/abishekarun/Student-Evaluation-dataset/blob/master/weka_improved_results.csv).

The resources that helped me are:

+ [Weka Explorer vs Experimenter](https://stackoverflow.com/questions/12495877/weka-differences-between-explorer-and-experimenter-outcomes)
+ [Machine learning with matlab](https://pdfs.semanticscholar.org/presentation/2c02/efcb9ac85f7230e4e4687fdd9607b385337b.pdf)
+ [Data Analysis with matlab](https://github.com/abishekarun/Transport-Problem/blob/master/data_analysis.pdf) 
+ [Data mining with Weka](https://www.ibm.com/developerworks/library/os-weka1/index.html)

