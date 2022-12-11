# Traffic Sign Recognition

Identifying traffic signs (examples pictured in Figure 1) is a crucial milestone in self-driving technologies. Our work solves this problem with multiple machine learning algorithms; given the unbalanced nature of our data, we compare the macro and weighted F1 scores of classical methods against modern deep learning architectures. We find that in our filtered dataset, a convolutional neural network (activations visualized in Figure 2) achieves a macro F1 score of 0.99, with simpler methods also achieving scores of over 0.90. We also investigate our methods on a smaller training dataset, in which transfer learning outperforms other strategies by a huge margin.

Our full paper is included in the repository as `paper.pdf`.

<img src="https://user-images.githubusercontent.com/34076345/206936373-869c144d-b28d-480f-b284-95dfd3d308e8.png" width="500">

*Figure 1: Sample images from each of our 36 classes.*

<img src="https://user-images.githubusercontent.com/34076345/206936514-f5919b6a-b481-482b-98e2-f6d4ec2634c0.png" width="500">

*Figure 2: t-SNE projections of final convolutional layer activations onto 2-dimensional space.*
