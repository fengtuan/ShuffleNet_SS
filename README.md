Protein Secondary Structure Prediction Using a Lightweight Convolutional Network and Label Distribution Aware Margin Loss
=

**Abstract**
<br>
Protein secondary structure prediction (PSSP) is an important task in computational molecular biology. Recently, deep neural networks have demonstrated great potential in improving the performance of eight-class PSSP. However, the existing deep predictors usually have higher model complexity and ignore the class imbalance of eight-class secondary structure data in training. In addition, the current methods cannot guarantee that the features corresponding to the padded residue positions are always zero during the forward propagation of the network, which will cause the prediction results of the same protein chain to be different under varied zero-padding numbers. To this end, we propose a novel lightweight convolutional network ShuffleNet_SS, which adopts modified 1-dimensional batch normalization to eliminate the impact of padded residue positions on nonpadded residue positions and uses the label distribution aware margin loss to enhance the network’s ability to learn rare classes. In particular, in order to enable ShuffleNet_SS to fully achieve cross-group information exchange, we further improve the standard channel shuffle operation. Experimental results on the benchmark datasets including CASP10, CASP11, CASP12, CASP13, CASP14 and CB513 show that the proposed method achieves state-of-the-art performance with much lower parameters compared to the five existing deep predictors.





please run ./run_evaluate.sh
