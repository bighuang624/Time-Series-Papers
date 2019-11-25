# Awesome Time Series Papers

> **Notice:** Thanks for all your interest. However, unfortunately, since I'm no longer working on time series analysis, **I may not be able to continue adding content by myself**. If someone **pulls requests**, I will update it. Better yet, if you are willing to **fork this repository and keep updating**, please contact me and I will add your repository link here. Thanks.

- Introduction:
    - List of awesome papers from various research fields in time series analysis, mainly including algorithms based on machine learning. 
    - `*` after the title of the paper indicates that the full paper has been carefully read by [me](https://github.com/bighuang624).
    - A link of open source code is given if avaliable on [Papers With Code](https://paperswithcode.com/) or [Github](https://github.com/).
    - Some papers will give links to reading notes.
    - Any [contributions](https://github.com/bighuang624/Time-Series-Papers/blob/master/.github/contribution_template.md) including PR are welcomed.

- 介绍：
    - 收录时间序列分析中各个研究领域的高水平文章，主要包含基于机器学习的算法。
    - 论文标题后的`*`号表示[我](https://github.com/bighuang624)仔细地读过全文。
    - 如果 [Papers With Code](https://paperswithcode.com/) 或 [Github](https://github.com/) 上存在该论文的开源代码，则给出其链接。
    - 有些论文会给出解读的链接。
    - 欢迎包括 PR 在内的一切[贡献](https://github.com/bighuang624/Time-Series-Papers/blob/master/.github/contribution_template.md)。

- - -

## Catelog (目录)     <!-- omit in toc -->
- [Awesome Time Series Papers](#awesome-time-series-papers)
  - [Review (综述)](#review-%e7%bb%bc%e8%bf%b0)
      - [2014](#2014)
      - [2012](#2012)
  - [Time Series Forecasting (时间序列预测)](#time-series-forecasting-%e6%97%b6%e9%97%b4%e5%ba%8f%e5%88%97%e9%a2%84%e6%b5%8b)
    - [Univariate (单变量)](#univariate-%e5%8d%95%e5%8f%98%e9%87%8f)
      - [2018](#2018)
    - [Multivariate to Univariate (多变量预测单变量)](#multivariate-to-univariate-%e5%a4%9a%e5%8f%98%e9%87%8f%e9%a2%84%e6%b5%8b%e5%8d%95%e5%8f%98%e9%87%8f)
      - [2018](#2018-1)
      - [2017](#2017)
    - [Multivariate to Multivariate (多变量预测多变量)](#multivariate-to-multivariate-%e5%a4%9a%e5%8f%98%e9%87%8f%e9%a2%84%e6%b5%8b%e5%a4%9a%e5%8f%98%e9%87%8f)
      - [2018](#2018-2)
  - [Time Series Classification (时间序列分类)](#time-series-classification-%e6%97%b6%e9%97%b4%e5%ba%8f%e5%88%97%e5%88%86%e7%b1%bb)
      - [2018](#2018-3)
      - [2017](#2017-1)
  - [Time Series Clustering (时间序列聚类)](#time-series-clustering-%e6%97%b6%e9%97%b4%e5%ba%8f%e5%88%97%e8%81%9a%e7%b1%bb)
      - [2019](#2019)
      - [2018](#2018-4)
      - [2016](#2016)
      - [2015](#2015)
  - [Anomaly Detection (异常检测)](#anomaly-detection-%e5%bc%82%e5%b8%b8%e6%a3%80%e6%b5%8b)
      - [2019](#2019-1)
      - [2018](#2018-5)
      - [2017](#2017-2)
  - [Sequence Modeling (序列建模)](#sequence-modeling-%e5%ba%8f%e5%88%97%e5%bb%ba%e6%a8%a1)
    - [Supervised (有监督)](#supervised-%e6%9c%89%e7%9b%91%e7%9d%a3)
      - [2018](#2018-6)
    - [Unsupervised (无监督)](#unsupervised-%e6%97%a0%e7%9b%91%e7%9d%a3)
      - [2019](#2019-2)
      - [2018](#2018-7)
  - [Query by Content (按内容查询)](#query-by-content-%e6%8c%89%e5%86%85%e5%ae%b9%e6%9f%a5%e8%af%a2)
  - [Time Series Segmentation (时间序列分割)](#time-series-segmentation-%e6%97%b6%e9%97%b4%e5%ba%8f%e5%88%97%e5%88%86%e5%89%b2)
  - [Motif Discovery (重复模式发现)](#motif-discovery-%e9%87%8d%e5%a4%8d%e6%a8%a1%e5%bc%8f%e5%8f%91%e7%8e%b0)
  - [Study of Stock Market (股票市场研究)](#study-of-stock-market-%e8%82%a1%e7%a5%a8%e5%b8%82%e5%9c%ba%e7%a0%94%e7%a9%b6)
      - [2017](#2017-3)
  - [Spatio-temporal Forecasting (时空预测)](#spatio-temporal-forecasting-%e6%97%b6%e7%a9%ba%e9%a2%84%e6%b5%8b)
    - [Traffic Prediction (交通预测)](#traffic-prediction-%e4%ba%a4%e9%80%9a%e9%a2%84%e6%b5%8b)
      - [2018](#2018-8)
  - [Others (其他)](#others-%e5%85%b6%e4%bb%96)
      - [2019](#2019-3)
      - [2018](#2018-9)

- - -

## Review (综述)

#### 2014

- **A review of unsupervised feature learning and deep learning for time-series modeling** [[paper](http://www.diva-portal.org/smash/get/diva2:710518/FULLTEXT02)]

#### 2012

- **Time-series data mining** [[paper](https://hal.archives-ouvertes.fr/hal-01577883/document)]

- - -

## Time Series Forecasting (时间序列预测)

Time series forecasting is the task of predicting future values of a time series (as well as uncertainty bounds).

### Univariate (单变量)

#### 2018

- **RESTFul: Resolution-Aware Forecasting of Behavioral Time Series Data** (**CIKM2018**) [[paper](https://dl.acm.org/citation.cfm?id=3271794)] *
    - Propose a multi-resolution time series forecasting model *RESTFul*, which develops a recurrent framework to encode the temporal patterns at each resolution, and a convolutional fusion framework to model the inter-dependencies between the sequential patterns with different time resolutions
    - 提出多粒度时序预测模型 *RESTFul*，该模型使用一个循环神经网络来编码每个粒度下的时间维度特征，以及一个卷积融合框架来模拟不同时间粒度的特征之间的互相依赖关系

### Multivariate to Univariate (多变量预测单变量)

The model predicts the current value of **a time series** based upon its previous values as well as the current and past values of multiple driving (exogenous) series.

#### 2018

- **TADA: Trend Alignment with Dual-Attention Multi-task Recurrent Neural Networks for Sales Prediction** (**ICDM2018**) [[paper](http://net.pku.edu.cn/daim/hongzhi.yin/papers/ICDM18.pdf)]
    - Divide the influential factors into internal feature and external feature, which are jointly modelled by a multi-task RNN encoder. In the decoding stage, *TADA* utilizes two attention mechanisms to compensate for the unknown states of influential factors in the future and adaptively align the upcoming trend with relevant historical trends to ensure precise sales prediction
    - 将影响因素分为内部特征和外部特征，由多任务 RNN 编码器联合建模。在解码阶段，*TADA* 利用两种注意力机制来补偿未来影响因素的未知状态，并将未来的趋势与相关的历史趋势相适应，以确保准确预测销量

#### 2017

- **A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction** (**IJCAI2017**) [[paper](https://arxiv.org/pdf/1704.02971v4.pdf)] [[code](https://paperswithcode.com/paper/a-dual-stage-attention-based-recurrent-neural)] *
    - Propose *DA-RNN*, which consists of an encoder with an input attention mechanism to select relevant driving series, and a decoder with a temporal attention mechanism to capture long-range temporal information of the encoded inputs
    - 提出 *DA-RNN*，其包含一个带有 input attention 机制的编码器来选择相关外部序列，和一个带有 temporal attention 机制的解码器来捕获已编码输入中的长期时间信息

### Multivariate to Multivariate (多变量预测多变量)

The models predicts the future values of **multivariate time series** only based upon their previous values.

#### 2018

- **Modeling Long- and Short-Term Temporal Patterns with Deep Neural Network** (**SIGIR2018**) [[paper](https://arxiv.org/pdf/1703.07015v3.pdf)] [[code](https://paperswithcode.com/paper/modeling-long-and-short-term-temporal)] *
    - Propose *LSTNet*, which contains a recurrent-skip layer or a temporal attention layer to capture a mixture of short-term and long-term repeating patterns
    - 提出 *LSTNet*，使用 recurrent-skip layer 或 temporal attention layer 来建模短期和长期重复模式的混合

- **A Memory-Network Based Solution for Multivariate Time-Series Forecasting** [[paper](https://arxiv.org/pdf/1809.02105v1.pdf)] [[code](https://github.com/Maple728/MTNet)] *
    - Propose *MTNet*, which uses a memory component and attention mechanism to store the long-term historical data and deal with a period of time rather than a single time step
    - 提出 *MTNet*，使用一个记忆模块和注意力机制来存储长期的历史数据，并且可以同时处理一段序列而非单独的时间步

- **Temporal Pattern Attention for Multivariate Time Series Forecasting** [[paper](https://arxiv.org/pdf/1809.04206v2.pdf)] [[code](https://github.com/gantheory/TPA-LSTM)] *
    - Propose *Temporal Pattern Attention*, which learns to select not only time steps but also series relevant to the prediction
    - 提出 *Temporal Pattern Attention*，不仅能够选择与预测相关的时间步，还能够考虑到不同变量的影响

- - -

## Time Series Classification (时间序列分类)

Time series forecasting is the task of assigning time series pattern to a specific category.

#### 2018

- **Towards a Universal Neural Network Encoder for Time Series** (**CCIA2018**) [[paper](https://arxiv.org/pdf/1805.03908.pdf)] *
    - Use multi-task learning to enable a time series encoder to learn representations that are useful on data set types with which it has not been trained on. The encoder is formed of a convolutional neural network whose temporal output is summarized by a convolutional attention mechanism
    - 使用多任务学习的方式让时序编码器学习对未接触过的数据集类型有用的表示。本文使用的编码器由卷积神经网络构成，其时间输出由卷积注意机制汇总而成

- **Extracting Statistical Graph Features for Accurate and Efficient Time Series Classification** [[paper](http://openproceedings.org/2018/conf/edbt/paper-90.pdf)]
    - Present a multiscale graph representation for time series as well as feature extraction methods for classification, so that both global and local features from time series are captured
    - 提出时间序列的多尺度图表示以及从图中提取用于分类的特征的方法，以同时捕获时间序列中的全局和局部特征
    - [中文解读](https://zhuanlan.zhihu.com/p/58714287) 

- **The UEA multivariate time series classificationarchive, 2018** [[paper](https://arxiv.org/pdf/1811.00075.pdf)]
    - Release 30 multivariate time series classification datasets and benchmark results with three standard classifiers: 1NN + ED/DTW_I/DTW_D
    - 发布了 30 个多变量时间序列分类数据集，以及通过三个标准分类器（1NN + ED/DTW_I/DTW_D）得到的基准结果

- **Transfer learning for time series classification** (**IEEE Big Data 2018**) [[paper](https://arxiv.org/pdf/1811.01533.pdf)] [[code](https://github.com/hfawaz/bigdata18)] *
    - Extensive experiments show that transferring the network's weights works on time series classification task, and the choice of the source dataset impacts significantly on the model's generalization capabilities
    - 使用大量实验表明，模型权值的迁移促进其在目标数据集上分类任务的表现，且源数据集的选择对模型的泛化能力有显著影响

#### 2017

- **Time series classification from scratch with deep neural networks: A strong baseline** (**IJCNN2017**) [[paper](https://arxiv.org/pdf/1611.06455.pdf)] [[code](https://paperswithcode.com/paper/time-series-classification-from-scratch-with#code)]
    - Propose *Fully Convolutional Network (FCN)*, which can be a strong baseline for similar tasks as one of the earliest deep learning time series classifiers
    - 提出完全卷积网络（FCN），作为最早的深度学习时间序列分类器之一，它可以作为类似任务的强基准模型

- - -

## Time Series Clustering (时间序列聚类)

Time series clustering is the task of forming clusters given a set of unlabeled time series data.

#### 2019

- **SOM-VAE: Interpretable Discrete Representation Learning on Time Series** (**ICLR2019**) [[paper](https://arxiv.org/pdf/1806.02199.pdf)] [[code](https://paperswithcode.com/paper/som-vae-interpretable-discrete-representation)] [[SOM-YouTube](https://www.youtube.com/watch?v=3osKNPyAxPM)]
    - Design *SOM-VAE* for interpretable discrete representation learning on time series, and show that the latent probabilistic model in the representation learning architecture improves clustering and interpretability of the representations on time series
    - 针对时间序列上的可解释离散表示学习设计了 *SOM-VAE*，并表明在表示学习体系结构中的潜在概率模型提高了时间序列表示的聚类效果和可解释性

#### 2018

- **Deep Temporal Clustering: Fully Unsupervised Learning of Time-Domain Features** (**ICLR2018**) [[paper](https://arxiv.org/pdf/1802.01059.pdf)] [[code](https://github.com/saeeeeru/dtc-tensorflow)] *
    - Integrate dimensionality reduction and temporal clustering into a single end-to-end learning framework to jointly optimize
    - 将降维和时序聚类集成到一个端到端神经网络，以进行联合优化

#### 2016

- **Unsupervised Feature Learning from Time Series** (**IJCAI2016**) [[paper](https://pdfs.semanticscholar.org/b4f5/8e005541c54b146e67b09094f09ba3297906.pdf)]
    - Present a new Unsupervised Shapelet Learning Model (USLM) to learn shapelets, which combines pseudo-class label, spectral analysis, shapelets regularization and regularized least-squares for learning (shapelets are time series short segments that can best predict class labels)
    - 给出一个结合了伪类标签、谱分析、shapelets 正则化和正则化最小二乘法的无监督 Shapelet 学习模型（shapelets 是时间序列的短片段，能够最好地预测类标签）

#### 2015

- **k-Shape: Efficient and Accurate Clustering of Time Series** (**SIGMOD2015**) [[paper](http://web2.cs.columbia.edu/~gravano/Papers/2015/sigmod2015.pdf)] [[code](https://github.com/Mic92/kshape)]
    - Propose *k-Shape*, a partitional clustering algorithm that preserves the shapes of time series, which computes centroids effectively under the scaling and shift invariances
    - 提出一种保留时间序列形状的分区聚类算法 *k-Shape*，它在尺度不变性和位移不变性的前提下有效地计算聚类中心

- - -

## Anomaly Detection (异常检测)

Anomaly detection is the task of identifying rare items, events or observations which raise suspicions by differing significantly from the majority of the data.

#### 2019

- **A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data** (**AAAI2019**) [[paper](https://arxiv.org/pdf/1811.08055.pdf)]
    - Propose a *Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED)*, to perform anomaly detection and diagnosis in multivariate time series data
    - 提出一种多尺度卷积循环编码器-解码器（MSCRED），对多变量时间序列数据进行异常检测和诊断

#### 2018

- **Outlier Detection for Multidimensional Time Series Using Deep Neural Networks** (**MDM2018**)
    - First generates statistical features to enrich the feature space of raw time series, then utilizes an autoencoder to reconstruct the enriched time series, deviations of the enriched time series from the reconstructed time series can be taken as indicators of outliers
    - 首先生成统计特征来丰富原始时间序列的特征空间，然后利用自动编码器对丰富的时间序列进行重构，将丰富的时间序列与重构的时间序列的偏差作为判断是否异常值的指标
    - [中文解读](https://zhuanlan.zhihu.com/p/61227373)

#### 2017

- **Transfer Learning for Time Series Anomaly Detection** (**PKDD/ECML2017**) [[paper](https://pdfs.semanticscholar.org/189e/d4bac3df3068efe5a2fdd042431f848eaba6.pdf?_ga=2.170098407.67314887.1556198862-162043056.1543142349)]
   - Introduce two decision functions to guide instance-based transfer learning for time series anomaly detection. A decision function decides whether an instance from source domain should be transfered or not
   - 引入两种决策函数，指导用于时间序列异常检测的基于样本的迁移学习。决策函数决定一个源域中的样本是否应该被迁移

- - -

## Sequence Modeling (序列建模)

Specially designed sequence modeling methods can learn the representation of the input time series data, which will be helpful to solve many tasks, e.g. forecasting and classification. Considering that there is no distinct definition to classify them, this section also contains papers on representation learning in time series.

### Supervised (有监督)

#### 2018

- **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling** [[paper](https://arxiv.org/pdf/1803.01271v2.pdf)] [[code](https://paperswithcode.com/paper/an-empirical-evaluation-of-generic)] *
    - Present a *temporal convolutional network (TCN)*, which contains causal convolutions, dilated convolutions and residual connections
    - 提出时间卷积网络（TCN），其包含有因果卷积、扩张卷积以及残差连接

- **Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis** (**KDD2018**) [[paper](https://arxiv.org/pdf/1806.08946v1.pdf)] [[code](https://paperswithcode.com/paper/multilevel-wavelet-decomposition-network-for)]
    - Propose a wavelet-based neural network structure called *multilevel Wavelet Decomposition Network (mWDN)*, which preserves the advantage of multilevel discrete wavelet decomposition in frequency learning while enables the fine-tuning of all parameters under a deep neural network framework
    - 提出一种基于小波的多层次小波分解神经网络结构（mWDN），它既保留了多层次离散小波分解在频率学习中的优势，又能在深度神经网络框架下对所有参数进行微调
    - [中文解读](https://zhuanlan.zhihu.com/p/56821943) 

- **Learning Low-Dimensional Temporal Representations** (**ICML2018**) [[paper](http://proceedings.mlr.press/v80/su18a/su18a.pdf)]
    - Present a supervised dimensionality reduction method for sequence data, which learns the subspace and infers the latent alignments within it simultaneously
    - 提出一种有监督的序列数据降维方法，该方法能同时学习子空间并推断其中的潜在对齐

### Unsupervised (无监督)

#### 2019

- **Adversarial Unsupervised Representation Learning for Activity Time-Series** (**AAAI2019**) [[paper](https://arxiv.org/pdf/1811.06847.pdf)]
    - Propose an unsupervised representation learning method *activity2vec* for activity time-series, which learns representations at various levels of time granularity in the adversarial training setting
    - 针对活动时间序列提出一种无监督表示学习方法 *activity2vec*，该方法在对抗性训练设置中学习不同时间粒度的表示

- **Learning to Adaptively Scale Recurrent Neural Networks** [[paper](https://arxiv.org/pdf/1902.05696.pdf)]
    - Propose *Adaptively Scaled Recurrent Neural Networks (ASRNNs)*, a simple extension for existing RNN structures, which allows them to adaptively adjust the scale based on temporal contexts at different time steps 
    - 提出自适应尺度循环神经网络（ASRNNs），对现有 RNN 结构进行简单扩展，使其能够在处于不同时间步时根据上下文自适应调整尺度

- **Unsupervised Scalable Representation Learning for Multivariate Time Series** [[paper](https://arxiv.org/pdf/1901.10738v1.pdf)] [[code](https://paperswithcode.com/paper/unsupervised-scalable-representation-learning)]
    - Propose an unsupervised method to learn universal embeddings for variable length and multivariate time series, which combines an encoder based on causal dilated convolutions with a triplet loss employing time-based negative sampling
    - 提出一种无监督学习长度可变和多变量时间序列通用嵌入的方法，它结合了基于因果扩张卷积的编码器与基于时间的负采样的三元组损失

#### 2018

- **Learning representations for multivariate time series with missing data using Temporal Kernelized Autoencoders** [[paper](https://arxiv.org/pdf/1805.03473.pdf)]
    - Propose *Temporal Kernelized AutoEncoder (TKAE)* to learn representations of real-valued MTS with unequal lengths and missing data
    - 提出时态核化自动编码器（TKAE），用于学习具有不等长度和丢失数据的实值多变量时间序列的表示

- - -

## Query by Content (按内容查询)

Query by content focuses on retrieving a set of solutions that are most similar to a query provided by the user.

- - -

## Time Series Segmentation (时间序列分割)

Time series segmentation is a method of time-series analysis in which an input time-series is divided into a sequence of discrete segments in order to reveal the underlying properties of its source.

- - -

## Motif Discovery (重复模式发现)

Time series motifs are approximately repeating patterns in real-value data, the discovery of motifs is often the first step in various kinds of higher-level time series analytics.

- - -

## Study of Stock Market (股票市场研究)

#### 2017

- **Deep Neural-Network Based Stock Trading System Based on Evolutionary Optimized Technical Analysis Parameters** [[paper](https://www.sciencedirect.com/science/article/pii/S1877050917318252)] [[code](https://github.com/omerbsezer/SparkDeepMlpGADow30)]
    - Propose a stock trading system based on technical analysis parameters optimized by genetic algorithms, and the optimized parameters are then passed to a deep MLP neural network as features for buy-sell-hold predictions
    - 提出一种基于遗传算法优化的技术分析指标的股票交易系统，优化后的技术分析指标作为特征被传入深度 MLP 神经网络进行买入-卖出-持有预测

- - -

## Spatio-temporal Forecasting (时空预测)

### Traffic Prediction (交通预测)

#### 2018

- **Deep Sequence Learning with Auxiliary Information for Traffic Prediction** (**KDD2018**) [[paper](https://arxiv.org/pdf/1806.07380v1.pdf)] [[code](https://paperswithcode.com/paper/deep-sequence-learning-with-auxiliary)] [[video-YouTube](https://www.youtube.com/watch?v=Sw-XqR0MzhA)]
    - Integrates three kinds of implicit factors to predict traffic conditions with Seq2Seq: 1) offline geographical and social attributes 2) road intersection information 3) online crowd queries
    - 结合三种隐含因素，通过 Seq2Seq 结构预测交通状况：1）离线地理和社会属性；2）道路交叉口信息；3）在线人群查询

- - -

## Others (其他)

This section contains papers dealing with time series data or applying them to time series tasks, but not appropriate for previous topics.

#### 2019

- **Adversarial Attacks on Time Series** [[paper](https://arxiv.org/pdf/1902.10755v2.pdf)] [[code](https://github.com/houshd/TS_Adv)] *
    - Utilize an adversarial transformation network (ATN) on a distilled model to attack various time series classification models and datasets. Model distillation technique is used to solve the problem that traditional classification model is considered a black-box model with a non-differentiable internal computation
    - 利用基于蒸馏模型的对抗变换网络（ATN）攻击各种时间序列分类模型和数据集。模型蒸馏技术被用于解决传统分类模型被认为是一个内部计算不可微的黑盒模型的问题

- **Data-driven Neural Architecture Learning For Financial Time-series Forecasting** [[paper](http://arxiv.org/abs/1903.06751)]
    - Adapt *Heterogeneous Multilayer [Generalized Operational Perceptron](https://www.researchgate.net/publication/318327067_Generalized_model_of_biological_neural_networks_Progressive_operational_perceptrons) (HeMLGOP)* algorithm to progressively learn a heterogeneous neural architecture for the given financial time series forecasting problem with imbalanced data distribution problem
    - 采用异构多层[广义操作感知器](https://www.researchgate.net/publication/318327067_Generalized_model_of_biological_neural_networks_Progressive_operational_perceptrons)（HeMLGOP）算法，来逐步学习得到异构神经网络，以解决具有不平衡数据分布的给定财经时序预测问题

#### 2018

- **Recurrent Neural Networks for Multivariate Time Series with Missing Values** (**Scientific Reports 2018**) [[paper](https://www.nature.com/articles/s41598-018-24271-9.pdf)]
    - Propose *GRU-D*, which utilizes the missing patterns to achieve better prediction results by incorporating two representations of missing patterns, i.e., masking and time interval
    - 提出 *GRU-D*，其通过结合缺失的 patterns 的两种表示，即掩蔽（masking）和时间间隔（time interval），来获得更好的预测结果
    - [中文解读](https://zhuanlan.zhihu.com/p/59518293) 