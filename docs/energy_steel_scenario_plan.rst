高占比新能源电网与钢铁产业集群负荷耦合协同场景实现指南
==========================================

本指南围绕课题“高占比新能源电网与钢铁产业集群负荷耦合协同典型场景构建”提出的三大任务，梳理 PyPOTS 现有能力、可复用的算法接口、与外部生态的集成思路，并提供初步代码骨架与参考文献，帮助团队快速搭建原型系统。

任务拆解概览
------------

#. **数据离群值检测与缺失值重构**：需要处理多源异构时间序列的对齐、异常检测、物理约束下的缺失数据重构。
#. **源荷特性图谱构建**：需提取新能源与钢铁负荷的时序特征、聚类典型模式，并构建图谱实现语义关联。
#. **高维异构数据表征与协同控制**：面向多维特征降维、非线性嵌入、灵活性评估及多智能体协同。

PyPOTS 内的能力映射
------------------

数据处理与建模的通用能力
~~~~~~~~~~~~~~~~~~~~~~

* **统一数据装载与缺失掩码生成**：`BaseDataset` 支持从内存或 HDF5 读取时间序列，自动生成缺失掩码并返回原始/预测序列，是构建对齐、重构流程的基础。该类还在初始化阶段完成数据类型检查与懒加载函数注册，方便异构数据接入。 【F:pypots/data/dataset/base.py†L1-L123】
* **模型训练与设备管理**：`BaseModel` 统一封装设备选择、AMP、日志记录与检查点策略，支持多 GPU 并行，可作为自定义异常检测或物理约束模型的基类。 【F:pypots/base.py†L1-L121】

任务 1：离群值检测与缺失值重构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **时间戳对齐**：PyPOTS 暂未直接提供动态时间规整 (DTW)，可在数据预处理阶段调用 `tslearn.metrics.dtw_path` 或 `dtaidistance` 等库对多源序列进行软对齐，再将齐次样本构造为 `BaseDataset` 可接受的字典。
* **DBSCAN 异常检测**：缺失值填充后可直接利用 `scikit-learn` 的 `DBSCAN` 对拼接后的特征窗口聚类，从低密度区域识别离群点。PyPOTS 的缺失值填充模型（如 SAITS、BRITS）可先行获得完整输入用于聚类。 【F:pypots/imputation/saits/model.py†L1-L120】【F:pypots/imputation/brits/model.py†L1-L120】
* **高斯过程缺失重构**：PyPOTS 内的 `GPVAE` 与 `CSDI` 等模型原生处理不规则与噪声序列，可作为高斯过程近似的神经实现。如需融合额定功率、爬坡率等约束，可在训练循环中继承 `BaseModel`，引入基于期望最大化的自定义损失，对违反物理边界的重构样本加罚。 【F:pypots/imputation/gpvae/model.py†L1-L120】

初步代码骨架：

.. code-block:: python

   from pypots.data.dataset import BaseDataset
   from pypots.imputation.saits import SAITS
   from sklearn.cluster import DBSCAN
   from tslearn.metrics import dtw_path
   import numpy as np

   def align_series(source_ts, target_ts):
       path, _ = dtw_path(source_ts, target_ts)
       aligned_source = np.array([source_ts[i] for i, _ in path])
       aligned_target = np.array([target_ts[j] for _, j in path])
       return aligned_source, aligned_target

   data_dict = {"X": aligned_tensor, "X_ori": original_tensor}
   dataset = BaseDataset(data_dict, return_X_ori=True, return_X_pred=False, return_y=False)

   model = SAITS(
       n_steps=dataset.n_steps,
       n_features=dataset.n_features,
       n_layers=2,
       d_model=256,
       n_heads=4,
       d_k=64,
       d_v=64,
       d_ffn=512,
       epochs=50,
       batch_size=64,
   )
   model.fit({"train": dataset})
   imputed = model.impute(dataset.X, dataset.indicating_mask)

   clustering_data = imputed.reshape(imputed.shape[0], -1)
   dbscan = DBSCAN(eps=0.5, min_samples=10)
   labels = dbscan.fit_predict(clustering_data)

   # 结合物理约束的自定义罚项可在继承 SAITS/GPVAE 的训练循环内追加

任务 2：源荷特性图谱
~~~~~~~~~~~~~~~~~~~

* **特征提取**：PyPOTS 的 `representation` 子包提供 TS2Vec，可在不完整数据上学习多尺度表示，为后续聚类与图谱输入提供低维嵌入。 【F:pypots/representation/ts2vec/model.py†L1-L88】
* **时序建模**：`forecasting` 子包内含 TimesNet、Autoformer、Transformer 等深度模型，可替换双层 LSTM，并允许在输入侧合并气象特征。训练完毕后的隐状态或注意力分布可作为“源荷特征向量”。
* **聚类识别典型模式**：使用 `scikit-learn` 的 `KMeans` 对特征向量聚类，聚类标签与对应时间窗构成场景集合。
* **知识图谱实现**：PyPOTS 未内置图数据库，可组合 `Neo4j` 或 `NebulaGraph` 与 `py2neo`、`nebula3-python` 编程接口。嵌入模型可使用 `PyKEEN` 的 `TransE`，将实体（新能源场站、钢铁负荷单元、气象事件）与关系写入图库，实现语义检索。

任务 3：高维异构表征与协同控制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **降维与非线性嵌入**：在传统主成分分析 (PCA) 之后，可调用 PyPOTS 的 TS2Vec 或 `representation` 子包内的其它模型生成低维张量。外部库如 `scikit-learn` 的 `KernelPCA` 或 `umap-learn` 可以进一步增强非线性表达。
* **灵活性资源量化**：将物理边界作为奖励函数或约束，利用 PyPOTS 模型输出的缺失填补与预测结果构造状态空间。强化学习框架可选 `Ray RLlib`、`MARLlib`、`PettingZoo` 来搭建多智能体环境；状态输入可直接引用 PyPOTS 输出的张量表示。
* **深度耦合自律协同**：可通过 PyPOTS 的 `BaseModel` 自定义强化学习训练循环，或将 PyPOTS 产出的数据接口接入外部 MARL 库训练策略网络，实现源荷协同控制。

推荐的外部开源库
----------------

* **时间对齐**：`tslearn` (https://github.com/tslearn-team/tslearn)
* **异常检测/聚类**：`scikit-learn` (DBSCAN、KMeans)、`pyod` (可选)
* **知识图谱**：`Neo4j`/`NebulaGraph`，嵌入使用 `PyKEEN`
* **强化学习**：`Ray RLlib`、`MARLlib`、`PettingZoo`

后续实验建议
------------

1. 以 PyPOTS SAITS/BRITS/GPVAE 为基线，验证不同缺失率下的重构误差，并评估 DBSCAN 对异常点的召回率。
2. 使用 TS2Vec 表示与 TimesNet 预测结果组合，构建源荷特征向量，评估 KMeans 聚类后场景的稳定性和可解释性。
3. 在强化学习原型中引入物理约束惩罚项，对比不同奖励设计下的灵活性资源调度效率。

参考文献
--------

* Du, W., et al. "SAITS: A Self-Attention-based Time Series Imputation Model." AAAI 2023.
* Li, Y., et al. "CSDI: Conditional Score-based Diffusion Models for Imputation of Missing Values in Multivariate Time Series." NeurIPS 2021.
* Yue, X., et al. "TS2Vec: Towards Universal Representation of Time Series." AAAI 2022.
* Vaswani, A., et al. "Attention is All you Need." NeurIPS 2017.
* Sutton, R. S., & Barto, A. G. "Reinforcement Learning: An Introduction." MIT Press, 2018.

