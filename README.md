# autoflow

## 1. 项目结构

1. utils - 工具集

IP转化城市；GEO转城市；Hbase入库；DES加解密；其他工具函数。

2. samples - 抽样

一个通用的标签抽样工具。

3. train - 训练
   
    1. conf - 配置文件\
    目前有applist.txt、vocab.txt以及userprofile.json。
    2. estimators \
    常用的模型，融合为了Spark的Estimator模型。\
    目前有：pytorch的本地化训练模型，light_gbm模型。
    3. transformers/converter \
    可将applist、vocab以及userprofile数据转为向量。
    4. data \
    可将spark各个iter数据封装为pytorch所需的dataset形式。
    5. evaluation \
    本地化训练后的评价。
    6. samples \
    抽样
    7. process \
    训练和预测流程

4. works - 示例


