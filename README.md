# FinancialTrendForecast


## Requirements
```
python3.5
xgboost
pandas
```

# Getting Started
## Tourials
1. move the data to the data dir
2. cd to the root dir
3. run train.py to get the model file
4. run test.py to get the prediction result  


## Train
```
python3 train.py
```

## Test
```
python3 test.py
```



# 方法介绍
> 根据给出的前几个季度的一些特征指标，预测下一个季度的盈利趋势(上升、下降)（浮点数几乎可以不用考虑平的情况）

>思路方法: 划窗法构建特征

>主要实现: 挖掘相关特征+xgboost预测
步骤:
1. 挖掘特征(挖掘尽量多的特征)
2. 特征选择(挖掘的特征中可能会包含无效特征，已经反向特征，drop掉无效特征，drop方法为xgboost特征重要性排序较低的删除掉)
3. 得到模型(训练数据方面用了2012～2017的数据，2018用来预测)
4. 预测

主要特征包括:
1. 待预测节点的报表类型_Reporttype   1个特征
2. 倒数第一个节点与倒数第二个节点各指标的的差值(及带预测节点之前的最后两个的差值)(多个指标特征的位置如特征后面所述的顺序)  6个特征
3. 各指标近2年的均值  6个特征
4. 各指标近1年的均值  6个特征
5. 近1年的各季度相对于1年前各季度每个指标的增益情况  4*6=24个特征
6. 近2年各指标为正的次数  6个特征
7. 近2年各指标为负的次数  6个特征
8. 各指标近1年的均值和各指标近2年均值的差值   6个特征
9. 去年该季度是否上升    1个特征
10. 前年该季度是否上升   1个特征


初始特征共63,特征选择drop部分无效特征[46, 59, 24,28,61,4,32]后,剩56各特征，最终模型预测结果



>多个指标在提特征时候的顺序
1. 每股净资产(元/股)_NAPS
2. 净资产收益率(摊薄)(%)_ROE
3. 资产报酬率(%)_ROAEBIT
4. 投入资本回报率(%)_ROIC
5. 成本费用利润率(%)_Totprfcostrt
6. 营业利润率(%)_Opeprfrt

精度记录
| Q1 | Q2 | Q3 | Q4 |
|:-:|:-:|:-:|:-:|
| 85 % | 90 % | 80% | 75% |
