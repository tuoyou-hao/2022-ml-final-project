# 2022-ml-final-project
本项代码为2022机器学习期末大作业代码
详情：[kaggle](https://www.kaggle.com/competitions/jnuml2022/overview)

由于数据集较大，需要自行下载数据集
所需数据集为：[数据集](https://www.kaggle.com/competitions/jnuml2022/data)

## 环境准备
python3，需要下载numpy、pickle、tqdm库

## 操作步骤：
  - 1、下载数据集，放在和代码同个目录下即可
  - 2、在train.py的头文件中选择要训练的模型，在`import model as ml`中修改即可，如改为`import model1 as ml`
  - 3、在train.py中自行修改超参数，包括要使用的优化器等等
  - 4、train.py中每一个epoch都会保存一次pkl，可以通过选择文件路径读取pkl进行训练
