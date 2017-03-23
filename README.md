# AncientChinesePoemRNN

参照[char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow)，使用RNN的字符模型，学习并生成古诗。
数据来自于http://www16.zzu.edu.cn/qts/ ,总共4万多首唐诗。

## 准备环境

* tensorflow 1.0
* python2

## 训练
`python train.py`
在使用GPU的情况下，两个小时内即可完成训练

## 生成

* `python sample.py`
rnn神经网络会生成一首全新的古诗。例如：
**”帝以诚求备，堪留百勇杯。教官日与失，共恨五毛宣。鸡唇春疏叶，空衣滴舞衣。丑夫归晚里，此地几何人。”**
* `python sample.py --prime <这里输入指定汉字>`
rnn神经网络会利用输入的汉字生成一首藏头诗。例如：
  `python sample.py --prime 如花似月` 会得到
  **“如尔残回号，花枝误晚声。似君星度上，月满二秋寒。”**
