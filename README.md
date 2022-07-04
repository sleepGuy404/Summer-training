# 四川省高校暑期实训挑战第11名方案

## 数据集处理

通过查看数据集，发现法控时间（timecc）、五杀（pentakills）这两种数据，其中法控时间均为0，五杀数据几乎没有大于0的，因此可以删掉这两个数据字段。

在游戏中，会使用KDA来评价游戏玩家本局的水平，因此添加了新的字段KDA，KDA=(K+A)/D（如果D为0，则除以1）。此外，视野也是影响游戏胜利的一部分。wardsplaced（侦查守卫放置次数）和wardskilled（侦查守卫摧毁次数）都是属于视野方面的一部分，因此可以新加一个关于视野的数据字段wardsgrade（事业得分），wardsgrade=wardsplaced+wardskilled

## 模型训练

在训练模型前，对参数做一些测试后，最后还是选择了学习率为0.01的。

在训练模型前，考虑到数据集分为三种：训练集，验证集，测试集。因此，18万数据中，将最后2万的数据作为验证集。

```python
# 从第0行到倒数20000行（不包括），设置为training_data
training_data = train_df.iloc[:-20000,].values.astype(np.float32)
# 从倒数20000行到最后一行设置为val_data
val_data = train_df.iloc[-20000:, ].values.astype(np.float32)
```

去掉了训练轮次，改为了验证集用于验证。即：在一个无尽循环中，每训练完成一次，就使用验证集验证一次，如果此次的准确率大于0.85，且训练轮次大于200，该训练结束。

```python
if correct/len(a)>0.85 and epoch_number>200:
	print("final: ",correct/len(a),' epoch_number: ',epoch_number,' max: ',max)
	break
```

由于模型一直训练，两个要求都没有达到，所以在1150轮前时候停止，此时最佳的效果为0.8467，然后使用测试集测试，得出了最高的分数83.93。

![最高分](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/%E6%9C%80%E9%AB%98%E5%88%86.png)

## 一些失败的尝试

在数据集处理时，发现此处的处理有些问题，例如，如果在训练集中kills（击杀次数）最大为60，在测试集中的最大kills为30，在训练集中有一个id为A的玩家击杀次数为30，在测试机中的一个id为B的玩家击杀次数为15。通过数据可以直观地看出二者的差距，但是在经过以下代码处理后，`train_df['kills']['A']`和`test_df['kills']['B']`的值更新后都是0.5。因此以下的代码并不能准确地体现不同的数据集上的玩家的真实水平。

```python
# 将每一列的数据全部转化为比例
for col in train_df.columns[1:]:
    train_df[col] /= train_df[col].max()
    test_df[col] /= test_df[col].max()
print(train_df)
```

此外，还考虑到异常值的处理，如果在训练集中kills的值普遍在50以下，但是有一个异常值为80，那么上述代码则会减小不同数据的差距，例如两个玩家的kills为10和20，算出的比例为1/8和1/4，而如果选用50作为除数，则比例为1/5和2/5，二者差异更大。如何选择合理的除数，是一个关键问题。

![下载](https://cyan-1305222096.cos.ap-nanjing.myqcloud.com/%E4%B8%8B%E8%BD%BD.png)

在代码运行时，注意到了箱型图的结构，有上下四分位数和上下边缘，其中的上边缘正好适合需求，于是，通过函数构造，获得了每列的上边缘值。

```python
#计算上下四分位数
def count_quartiles(lis):
    q1 = lis[int(1 + (float(len(lis)) - 1) * 1 / 4)]
    q3 = lis[int(1 + (float(len(lis)) - 1) * 3 / 4)]
    return q1, q3

#计算上下边缘
def count_margin(q1, q3):
    q4 = q3 + 1.5 * (q3 - q1)
    q5 = q1 - 1.5 * (q3 - q1)
    return q4, q5
# 获取列名
def upper_limit(df):
    columns = df.columns.values.tolist()  #列名
    print(columns)
    upper_limit_values = []
    for column in columns:
        sort_df = df.sort_values(by=column)
        q1,q3 = count_quartiles(list(sort_df[column]))
        column_upper=count_margin(q1,q3)[0]
        upper_limit_values.append(column_upper)
        print(upper_limit_values)
    return upper_limit_values

upper_limit_values_train_df=upper_limit(train_df)
del(upper_limit_values_train_df[0])
upper_limit_values_test_df=upper_limit(test_df)
```

对于训练数据集，需要删除win字段，而测试集则不需要。

以上二者的数据处理，在模型训练完成，提交结果后的成绩并不理想，因此以上的尝试也以失败告终。