pip install h5py
pip install sklearn
pip install pandas
pip install scipy
pip install easydict

重要提示：需求方提供的需求不详细
我们的模型是预测2017/6/30号后出现的购买记录的用户，在2017/6/30之前有过一次首笔购买记录，并且两次行为记录的时间差在45天内。
据统计：商品数量共405个，用户数量为50000人

您需要：安装python，sklearn，pandas，easydict等依赖包
工程的目录
./data/action_data.csv 是原始数据文件
./config.py  模型和程序的一些配置信息
./gen_feat.py 特征生成文件
./main.py 主函数
./utils.py 常用函数文件,该程序中貌似没有用到，为后期维护扩展做准备

第一步:
	将原始数据添加到data目录下，并修改文件名字为action_data.csv

第二步:
	执行gen_feat.py文件中的split_train_test()函数划分训练集和测试集，将在./cache/目录下生成train.csv和test.csv文件

第三步: 
	执行main.py 大约20s后程序运行出最终结果，结果保存在./cache/目录下的ans.csv文件里user_id列即为预测为未来45天内可能购买商品的用户名称


模型设计思路
# groupby CustomerID 前操作
1、去除掉退货的数据: Trans = -1的列
2、删除: | OrderDate - FirstDate | > 45天的记录 噪声
3、标注: | OrderDate - FirstDate | == 0天 时的label=0
4、标注: 1 <= | OrderDate - FirstDate | <= 45天 时的label=1
5、根据Birthday构造用户age列
6、ProductCategory列 one-hot 处理
7、Product编码转换成 int值
8、将第一次购买在2017/5/20之前的用户切分给训练集
9、将第一次购买在2017/5/20之后的用户切分给测试集
注: 
1、训练集和测试集都做根据CustomerID的groupby操作
Product按道理来说得用one-hot处理(405维对于机器学习来说维度很大)，考虑到内存与需求方价格方面的原因，没必要拉成one-hot并使用深度模型在GPU上训练

# groupby CustomerID 操作
统计lable=1的用户,只保留label=1的用户的首笔购买记录
5月20号后的特征: 直接删除label=1的样本

# groupby CustomerID 后操作
训练集做训练与验证
最终预测结果在./cache/文件夹下的ans.csv里


最终样本特征为(首笔购买的)： 用户年龄、Product、Items、UnitPrice、ProductCategory(one-hot中表示)






















