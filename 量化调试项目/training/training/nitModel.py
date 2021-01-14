#encoding:utf-8
import xgboost as xgb

def get_xgb_model(xgb_train, xgb_val, num_class):
    params={
    'booster':'gbtree',
    'objective': 'multi:softprob', #多分类的问题 multi:softprob  multi:softmax
    'num_class': num_class, # 类别数，与 multisoftmax 并用
    'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':12, # 构建树的深度，越大越容易过拟合
    'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample':0.7, # 随机采样训练样本
    'colsample_bytree':0.7, # 生成树时进行的列采样
    'min_child_weight':1,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.001, # 如同学习率
    'seed':1000,
    'nthread':16,# cpu 线程数
    'eval_metric': 'merror'
    }

    plst = list(params.items())
    num_rounds = 20000 # 迭代次数
    watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

    #训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)

    return model

# 分类别计算预测准确率
def class_accuracy_score(y_test, y_pred, num_class):
	pred_accuracy = []
	pred_num = []
	true_num = []
	for i in range(num_class):
		pred_accuracy.append(0)
		pred_num.append(0)
		true_num.append(0)

	for i in range(len(y_test)):
		true_num[y_test[i]] += 1
		pred_num[y_pred[i]] += 1
		if y_pred[i]==y_test[i]:
			pred_accuracy[y_pred[i]] += 1

	for i in range(num_class):
		if pred_num[i]==0:
			pred_accuracy[i] = 0
		else:
			pred_accuracy[i] = str(round(pred_accuracy[i]*100.0/pred_num[i],2)) + "%"
	return true_num, pred_num, pred_accuracy

def lable_Y(Y, num_class):
	if num_class==2:
		for i in range(len(Y)):
			if Y[i]>1:
				Y[i]=1
			else:
				Y[i]=0
	if num_class==3:
		for i in range(len(Y)):
			if Y[i]>=1:
				Y[i]=2
			elif Y[i]>-1 and Y[i]<1:
				Y[i]=1
			elif Y[i]<=-1:
				Y[i]=0