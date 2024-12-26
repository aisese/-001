import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# 读取数据
train_df = pd.read_csv('./data/train_set.csv', sep='\t',nrows=15000)
test_df = pd.read_csv('./data/test_a.csv', sep='\t',nrows=50000)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)

# 创建管道，包括TF-IDF向量化器和SVM分类器
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(2, 4), max_df=0.9, min_df=2)),
    ('svc', SVC(kernel='rbf', probability=True))  # 使用RBF核函数，并启用概率估计
])

# 定义参数网格进行网格搜索
param_grid = {
    'tfidf__max_features': [5000],
    'svc__C': [10],  # SVM的正则化参数
    'svc__gamma': [0.1]  # SVM的gamma参数，用于RBF核函数
}

# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(pipeline, param_grid, refit=True, verbose=2, cv=3, scoring='f1_macro')  # 使用宏平均F1分数作为评估指标
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# 使用最佳模型进行预测
best_pipeline = grid_search.best_estimator_
y_val_pred = best_pipeline.predict(X_val)
f1 = f1_score(y_val, y_val_pred, average='macro')
print(f'F1 Score using optimized pipeline: {f1}')

# 对测试集进行预测
test_pred = best_pipeline.predict(test_df['text'])

# 保存预测结果
predictions_df = pd.DataFrame(test_pred, columns=['label'])
predictions_df.to_csv('./data/test008.csv', index=False)