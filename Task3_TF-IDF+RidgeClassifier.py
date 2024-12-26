import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# 读取训练数据
train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=5000)

# 划分训练集和验证集
train_size = 4000
val_size = train_df.shape[0] - train_size

train_texts = train_df['text'].values[:train_size]
val_texts = train_df['text'].values[train_size:]
train_labels = train_df['label'].values[:train_size]
val_labels = train_df['label'].values[train_size:]

# 创建管道
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RidgeClassifier(random_state=42))  # 设置随机状态以确保结果可重复
])

# 定义参数网格
param_grid = {
    'tfidf__ngram_range': [(2, 3), (2, 4)],
    'tfidf__max_df': [0.7, 0.8],
    'tfidf__min_df': [1, 2, 5],
    'tfidf__max_features': [4000, 5000, 6000],
    'clf__alpha': [0.1, 1.0, 10.0]
}

# 使用GridSearchCV进行网格搜索
grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=3, n_jobs=1)
grid_search.fit(train_texts, train_labels)

# 输出最佳参数和得分
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation F1 Score (Macro): {grid_search.best_score_}")

# 读取测试数据并使用最佳模型进行预测
test_df = pd.read_csv('./data/test_a.csv', sep='\t', nrows=500)
test_preds = grid_search.best_estimator_.predict(test_df['text'])

# 将预测结果保存到DataFrame中
predictions_df = pd.DataFrame(test_preds, columns=['label'])

# 将预测结果DataFrame保存到CSV文件中
predictions_df.to_csv('./data/test_009.csv', index=False)