import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=50000)
test_df = pd.read_csv('./data/test_a.csv', sep='\t', nrows=50000)

# 定义要尝试的TF-IDF参数组合
param_combinations = [
    # {'max_features': 5000, 'ngram_range': (1, 3), 'max_df': 0.80, 'min_df': 10},
    # {'max_features': 5000, 'ngram_range': (2, 4), 'max_df': 0.75, 'min_df': 15},
    # {'max_features': 5000, 'ngram_range': (1, 4), 'max_df': 0.75, 'min_df': 20},
    # {'max_features': 5000, 'ngram_range': (1, 4), 'max_df': 0.70, 'min_df': 25},
    # {'max_features': 6000, 'ngram_range': (2, 4), 'max_df': 0.75, 'min_df': 30},
    # {'max_features': 6000, 'ngram_range': (1, 4), 'max_df': 0.70, 'min_df': 40},
    # {'max_features': 6000, 'ngram_range': (2, 4), 'max_df': 0.75, 'min_df': 30},
    # {'max_features': 6000, 'ngram_range': (1, 4), 'max_df': 0.70, 'min_df': 30},
    # {'max_features': 6000, 'ngram_range': (1, 4), 'max_df': 0.70, 'min_df': 35},
    {'max_features': 6000, 'ngram_range': (1, 4), 'max_df': 0.65, 'min_df': 30},
    {'max_features': 6000, 'ngram_range': (1, 4), 'max_df': 0.60, 'min_df': 30},
    {'max_features': 6000, 'ngram_range': (1, 4), 'max_df': 0.50, 'min_df': 30},
]
best_f1 = 0
best_params = None
best_model = None

# 遍历每种参数组合
for params in param_combinations:
    vectorizer = TfidfVectorizer(**params)

    # 划分训练集和验证集（使用0.2的比例划分验证集）
    X_train, X_val, y_train, y_val = train_test_split(train_df['text'],
      train_df['label'], test_size=0.2,random_state=42)

    # 训练逻辑回归模型
    clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    clf.fit(vectorizer.fit_transform(X_train), y_train)

    # 验证模型
    val_pred = clf.predict(vectorizer.transform(X_val))
    f1 = f1_score(y_val, val_pred, average='macro')

    # 记录最佳F1分数和对应的参数及模型
    if f1 > best_f1:
        best_f1 = f1
        best_params = params
        best_model = clf
        best_vectorizer = vectorizer  # 保存最佳向量化器以便后续使用

print(f"Best F1 Score: {best_f1}")
print(f"Best Parameters: {best_params}")

# 使用最佳模型进行测试集预测
test_pred = best_model.predict(best_vectorizer.transform(test_df['text']))

# 保存预测结果
predictions_df = pd.DataFrame(test_pred, columns=['label'])
predictions_df.to_csv('./data/test011.csv', index=False)