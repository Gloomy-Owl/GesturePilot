import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Загружаем данные...")
df = pd.read_csv('gestures_dataset.csv')

# оставляем только нужные жесты (1 - Ладонь, 2 - Кулак)
df = df[df['label'].isin([1, 2])]

print("\nКоличество примеров для каждого жеста:")
print("Ладонь (1):", len(df[df['label'] == 1]))
print("Кулак (2):", len(df[df['label'] == 2]))

# разделяем таблицу: х - это 63 координаты, y - это метка жеста (1 или 2)
X = df.drop('label', axis=1)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# обучение
print("\nСоздаем и обучаем Нейронную сеть...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)

model.fit(X_train, y_train)

# проверка
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nТочность Нейросети: {accuracy * 100:.2f}%")

# сохранение
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Нейросеть успешно сохранена в файл 'gesture_model.pkl'!")