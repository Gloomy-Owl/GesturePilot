import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Загружаем данные...")
df = pd.read_csv('data/gestures_dataset.csv')

# Учитываем все 8 жестов
df = df[df['label'].isin([1, 2, 3, 4, 5, 6, 7, 8])]

print("\nКоличество примеров:")
for i in range(1, 9):
    print(f"Жест {i}: {len(df[df['label'] == i])}")

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nОбучаем Нейросеть (MLP)...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Точность Нейросети: {accuracy_score(y_test, predictions) * 100:.2f}%")

with open('models/gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Мозг обновлен!")