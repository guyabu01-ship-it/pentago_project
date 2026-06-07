import keras
import numpy as np
from sklearn.model_selection import train_test_split
import json


def load_training_data(filename='processed_data.json'):
    print(f"Loading {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['X'], dtype='float32'), np.array(data['Y'], dtype='float32').reshape(-1, 1)


# 1. טעינה והכנה
X_list, y = load_training_data()
X_flat = X_list.reshape(-1, 36)  # שיטוח ל-36 כניסות

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# 2. בניית מודל ANN "חכם" יותר
model = keras.models.Sequential([
    keras.layers.Input(shape=(36,)),

    # שימוש ב-Tanh בשכבה הראשונה כדי "להרגיש" את הקוטביות של הלוח
    keras.layers.Dense(256, activation='leaky_relu'),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(256, activation='leaky_relu'),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(256, activation='leaky_relu'),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(128, activation='leaky_relu'),
    keras.layers.Dropout(0.2),


    keras.layers.Dense(64, activation='leaky_relu'),
    keras.layers.Dense(1, activation='tanh')
])
# 3. הגדרת אופטימייזר עם קצב למידה מותאם
optimizer = keras.optimizers.Adam(learning_rate=0.00001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

# 4. מנגנוני בקרה (Callbacks)
# יעצור את האימון אם אין שיפור במשך 5 אפוקים
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# יוריד את קצב הלמידה אם המודל נתקע ב"מישור"
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3
)

# 5. אימון עם הרבה אפוקים
print("Starting optimized training...")
history = model.fit(
    X_train, y_train,
    epochs=50,  # הגדרנו 100, אבל ה-EarlyStopping כנראה יעצור קודם
    batch_size=2048,  # Batch גדול למהירות ויציבות (5.8 מיליון דגימות!)
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

import matplotlib.pyplot as plt

# --- גרף 1: Loss ו-MSE (ביחד כפי שהגדרת ב-subplot) ---
plt.figure(figsize=(12, 5))

# תת-גרף ל-Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.legend()

# תת-גרף ל-MSE
plt.subplot(1, 2, 2)
plt.plot(history.history['mse'], label='Train MSE')
plt.plot(history.history['val_mse'], label='Val MSE')
plt.title("Model MSE")
plt.legend()

# שמירת התמונה המשותפת
plt.savefig('loss_and_mse_comparison_ann.png', dpi=300) # dpi=300 מבטיח איכות גבוהה להדפסה בספר

# --- גרף 2: MAE בנפרד ---
plt.figure(figsize=(6, 5)) # יצירת דף חדש לגרף הבא
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("Model MAE")
plt.legend()

# שמירת הגרף השני
plt.savefig('model_mae_results_ann.png', dpi=300)

# 6. שמירת המודל - חשוב כדי שלא תצטרך לאמן שוב!
model.save('pentago_ann_model.keras')
print("Model saved as pentago_ann_model.keras")

# 7. הערכה סופית
results = model.evaluate(X_test, y_test)
print(f"Final Test MAE: {results}")