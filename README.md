# 📗 ProuDT: Revisiting Simplicity 🎉

_Revisiting Simplicity: An Accurate Probabilistic Univariate Decision Tree_

## 🔥 Overview
ProuDT is a novel univariate probabilistic decision tree that is easy to deploy. It achieves **SOTA accuracy** while maintaining **interpretability** by attributing individual features to decision-making. The ProuDT strategy is proposed in the paper ___Revisiting Simplicity: An Accurate Probabilistic Univariate Decision Tree___.

## 🚀 Get Started

- For experiment reproducibility, please see the /experiments folder.
- The implementation of ProuDT is straightforward for classification tasks. The following is an example.

### Prepare your data
```python
from proudt import ProuDT
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Normalize data 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### Create and train model

```python
# model = ProuDT(random_state=42) # default depth is 8 if no specified depth
model = ProuDT(depth=3,random_state=42)
model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
```
### Make predictions

```python
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

# Evaluate model
metrics = model.evaluate(X_test_scaled, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```
## Important Notes

- Always normalize your data before using ProuDT
- Provide both training and validation data for optimal results
- Deeper trees may be needed for complex datasets