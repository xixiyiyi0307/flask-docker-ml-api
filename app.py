from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Training data
#Question 1
#a)
X = np.array([
    [0, 19.8], [1, 23.4], [1, 27.7], [1, 24.6], [0, 21.5],
    [1, 25.1], [1, 22.4], [0, 29.3], [0, 20.8], [1, 20.2],
    [1, 27.3], [0, 24.5], [0, 22.9], [1, 18.4], [0, 24.2],
    [1, 21.0], [0, 25.9], [0, 23.2], [1, 21.6], [1, 22.8]
])
y = np.array([137, 118, 124, 124, 120, 129, 122, 142, 128, 114,
              132, 130, 130, 112, 132, 117, 134, 132, 121, 128])
model = LinearRegression()
#b)
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred
n, k = X.shape
df = n - (k + 1)  # 自由度：n - (自变量个数 + intercept)
sigma_squared = np.sum(residuals**2) / df
X_design = np.hstack((np.ones((n, 1)), X))
XtX_inv = np.linalg.inv(X_design.T @ X_design)
var_b = sigma_squared * XtX_inv
se_b = np.sqrt(np.diag(var_b))
coef = np.insert(model.coef_, 0, model.intercept_)
t_stats = coef / se_b
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
tau_hat = coef[1]
tau_p_value = p_values[1]
print(f"Estimated ATE (taû): {tau_hat:.4f}")
print(f"Standard error: {se_b[1]:.4f}")
print(f"t-statistic: {t_stats[1]:.4f}")
print(f"p-value: {tau_p_value:.4f}")

@app.route("/predict")
def predict():
    w = float(request.args.get("w", 0))   
    x = float(request.args.get("x", 0))   

    y_pred = model.predict([[w, x]])[0]

    # Log prediction
    with open("output.txt", "w") as f:
        f.write(f"Input: w={w}, x={x}\nPrediction: {y_pred}\n")

    return jsonify({"w": w, "x": x, "prediction": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

#docker build -t my-api .
#docker run -p 5000:5000 my-api
#curl "http://localhost:5000/predict?w=1&x=20"

