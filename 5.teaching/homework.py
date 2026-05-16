import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

housing = fetch_california_housing()
X_all   = housing.data
y       = housing.target

feature_names = list(housing.feature_names)

# 1 

print(" 1 ")
print("")

X_medinc = X_all[:, feature_names.index("MedInc")].reshape(-1, 1)

poly2    = PolynomialFeatures(degree=2, include_bias=False)
X_poly2  = poly2.fit_transform(X_medinc)

X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
    X_poly2, y, test_size=0.2, random_state=42
)

model1 = LinearRegression().fit(X_tr1, y_tr1)
r2_t1  = r2_score(y_te1, model1.predict(X_te1))

print(f"Ознаки після PolynomialFeatures : {X_poly2.shape[1]}")
print(f"R² на тестовій вибірці         : {r2_t1:.4f}\n")

# 2
print("")
print("2")
print("")
print(f"{'Ступінь':<12} {'R² train':<12} {'R² test':<12}")
print("-" * 36)

for deg in [1, 2, 3]:
    pf   = PolynomialFeatures(degree=deg, include_bias=False)
    X_p  = pf.fit_transform(X_medinc)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_p, y, test_size=0.2, random_state=42
    )
    m     = LinearRegression().fit(X_tr, y_tr)
    r2_tr = r2_score(y_tr, m.predict(X_tr))
    r2_te = r2_score(y_te, m.predict(X_te))
    print(f"{deg:<12} {r2_tr:<12.4f} {r2_te:<12.4f}")

print()

feat_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
feat_idx  = [feature_names.index(f) for f in feat_cols]
X_feat    = X_all[:, feat_idx]

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

X_tr4, X_te4, y_tr4, y_te4 = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 3
print("" )
print(" 3 ")
print("")

lr3    = LinearRegression().fit(X_tr4, y_tr4)
lasso3 = Lasso(alpha=0.1, max_iter=10_000).fit(X_tr4, y_tr4)

r2_lr3    = r2_score(y_te4, lr3.predict(X_te4))
r2_lasso3 = r2_score(y_te4, lasso3.predict(X_te4))
zeros     = int(np.sum(np.abs(lasso3.coef_) < 1e-10))

print(f"\nR² тест — LinearRegression : {r2_lr3:.4f}")
print(f"R² тест — Lasso(α=0.1)    : {r2_lasso3:.4f}")
print(f"\nКоефіцієнтів = 0 у Lasso  : {zeros} з {len(feat_cols)}")
print(f"\n{'Ознака':<12} {'LR коеф.':<14} {'Lasso коеф.':<14} {'Обнулено?'}")
print("-" * 52)
for f, c_lr, c_las in zip(feat_cols, lr3.coef_, lasso3.coef_):
    zero_flag = "← 0" if abs(c_las) < 1e-10 else ""
    print(f"{f:<12} {c_lr:<14.4f} {c_las:<14.4f} {zero_flag}")

print()

# 4
print("")
print(" 4 ")
print("")

ridge4 = Ridge(alpha=1.0).fit(X_tr4, y_tr4)

r2_lr4    = r2_score(y_te4, lr3.predict(X_te4))
r2_ridge4 = r2_score(y_te4, ridge4.predict(X_te4))

lr_nonzero    = int(np.sum(np.abs(lr3.coef_) > 1e-10))
ridge_nonzero = int(np.sum(np.abs(ridge4.coef_) > 1e-10))

print(f"\n{'Модель':<22} {'R² test':<12} {'Ненульових коеф.'}")
print("-" * 50)
print(f"{'LinearRegression':<22} {r2_lr4:<12.4f} {lr_nonzero} з {len(feat_cols)}")
print(f"{'Ridge(alpha=1.0)':<22} {r2_ridge4:<12.4f} {ridge_nonzero} з {len(feat_cols)}")

print(f"\n{'Ознака':<12} {'LR коеф.':<14} {'Ridge коеф.':<14} {'Різниця'}")
print("-" * 52)
for f, c_lr, c_rid in zip(feat_cols, lr3.coef_, ridge4.coef_):
    diff = c_rid - c_lr
    print(f"{f:<12} {c_lr:<14.4f} {c_rid:<14.4f} {diff:+.6f}")

