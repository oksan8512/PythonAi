
import numpy as np

from sklearn.linear_model import LinearRegression

print("Learning Ai")

# із рядка буде стопчик
area = np.array([50,100,150,200,250]).reshape((-1,1))
price = np.array([100,150,200,300,400])

print("area: ",area)
print("price: ",price)

# add modelS
model = LinearRegression()
# передаємо дані у модель
model.fit(area,price)
# Навчання моделі y = b0+b1*x
# Нова площа житла Буде проводити прогнозування
new_area = np.array([[180]]) # Яка буде ціев для житла 180кв.
# Робиом прогнозування
prediction = model.predict(new_area)
print(f"Площа: {new_area[0][0]} m2 -> Прогноз: {prediction[0]:0f} тис грн")

print("-"*70)
