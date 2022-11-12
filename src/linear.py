import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# fig=plt.figure()
# ax=Axes3D(fig)
#
# #输入的数据
# x=[1,2,3,4,5,6,7]
# y=[4,7,10,13,16,19,22]
#
# #θ0，θ1的取值范围和精度
# parameter0=np.arange(-10,10,0.2)
# parameter1=np.arange(-10,10,0.2)
# def func_j(p0,p1):
#     sum=0
#     for i in range(0,7):
#         h=p0+p1*x[i]
#         sum+=(h-y[i])**2
#     sum=sum/14
#     return sum
#
# parameter0,parameter1=np.meshgrid(parameter0,parameter1)
# z=func_j(parameter0,parameter1)
# surf=ax.plot_surface(parameter0,parameter1,z)
#
# min_value=np.min(z)
# min_index=np.argmin(z)
# print (np.unravel_index(min_index,z.shape))
#
# min_point=np.unravel_index(min_index,z.shape)
# min_x=min_point[0]
# min_y=min_point[1]
#
# print (parameter0[min_x][min_y])
# print (parameter1[min_x][min_y])
#
# plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']],bmi_life_data[['Life expectancy']])
laos_life_exp = bmi_life_model.predict([ [21.07931]])
print(laos_life_exp)
ax = plt.scatter(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])
plt.title('bmi_and_life_expectancy(LinearRegression)')
plt.xlabel("BMI")
plt.ylabel("Life expectancy")
print("coef_:",bmi_life_model.coef_)
print("intercept_:",bmi_life_model.intercept_)
x = np.linspace(18, 30, 1000)
y = bmi_life_model.coef_[0][0] *x+ bmi_life_model.intercept_[0]
A = [0,bmi_life_model.intercept_[0]]
B = [-bmi_life_model.intercept_[0]/bmi_life_model.coef_[0][0],0]
plt.plot(x,y,c='r')
plt.show()
# from sklearn import linear_model
# reg = linear_model.Ridge (alpha = .505)
# reg.fit(bmi_life_data[['BMI']],bmi_life_data[['Life expectancy']])
# print(reg.predict([ [21.07931]]))
# print("coef_:",reg.coef_)
# print("intercept_:",reg.intercept_)
#
# ax = plt.scatter(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])
# plt.title('bmi_and_life_expectancy(Ridge Regression)')
# plt.xlabel("BMI")
# plt.ylabel("Life expectancy-")
# print("coef_:",reg.coef_)
# print("intercept_:",reg.intercept_)
# x = np.linspace(18, 30, 1000)
# y = reg.coef_[0][0] *x+ reg.intercept_[0]
# A = [0,reg.intercept_[0]]
# B = [-reg.intercept_[0]/reg.coef_[0][0],0]
# plt.plot(x,y,c='r')
