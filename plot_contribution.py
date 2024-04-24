import matplotlib.pyplot as plt
optimal_policy = [0.0, 0.4, 0.5333333333333333, 0.6, 0.64, 0.6666666666666666, 0.6857142857142857, 0.7, 0.7111111111111111, 0.72, 0.7272727272727273, 0.7333333333333333, 0.7384615384615385, 0.7428571428571429, 0.7466666666666667, 0.75, 0.7529411764705882, 0.7555555555555555, 0.7578421052631579, 0.7599]
three_states = [0.0, 0.261, 0.30266666666666664, 0.33825, 0.363, 0.3993333333333333, 0.43457142857142855, 0.46875, 0.49922222222222223, 0.5236, 0.5451818181818182, 0.5634166666666667, 0.5796923076923077, 0.5932857142857143, 0.6052, 0.6158125, 0.6244705882352941, 0.6326666666666667, 0.6401052631578947, 0.6464]

x_axis = [i for i in range(1000,20001,1000)]

#plt.plot(x_axis, optimal_policy, label = "Optimal policy start", marker='.',color='b')
plt.plot(x_axis, three_states, label = "Recommendations for three states", marker='.',color='g') 
for i in range(0,len(x_axis)-3,2):
    plt.text(x_axis[i], three_states[i], f'({x_axis[i]//1000}k, {round(three_states[i],2)})', fontsize=8, verticalalignment='top')
plt.text(x_axis[-3], three_states[-3], f'({x_axis[-3]//1000}k, {round(three_states[-3],2)})', fontsize=8, verticalalignment='bottom')
plt.text(x_axis[-1], three_states[-1], f'({x_axis[-1]//1000}k, {round(three_states[-1],2)})', fontsize=8, verticalalignment='bottom')
plt.xlabel("Number of interactions with the environment")
plt.ylabel("Percentage of contribution of kick-starting loss")
plt.legend() 
#plt.grid(True)
plt.show()