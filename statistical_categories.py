import numpy as np

data = np.loadtxt('test_k_7_image_result.csv')

class_group = []
for i in range(30):
	num = 0
	for j in data[:, i]:
		if j == np.max(data[i, :]):
			num += 1
	class_group.append(num)
print(class_group)
print(np.sum(class_group))