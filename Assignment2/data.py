import os

def read_file(name, cur, train_data, test_data, test_label):
	X = []
	label = []
	with open(name) as f:
		data = f.read().splitlines()
		for line in data:
			x, y = line.split()
			X.append([float(x), float(y) ] )
			label.append(cur)
		
		train_data += X[:int(0.75*len(X))]
		test_data += X[int(0.75*len(X)):]
		test_label += label[int(0.75*len(label)):]
		
def get_data(filename):
	train_data = []
	test_data = []
	test_label = []
	files = os.listdir(filename)
	cur = 0
	for i in range(len(files)):
		read_file(filename+files[i], cur, train_data, test_data, test_label)
		cur += 1

	return train_data, test_data, test_label


# train, test, test_l = get_data("./Data1/")
# print(len(train),len(test),len(test_l),len(train[0]))
# print(test)
# print(test_l)




