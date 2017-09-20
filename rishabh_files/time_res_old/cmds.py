# TIME

# states = 9, 21, 22

# 1. hotCold
# 	a. Sklearn
# 	python -m himanshu.nn.sklearn_hotcold 
# 		-state --include_0 --rand_forest
# 		3 x 2 x 2 = 12

# 	b. Neural Net
# 	python -m himanshu.nn.nn_features_hotcold
# 		-state --include_0
# 		3 x 2 = 6

# 2. binary
# 	a. Sklearn
# 	python -m himanshu.nn.sklearn_train
# 		-state -label --include_0 --rand_forest
# 		3 x 7 x 2 x 2 = 84

# 	b. Neural Net
# 	python -m himanshu.nn.nn_feature_binary_arg
# 		-state -label --include_0
# 		3 x 7 x 2 = 42
states = [9, 21, 22]
labels = [1,2,3,4,5,6,7]

inc_0 = ['--include_0 ', '']
r_for = ['--rand_forest ', '']


# for state in states:
# 	for i in inc_0:
# 		for r in r_for:
# 			p = ('python -m himanshu.nn.sklearn_hotcold ' + '-state=' + str(state) + ' ' + i + r)
# 			print('echo ' + p)
# 			print(p)
# 			print('echo ----------------------------')
# print('')
# for state in states:
# 	for i in inc_0:
# 		p = ('python -m himanshu.nn.nn_features_hotcold ' + '-state=' + str(state) + ' ' + i)
# 		print('echo ' + p)
# 		print(p)
# 		print('echo ----------------------------')
# print('')
# for state in states:
# 	for label in labels:
# 		for i in inc_0:
# 			for r in r_for:
# 				p = ('python -m himanshu.nn.sklearn_train ' + '-state=' + str(state) + ' ' + '-label=' + str(label) + ' ' + i + r)
# 				print('echo ' + p)
# 				print(p)
# 				print('echo ----------------------------')
# print('')
# for state in states:
# 	for i in inc_0:
# 		for label in labels:
# 			p = ('python -m himanshu.nn.nn_feature_binary_arg ' + '-state=' + str(state) + ' ' + '-label=' + str(label) + ' ' + i)
# 			print('echo ' + p)
# 			print(p)
# 			print('echo ----------------------------')


for state in states:
	for i in inc_0:
		p = 'python -m himanshu.nn.nn_features_hotcold ' + '-state=' + str(state) + ' ' + i
		print('echo ' + p)
		print(p)
		print('echo ----------------------------')

		for r in r_for:
			for label in labels:
				p = 'python -m himanshu.nn.sklearn_train ' + '-state=' + str(state) + ' ' + '-label=' + str(label) + ' ' + i + r
				print('echo ' + p)
				print(p)
				print('echo ----------------------------')

		for r in r_for:
			p = 'python -m himanshu.nn.sklearn_hotcold ' + '-state=' + str(state) + ' ' + i + r
			print('echo ' + p)
			print(p)
			print('echo ----------------------------')

		for label in labels:
			p = 'python -m himanshu.nn.nn_feature_binary_arg ' + '-state=' + str(state) + ' ' + '-label=' + str(label) + ' ' + i
			print('echo ' + p)
			print(p)
			print('echo ----------------------------')

		print('')
		print('')
	print('')
	print('')






