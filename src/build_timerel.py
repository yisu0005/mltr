import argparse
import numpy as np
import timeit

'''
Build additional training dataset by running a linear model with relevance and top 20 features (mimic the behaviour that rank depends on the time relevance).
'''


def get_data(file_loc):
	start = timeit.default_timer()
	f = open(file_loc, 'r')
	data = []
	for line in f:
		new_arr = []
		feat_len = 700
		arr = line.split(' #')[0].split()
		score = arr[0]
		q_id = arr[1].split(':')[1]
		new_arr.append(int(score))
		new_arr.append(int(q_id))
		arr = arr[2:]
		cur_idx = 0
		for el in arr:
			idx = int(el.split(':')[0])
			for i in range(cur_idx, idx):
				new_arr.append(0.0)
			new_arr.append(float(el.split(':')[1]))
			cur_idx = idx + 1
		for i in range(cur_idx, feat_len):
			new_arr.append(0.0)
		data.append(new_arr)
	end = timeit.default_timer()
	print("finished reading data: {}".format(end-start))
	f.close()
	return np.array(data)


def get_time_rel(data):
	start = timeit.default_timer()
	feat_len = len(data[0][2:])
	para = np.zeros(feat_len)
	para[:20] = np.random.uniform(1,10,20)
	feat_time = []
	time_rel = []
	for new_arr in data:
		q_id = new_arr[1]
		feat = np.array(new_arr[2:])
		feat_st = np.dot(feat, para)
		feat_time.append(feat_st)
	sorted_feat_time = sorted(feat_time)
	cutoff = sorted_feat_time[int(9.0/10.0*len(sorted_feat_time))]
	print("cutoff: {}".format(cutoff))
	timerel = [1 if time >= cutoff else 0 for time in feat_time]
	end = timeit.default_timer()
	print("finished relevant data: {}".format(end-start))
	return timerel


def save_time_rel(path, data, time_rel):
	start = timeit.default_timer()
	print("start wrting")
	with open(path, 'w') as fout:
		for i in range(len(time_rel)):
			temp_str = ''
			for j in range(len(data[0][2:])):
				if data[i][2+j] != 0.0:
					temp_str += str(j)+':'+str(data[i][2+j]) + ' '
			fout.write("{} qid:{} {}\n".format(time_rel[i], int(data[i][1]), temp_str))
	end = timeit.default_timer()
	print("finished writing data: {}".format(end-start))



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('feature_path', help='feature path')
	parser.add_argument('data_path', help='output data path')
	FLAGS, unparsed = parser.parse_known_args()

	data = get_data(FLAGS.feature_path)
	time_rel = get_time_rel(data)
	save_time_rel(FLAGS.data_path, data, time_rel)


if __name__ == '__main__':
	main()
