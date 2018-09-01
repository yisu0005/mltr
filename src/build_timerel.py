import argparse
import numpy as np


def get_data(file_loc):
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
	f.close()
	return np.array(data)

## TODO: Rewrite rel based on a particular function
def get_time_rel(data):
    feat_len = len(data[0][2:])
    para = np.zeros(feat_len)
    para[:20] = np.random.uniform([1,10],20)
    
    for new_arr in data:
        q_id = new_arr[1]
        feat = np.array(new_arr[2:])
        feat_st = np.dot(feat, para)





## TODO: save as same format as set1bin.train.txt
def save_time_rel(data):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_path', help='feature path')
    FLAGS, unparsed = parser.parse_known_args()
