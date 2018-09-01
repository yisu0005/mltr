import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_path', help='result path')
    FLAGS, unparsed = parser.parse_known_args()

    best_ips = 99999
    best_c = 0
    with open(FLAGS.result_path, 'r') as f:
        lines = f.readlines()
        number = len(lines)
        for i in range(number):
            current_c = float(lines[i].split(":")[0])
            current_ips = float(lines[i].split(":")[1])
            if current_ips <= best_ips:
                best_c = current_c
                best_ips = current_ips
    print(best_c);



if __name__ == '__main__':
    main()  
