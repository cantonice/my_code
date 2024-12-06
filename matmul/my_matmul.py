import numpy as np
import argparse


parser = argparse.ArgumentParser(description="my_matmul", add_help=False)
parser.add_argument("--matrix_1_path", required=True, help="做乘法的第一个矩阵的相对路径")
parser.add_argument("--matrix_2_path", required=True, help="做乘法的第二个矩阵的相对路径")
parser.add_argument("--save_file", default="result.txt", help="保存的文件名")
parser.add_argument("--load_stride", default=2, type=int, help="读取步长")
parser.add_argument("--load_lines", default=16, type=int, help="读取多少行做矩阵")
parser.add_argument("--system_of_numeration", default=16, type=int, choices=[2, 8, 10, 16], help="进制规则")

opt = parser.parse_args()

with open(opt.matrix_1_path, 'r') as f:
    matrix_data_1 = [[line[i:i + 2] for i in range(0, len(line.strip()), 2)] for line in
                     f.readlines()[:opt.load_lines]]

with open(opt.matrix_2_path, 'r') as f:
    matrix_data_2 = [[line[i:i + 2] for i in range(0, len(line.strip()), 2)] for line in
                     f.readlines()[:opt.load_lines]]

# 将矩阵数据转为numpy好做计算
matrix_1 = np.array([[int(x, opt.system_of_numeration) for x in row] for row in matrix_data_1])
matrix_2 = np.array([[int(x, opt.system_of_numeration) for x in row] for row in matrix_data_2])

# 做矩阵乘法,第一个矩阵转置乘以第二个矩阵
result = np.dot(matrix_1.T, matrix_2)

# 保存result为txt文件
np.savetxt(opt.save_file, result, fmt='%X')


def test_print_matrix():
    print(matrix_data_1)
    print(type(matrix_data_1))
    print(type(matrix_data_1[0]))
    print(type(matrix_data_1[0][0]))
    print(matrix_data_2)
    print(type(matrix_data_2))


def test_print_result():
    print(result)


if __name__ == '__main__':
    test_print_result()
    print(f"计算完成！result 文件存放在 {opt.save_file} ")
