import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np




def draw_rect(img, result, T=0.0):
    x_min = int(result[1])
    y_min = int(result[2])
    x_max = int(result[3])
    y_max = int(result[4])
    score = float(result[6])
    if score > T:
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (int(255 * score), 255 - int(255 * score), 255), thickness=2)
        cv2.putText(img, '{:.2f}'.format(score), (x_min, y_min - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,120,255), thickness=1, lineType=cv2.LINE_AA)
#
def upgrade_result(old_file,new_file,T):
    new_f = open(new_file, 'w', encoding='utf-8')
    csv_writer = csv.writer(new_f)
    with open(old_file, 'r') as f:
        lines = csv.reader(f, delimiter=',')

        for line in lines:
            if float(line[6]) > T:
                text = line
                csv_writer.writerow(text)
    f.close()
    new_f.close()



def draw(scores):
    x = np.linspace(1,10,10)
    scores = np.array(scores)
    plt.plot(x,scores)
    plt.show()


#result_file.close()

def show_sample():
    with open(result_file, 'r') as f:
        lines = csv.reader(f, delimiter=',')
        img_path = None
        img = None
        scores = []
        for line in lines:
            # upgrade_result(line,T=T)
            if line[0] != img_path:
                if img_path is not None:
                    print(img_path)
                    cv2.imshow('img', img)
                    #draw(scores)
                    cv2.waitKey(0)
                img_path = line[0]
                img = cv2.imread(test_dir + img_path)
                draw_rect(img, line,T)
                scores = []
                scores.append(float(line[6]))
            else:
                draw_rect(img, line,T)
                scores.append(float(line[6]))
    f.close()


def cal_T(scores):
    return T




if __name__ == '__main__':
    test_dir = '/data/byh/CartoonFace/personai_icartoonface_detval/'  # 测试图片数据集路径
    result_file = '/home/byh/CartoonFace/result/result.csv'  # 原result.csv 路径
    new_result_file = '/home/byh/CartoonFace/result/result_T4.csv'  # 经过过滤的新result路径

    T = 0.18
    # 显示10 boxes + 折线图
    # 显示过滤后的 boxes
    # 更新result
    show_sample()

    #upgrade_result(result_file,new_result_file,T)

