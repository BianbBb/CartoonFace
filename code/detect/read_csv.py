import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np

T = 0
test_dir = '/data/byh/CartoonFace/personai_icartoonface_detval/'
# result_file = open('/home/byh/CartoonFace/result/result_T.csv', 'w', encoding='utf-8')
# csv_writer = csv.writer(result_file)

def draw_rect(img, result):
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
# def upgrade_result(result,T):
#     if float(result[6]) > T:
#         text = result
#         csv_writer.writerow(text)

def draw(scores):
    x = np.linspace(1,10,10)
    plt.plot(x,scores)
    print(x.shape())
    print(scores.shape())
    plt.show()



with open('/home/byh/CartoonFace/result/result_.csv','r') as f:
    lines = csv.reader(f, delimiter = ',')
    img_path = None
    img = None
    scores = []
    for line in lines:
        #upgrade_result(line,T=0.23)

        if line[0] != img_path:
            if img is not None:
                cv2.imshow('img', img)
                cv2.waitKey(0)
            img_path = line[0]
            img = cv2.imread(test_dir+img_path)
            draw_rect(img,line)
            scores.append(float(line[6]))
            draw(scores)
        else:
            draw_rect(img, line)
            scores.append(float(line[6]))

f.close()
#result_file.close()



# if __name__ == '__main__':
#     show_sample()
#     upgrade_result()
#

