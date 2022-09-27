"""
@Name           :exp2_camera_position.py
@Description    :
@Time           :2022/07/24 23:17:11
@Author         :Zijie NING
@Version        :1.0
"""


import torch
import argparse
import sys
from exp1_multiview import strategie_max, strategie_mean, strategie_sum

parser = argparse.ArgumentParser(description="Yolov5 camera position experiment")
parser.add_argument(
    "--weight",
    default="/homes/z20ning/Bureau/Yolo/yolov5/YOLOv5/yolov5n_sm/weights/best.pt",
    type=str,
    help="weight of trained model",
)
parser.add_argument("--batch_size", default=80, type=int, help="how many IMAGES to predict at once")
parser.add_argument(
    "--strategie", default="max", type=str, help="strategie to decide the pred of scene: max, mean, sum"
)
parser.add_argument(
    "--data", default="test_exp2.txt", type=str, help="list of test images"
)
args = parser.parse_args()


# Model
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.hub.load("ultralytics/yolov5", "custom", args.weight)  # custom trained model
model.multi_label = True

# Load test images
im = []  # or file, Path, URL, PIL, OpenCV, numpy, list
label = []
file = open(args.data, "r")
print(f"file name = ", file.name)
for line in file.readlines():
    line = line.strip("\n")
    im.append(line)
    line = line.strip(".png")
    label.append(line + ".txt")
file.close()
# print(im)
# print(label)

n_img = len(im)
n_scene = n_img / 8


cam_list_list = []
cam_list1 = [0, 1, 3, 7]
cam_list2 = [1, 2, 4, 5]
cam_list_list.append(cam_list1)
cam_list_list.append(cam_list2)

n_truepred = 0
n_truepred_im = 0
n_truepred_list = [0] * len(cam_list_list)
count_img = 0  # overall index of image

while count_img < n_img:
    count_batch = 0  # index of image in each batch

    # Inference
    results = model(im[count_img : count_img + args.batch_size], size=640)
    print(f"\n\033[32mresult.pred:\n {results.pred}\033[0m\n")
    # exit()

    # Print result for each image
    for i in range(0, args.batch_size):
        print(
            f"\n\033[32mresult for img", count_img + count_batch, "\033[0m\n", results.pandas().xyxy[i]
        )  # im predictions (pandas)
        break  # Print only the result of first image
    # exit()

    while count_batch < args.batch_size:
        # To get the j-th instance pred of i-th image, use:
        # x0, y0, x1, y1, conf, pred = results.xyxy[i][j].cpu().numpy()
        #      xmin    ymin    xmax   ymax  confidence  class    name
        # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
        # 1  114.75  195.75  1095.0  708.0    0.624512      0  person
        # 2  986.00  304.00  1028.0  420.0    0.286865     27     tie

        pred_scene_list = []

        for cam_list in cam_list_list:
            if args.strategie == "max":
                # Choose the pred from a cam with highest confidence
                pred_scene = strategie_max(results, count_batch, count_img, cam_list)

            elif args.strategie == "mean":
                # Choose the pred with highest mean confidence from all cams
                pred_scene = strategie_mean(results, count_batch, count_img, cam_list)

            elif args.strategie == "sum":
                # Choose the pred with highest sum confidence from all cams
                pred_scene = strategie_sum(results, count_batch, count_img, cam_list)

            else:
                print(f"\033[31mStrategie Error: ", args.strategie, "\033[0m")
                sys.exit()

            pred_scene_list.append(pred_scene)

        # Read ground truth
        ground_truth_im = []
        for cam_index in range(0, 8):
            file = open(label[count_img + count_batch + cam_index], "r")
            label_read = file.read(1)
            try:
                ground_truth_im.append(int(label_read))  # if label is not empty
                ground_truth = int(label_read)
            except:
                ground_truth_im.append(-1)
            file.close()
        print(f"\033[32mground_truth:\033[0m", ground_truth, "")

        # Calculate precision with multi-view

        for i in range(0, len(pred_scene_list)):
            if int(pred_scene_list[i]) == int(ground_truth):
                n_truepred_list[i] = n_truepred_list[i] + 1

        # Calculate precision without multi-view
        for cam_index in range(0, 8):
            if len(results.xyxy[count_batch + cam_index]) > 0:
                pred_img = results.xyxy[count_batch + cam_index][0][5].cpu().numpy().astype(int)
            else:
                pred_img = -1
            if int(pred_img) == int(ground_truth_im[cam_index]):
                n_truepred_im = n_truepred_im + 1

        count_batch = count_batch + 8

    count_img = count_img + args.batch_size

print(f"\033[32mPrecision of image:\033[0m", n_truepred_im / n_img, "")
for i in range(0, len(cam_list_list)):
    print(
        f"\033[32mPrecision of scene by\033[0m {cam_list_list[i]}\033[32m: \033[0m{float(n_truepred_list[i] / n_scene)}"
    )


# if __name__ == "__main__":
#     main()
