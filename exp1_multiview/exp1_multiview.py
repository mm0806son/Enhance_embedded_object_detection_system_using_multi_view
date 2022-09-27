"""
@Name           :test_image.py
@Description    :
@Time           :2022/06/16 15:14:50
@Author         :Zijie NING
@Version        :2.0
"""

import torch
import argparse
import sys


def strategie_max(results, count_batch, count_img, cam_list):
    conf_scene = []
    # j = 0  # j-th pred of the image
    for cam_index in cam_list:
        if len(results.xyxy[count_batch + cam_index]) > 0:
            # conf_scene.append(results.xyxy[count_batch + cam_index][j][4].cpu().numpy())
            conf_scene.append(results.xyxy[count_batch + cam_index][0][4].cpu().numpy())
        else:
            conf_scene.append(-1)
    conf_max = max(conf_scene)  # Highest confidence
    if conf_max == -1:
        cam_max = -1
        pred_scene = -1
    else:
        cam_max = conf_scene.index(conf_max)  # Where conf_max comes from
        pred_scene = results.xyxy[count_batch + cam_list[cam_max]][0][5].cpu().numpy().astype(int)
    print(
        f"\033[32mResult_max: scene \033[0m",
        count_img + count_batch,
        " \033[32mpred:\033[0m ",
        pred_scene,
        " \033[32mwith conf:\033[0m ",
        conf_max,
        " \033[32mfrom cam \033[0m ",
        cam_list[cam_max],
    )
    return pred_scene


def strategie_mean(results, count_batch, count_img, cam_list):
    pred_list = {}
    pred_count = {}
    for cam_index in cam_list:
        if len(results.xyxy[count_batch + cam_index]) > 0:
            _, _, _, _, conf, pred = results.xyxy[count_batch + cam_index][0].cpu().numpy()
            if pred not in pred_list:
                pred_list[int(pred)] = conf
                pred_count[int(pred)] = 1
            else:
                pred_count[int(pred)] = pred_count[int(pred)] + 1
                pred_list[int(pred)] = (pred_list[pred] * (pred_count[int(pred)] - 1) + conf) / pred_count[int(pred)]
    print(f"\033[32mpred_list ", pred_list, "\033[0m")
    if len(pred_list) > 0:
        pred_scene = max(pred_list, key=pred_list.get)
        conf_mean = pred_list[pred_scene]
    else:
        pred_scene = -1
        conf_mean = -1
    print(
        f"\033[32mResult_mean: scene \033[0m",
        count_img + count_batch,
        " \033[32mpred:\033[0m ",
        pred_scene,
        " \033[32mwith conf:\033[0m ",
        conf_mean,
    )
    return pred_scene


def strategie_sum(results, count_batch, count_img, cam_list):
    pred_list = {}
    for cam_index in cam_list:
        if len(results.xyxy[count_batch + cam_index]) > 0:
            _, _, _, _, conf, pred = results.xyxy[count_batch + cam_index][0].cpu().numpy()
            if pred not in pred_list:
                pred_list[int(pred)] = conf
            else:
                pred_list[int(pred)] = pred_list[pred] + conf
    print(f"\033[32mpred_list ", pred_list, "\033[0m")
    if len(pred_list) > 0:
        pred_scene = max(pred_list, key=pred_list.get)
        conf_sum = pred_list[pred_scene]
    else:
        pred_scene = -1
        conf_sum = -1
    print(
        f"\033[32mResult_sum: scene \033[0m",
        count_img + count_batch,
        " \033[32mpred:\033[0m ",
        pred_scene,
        " \033[32mwith conf:\033[0m ",
        conf_sum,
    )
    return pred_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yolov5 multi-view experiment")
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
        "--data", default="test_exp1.txt", type=str, help="list of test images"
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
    n_scene = n_img / 4
    n_truepred = [0, 0, 0]
    n_truepred_im = 0
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

            cam_list = range(0, 4)

            # if args.strategie == "max":
            # Choose the pred from a cam with highest confidence
            pred_scene_max = strategie_max(results, count_batch, count_img, cam_list)

            # elif args.strategie == "mean":
            # Choose the pred with highest mean confidence from all cams
            pred_scene_mean = strategie_mean(results, count_batch, count_img, cam_list)

            # elif args.strategie == "sum":
            # Choose the pred with highest sum confidence from all cams
            pred_scene_sum = strategie_sum(results, count_batch, count_img, cam_list)

            # else:
            #     print(f"\033[31mStrategie Error: ", args.strategie, "\033[0m")
            #     sys.exit()

            # Read ground truth
            ground_truth_im = []
            for cam_index in range(0, 4):
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
            for i in range(0, 3):
                pred_scene = [pred_scene_max, pred_scene_mean, pred_scene_sum][i]
                if int(pred_scene) == int(ground_truth):
                    n_truepred[i] = n_truepred[i] + 1

            # Calculate precision without multi-view
            for cam_index in range(0, 4):
                if len(results.xyxy[count_batch + cam_index]) > 0:
                    pred_img = results.xyxy[count_batch + cam_index][0][5].cpu().numpy().astype(int)
                else:
                    pred_img = -1
                if int(pred_img) == int(ground_truth_im[cam_index]):
                    n_truepred_im = n_truepred_im + 1

            count_batch = count_batch + 4

        count_img = count_img + args.batch_size

    print(f"\033[32mPrecision of image:\033[0m", n_truepred_im / n_img, "")
    print(f"\033[32mPrecision of scene_max: \033[0m", n_truepred[0] / n_scene, "")
    print(f"\033[32mPrecision of scene_mean:\033[0m", n_truepred[1] / n_scene, "")
    print(f"\033[32mPrecision of scene_sum: \033[0m", n_truepred[2] / n_scene, "")
