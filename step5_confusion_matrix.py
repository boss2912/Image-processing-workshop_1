"""
Step 5 — Confusion Matrix
โจทย์ข้อ 5 (Lecture 10 สไลด์หน้า 63): วัดประสิทธิภาพ แสดงผล Confusion Matrix
(ROC Curve ของข้อ 5 อยู่ใน step6)

นับทีละพิกเซล แล้วรวมทั้ง 50 ภาพเป็นตารางเดียว (แบบตัวอย่างสไลด์หน้า 51)
    เฉลย          = dataset/ground_truth      (คนระบาย)
    ผลที่โปรแกรมทาย = dataset/mask_threshold    (step3 ก่อน Morphology)
                    dataset/mask_morphology   (step4 หลัง Morphology)
    ขาว 255 = Positive (ดอกทานตะวัน)    ดำ 0 = Negative (พื้นหลัง)

สูตร (สไลด์หน้า 54-55)
    P = TP + FN (พิกเซลที่เฉลยเป็นดอก)    N = FP + TN (พิกเซลที่เฉลยเป็นพื้นหลัง)
    Accuracy        = (TP + TN) / (P + N)
    Precision       = TP / (TP + FP)
    Recall (TPR)    = TP / P
    Miss rate (FNR) = FN / P
    Fall-out (FPR)  = FP / N
    TNR             = TN / N
    F1              = 2 * Precision * Recall / (Precision + Recall)

    หมายเหตุ: สไลด์หน้า 55 เขียน FPR = FP/P และ FNR = FN/N ตัวหารสลับกับหน้า 54
    ไฟล์นี้ใช้ตามหน้า 54 (FPR = FP/N) เพราะ FP คือพิกเซลพื้นหลัง ต้องหารด้วยพิกเซลพื้นหลังทั้งหมด
    ถ้าหารด้วย P ค่าเกิน 1 ได้ แล้ววาดบนแกน ROC ที่มีแค่ 0 ถึง 1 (สไลด์หน้า 60) ไม่ได้

รัน: python step5_confusion_matrix.py
     (ต้องมี ground truth ครบ และรัน step3 step4 แล้ว)
     ผลเป็นภาพตารางอยู่ใน dataset/results/
"""

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "ground_truth")
THRESHOLD_MASK_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "mask_threshold")
MORPHOLOGY_MASK_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "mask_morphology")
RESULT_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "results")


def count_pixels(predicted, truth):
    """
    นับ TP FP FN TN ของภาพ 1 ภาพ
    รับ     : predicted mask ที่โปรแกรมทาย   truth เฉลย   (ทั้งคู่มีค่า 0 กับ 255 ขนาดเท่ากัน)
    ส่งกลับ : tp, fp, fn, tn
    """

    # & ระหว่าง array สองก้อน = "และ" ทีละพิกเซล
    # count_nonzero นับว่ามีพิกเซลที่เป็นจริงกี่พิกเซล
    tp = np.count_nonzero((predicted == 255) & (truth == 255))   # ทายว่าดอก      เฉลยเป็นดอก
    fp = np.count_nonzero((predicted == 255) & (truth == 0))     # ทายว่าดอก      เฉลยเป็นพื้นหลัง
    fn = np.count_nonzero((predicted == 0) & (truth == 255))     # ทายว่าพื้นหลัง  เฉลยเป็นดอก
    tn = np.count_nonzero((predicted == 0) & (truth == 0))       # ทายว่าพื้นหลัง  เฉลยเป็นพื้นหลัง

    return tp, fp, fn, tn


def count_all_images(mask_folder, file_names):
    """
    รวม TP FP FN TN ของทุกภาพเข้าเป็นชุดเดียว
    รับ     : mask_folder โฟลเดอร์ของผลที่โปรแกรมทาย   file_names รายชื่อไฟล์ .png
    ส่งกลับ : tp, fp, fn, tn รวมทุกภาพ
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for file_name in file_names:
        # 0 = อ่านเป็นภาพเทาช่องเดียว
        truth = cv2.imread(os.path.join(GROUND_TRUTH_FOLDER, file_name), 0)
        predicted = cv2.imread(os.path.join(mask_folder, file_name), 0)

        tp, fp, fn, tn = count_pixels(predicted, truth)

        total_tp = total_tp + tp
        total_fp = total_fp + fp
        total_fn = total_fn + fn
        total_tn = total_tn + tn

    return total_tp, total_fp, total_fn, total_tn


def print_metrics(title, tp, fp, fn, tn):
    """
    พิมพ์ตาราง Confusion Matrix และค่าที่คำนวณจากตาราง ตามสูตรสไลด์หน้า 54-55
    รับ     : title ชื่อที่จะพิมพ์บอก   tp fp fn tn จาก count_all_images()
    ส่งกลับ : ไม่มี
    """

    p = tp + fn
    n = fp + tn

    accuracy = (tp + tn) / (p + n)
    precision = tp / (tp + fp)
    recall = tp / p
    miss_rate = fn / p
    fall_out = fp / n
    tnr = tn / n
    f1 = 2 * precision * recall / (precision + recall)

    print("===== " + title + " =====")
    print("                      ทายว่าดอก       ทายว่าพื้นหลัง")
    print("เฉลยเป็นดอก       TP = " + str(tp).rjust(9) + "   FN = " + str(fn).rjust(9) + "   P = " + str(p))
    print("เฉลยเป็นพื้นหลัง    FP = " + str(fp).rjust(9) + "   TN = " + str(tn).rjust(9) + "   N = " + str(n))
    print("")
    print("Accuracy        = " + str(round(accuracy, 4)))
    print("Precision       = " + str(round(precision, 4)))
    print("Recall (TPR)    = " + str(round(recall, 4)))
    print("Miss rate (FNR) = " + str(round(miss_rate, 4)))
    print("Fall-out (FPR)  = " + str(round(fall_out, 4)))
    print("TNR             = " + str(round(tnr, 4)))
    print("F1              = " + str(round(f1, 4)))
    print("")


def draw_confusion_matrix(title, tp, fp, fn, tn, save_path):
    """
    วาดตาราง Confusion Matrix 2x2 เป็นภาพ วางแบบเดียวกับสไลด์หน้า 51
        แถว = Actual (เฉลย)   คอลัมน์ = Predicted (ที่โปรแกรมทาย)
    รับ     : title ชื่อบนภาพ   tp fp fn tn   save_path ที่จะเซฟภาพ
    ส่งกลับ : ไม่มี (ได้ไฟล์ภาพ)
    """

    total = tp + fp + fn + tn

    # สีตามสไลด์หน้า 51: TP เขียว  FN แดง  FP ชมพู  TN เขียวอ่อน   (ค่าสีเป็น R G B ช่วง 0-1)
    colors = np.zeros((2, 2, 3))
    colors[0, 0] = (0.60, 0.80, 0.20)   # TP
    colors[0, 1] = (1.00, 0.00, 0.00)   # FN
    colors[1, 0] = (1.00, 0.60, 0.60)   # FP
    colors[1, 1] = (0.85, 0.92, 0.70)   # TN

    names = [["TP", "FN"], ["FP", "TN"]]
    values = [[tp, fn], [fp, tn]]

    # matplotlib ไม่มีฟอนต์ไทยมาให้ ข้อความบนภาพจึงเป็นภาษาอังกฤษ
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(colors)

    for row in range(2):
        for column in range(2):
            value = values[row][column]
            percent = value / total * 100
            text = names[row][column] + "\n" + str(value) + " px\n(" + str(round(percent, 1)) + "% of all)"
            axis.text(column, row, text, ha="center", va="center", fontsize=12)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Positive (PP)", "Negative (PN)"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Positive (P)", "Negative (N)"])
    axis.set_xlabel("Predicted condition")
    axis.set_ylabel("Actual condition")
    axis.set_title(title)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def main():
    # ให้ print ภาษาไทยได้เสมอ (เหตุผลอยู่ใน step1)
    sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    # เอาเฉพาะ .png กันไฟล์อื่น เช่น .DS_Store (Mac) ปนมา
    file_names = []
    for file_name in sorted(os.listdir(GROUND_TRUTH_FOLDER)):
        if file_name.endswith(".png"):
            file_names.append(file_name)

    print("นับพิกเซลรวม " + str(len(file_names)) + " ภาพ")
    print("")

    # ----- ก่อน Morphology -----
    tp, fp, fn, tn = count_all_images(THRESHOLD_MASK_FOLDER, file_names)
    print_metrics("ก่อน Morphology (step3)", tp, fp, fn, tn)
    draw_confusion_matrix("Before morphology (step3)", tp, fp, fn, tn,
                          os.path.join(RESULT_FOLDER, "confusion_matrix_before_morphology.png"))

    # ----- หลัง Morphology -----
    tp, fp, fn, tn = count_all_images(MORPHOLOGY_MASK_FOLDER, file_names)
    print_metrics("หลัง Morphology (step4)", tp, fp, fn, tn)
    draw_confusion_matrix("After morphology (step4)", tp, fp, fn, tn,
                          os.path.join(RESULT_FOLDER, "confusion_matrix_after_morphology.png"))

    print("เซฟภาพตารางไว้ที่โฟลเดอร์ dataset/results แล้ว")


if __name__ == "__main__":
    main()
