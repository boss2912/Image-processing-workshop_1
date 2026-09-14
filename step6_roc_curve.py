"""
Step 6 — ROC Curve
โจทย์ข้อ 5 (Lecture 10 สไลด์หน้า 63): แสดงผล ROC Curve

ROC Curve คือผลของ binary classifier "at varying threshold values" (สไลด์หน้า 59)
    กวาดค่าตัดความเหลืองตั้งแต่ 0 ถึง 255 ทีละ 1
    แต่ละค่าตัด นับ TP FP FN TN รวมทั้ง 50 ภาพ ได้ 1 จุดบนกราฟ
        แกนนอน = FPR = FP / N     แกนตั้ง = TPR = TP / P      (สูตรสไลด์หน้า 54)
    เส้นยิ่งเข้าใกล้มุมซ้ายบน ยิ่งดี   เส้นทแยงมุม = Random classifier (สไลด์หน้า 60)

วาด 2 เส้น เพื่อดูว่า Morphology (โจทย์ข้อ 4) ช่วยได้จริงไหม
    ก่อน Morphology = make_mask() ของ step3
    หลัง Morphology = make_mask() แล้วตามด้วย clean_mask() ของ step4
    import def มาจาก step3 step4 และ step5 โดยตรง จึงเป็นโค้ดชุดเดียวกับที่รันจริง

รัน: python step6_roc_curve.py   (ต้องมี ground truth ครบ)
     ผลเป็นภาพกราฟอยู่ใน dataset/results/roc_curve.png
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt

from step3_segment import make_yellow_score, make_mask, YELLOW_THRESHOLD
from step4_morphology import clean_mask
from step5_confusion_matrix import count_pixels


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "images")
GROUND_TRUTH_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "ground_truth")
RESULT_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "results")


def roc_points(yellow_scores, truths, use_morphology):
    """
    กวาดค่าตัด 0 ถึง 255 แล้วหา FPR กับ TPR ของแต่ละค่าตัด
    รับ     : yellow_scores รายการภาพความเหลืองของทุกภาพ
              truths รายการเฉลยของทุกภาพ (เรียงลำดับตรงกับ yellow_scores)
              use_morphology True = ทำ clean_mask() ต่อจาก make_mask() ด้วย
    ส่งกลับ : fpr_list, tpr_list  ยาว 256 ตัว  ตัวที่ i คือผลของค่าตัด i
    """

    fpr_list = []
    tpr_list = []

    for threshold in range(0, 256):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        for i in range(len(truths)):
            mask = make_mask(yellow_scores[i], threshold)

            if use_morphology:
                mask = clean_mask(mask)

            tp, fp, fn, tn = count_pixels(mask, truths[i])

            total_tp = total_tp + tp
            total_fp = total_fp + fp
            total_fn = total_fn + fn
            total_tn = total_tn + tn

        tpr = total_tp / (total_tp + total_fn)   # TP / P
        fpr = total_fp / (total_fp + total_tn)   # FP / N

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list


def main():
    # ให้ print ภาษาไทยได้เสมอ (เหตุผลอยู่ใน step1)
    sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    # อ่านทุกภาพครั้งเดียว เก็บไว้ในรายการ จะได้ไม่ต้องอ่านไฟล์ซ้ำ 256 รอบ
    yellow_scores = []
    truths = []
    for file_name in sorted(os.listdir(GROUND_TRUTH_FOLDER)):
        if file_name.endswith(".png"):
            image = cv2.imread(os.path.join(IMAGE_FOLDER, file_name.replace(".png", ".jpg")))
            yellow_scores.append(make_yellow_score(image))
            truths.append(cv2.imread(os.path.join(GROUND_TRUTH_FOLDER, file_name), 0))

    print("กวาดค่าตัด 0-255 กับ " + str(len(truths)) + " ภาพ (เส้นหลัง Morphology ใช้เวลานานกว่า)")

    fpr_before, tpr_before = roc_points(yellow_scores, truths, False)
    print("    เส้นก่อน Morphology เสร็จ")
    fpr_after, tpr_after = roc_points(yellow_scores, truths, True)
    print("    เส้นหลัง Morphology เสร็จ")
    print("")

    # พิมพ์ตัวเลขบางค่าตัดไว้ใช้ตอนอภิปรายผล
    print("ค่าตัด    ก่อน Morphology (FPR, TPR)    หลัง Morphology (FPR, TPR)")
    for threshold in range(100, 221, 10):
        print(str(threshold).rjust(5)
              + "      " + str(round(fpr_before[threshold], 3)).ljust(6) + "  " + str(round(tpr_before[threshold], 3)).ljust(6)
              + "                " + str(round(fpr_after[threshold], 3)).ljust(6) + "  " + str(round(tpr_after[threshold], 3)))
    print("")

    # ----- วาดกราฟ (ข้อความบนกราฟเป็นอังกฤษ เพราะ matplotlib ไม่มีฟอนต์ไทย) -----
    plt.figure(figsize=(7, 7))

    plt.plot(fpr_before, tpr_before, label="Before morphology (step3)")
    plt.plot(fpr_after, tpr_after, label="After morphology (step4)")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")

    # จุดของค่าตัดที่ step3 ใช้อยู่ตอนนี้
    plt.plot(fpr_before[YELLOW_THRESHOLD], tpr_before[YELLOW_THRESHOLD], "o", color="C0")
    plt.plot(fpr_after[YELLOW_THRESHOLD], tpr_after[YELLOW_THRESHOLD], "o", color="C1")
    plt.annotate("threshold " + str(YELLOW_THRESHOLD),
                 (fpr_after[YELLOW_THRESHOLD], tpr_after[YELLOW_THRESHOLD]),
                 textcoords="offset points", xytext=(15, -15))

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False positive rate (FPR = FP / N)")
    plt.ylabel("True positive rate (TPR = TP / P)")
    plt.title("ROC curve - sunflower segmentation, " + str(len(truths)) + " images, pixel level")
    plt.grid(True)
    plt.legend(loc="lower right")

    save_path = os.path.join(RESULT_FOLDER, "roc_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("เซฟกราฟไว้ที่ dataset/results/roc_curve.png แล้ว")


if __name__ == "__main__":
    main()
