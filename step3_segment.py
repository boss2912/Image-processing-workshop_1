"""
Step 3 — แยกดอกทานตะวันออกจากพื้นหลัง ด้วยการตัดค่าสี (Thresholding)
โจทย์ข้อ 3 (Lecture 10 สไลด์หน้า 63): ออกแบบ Algorithm แยกวัตถุออกจากพื้นหลัง
    วัตถุ = Positive Class (ขาว 255)   พื้นหลัง = Negative Class (ดำ 0)
    (Morphology เป็นโจทย์ข้อ 4 อยู่ใน step4)

หลักการ 2 ขั้น
    1. แปลงภาพสีเป็น "คะแนนความเหลือง" ตัวเลขเดียวต่อพิกเซล (0-255)
    2. พิกเซลที่คะแนนเกินค่าตัด = ดอก   ไม่เกิน = พื้นหลัง

ทำไมต้องเป็นตัวเลขเดียว
    ROC Curve คือผลของ classifier "at varying threshold values" (สไลด์หน้า 59)
    ต้องกวาดค่าตัดได้ทีละค่าเดียว ถ้าใช้ cv2.inRange กับ HSV ต้องตั้ง 6 ค่า จะกวาดไม่ได้

ทำไมใช้ช่อง b ของ Lab
    Lab มี 3 ช่อง  L = ความสว่าง  a = เขียว<->แดง  b = น้ำเงิน<->เหลือง
    ช่อง b บอกตรงๆ ว่าพิกเซลเหลืองแค่ไหน
    วัดคร่าวๆ จาก 5 ภาพที่พื้นหลังต่างกัน: ในดอกได้ราว 178-193  พื้นหลังราว 114-146
    ช่อง Hue ของ HSV ใช้ไม่ดี เพราะพิกเซลดำ เทา ขาว ไม่มีสี ค่า Hue จึงกระโดดมั่ว

make_yellow_score() กับ make_mask() แยกเป็น def ไว้ ให้ step ถัดไป import ไปใช้ได้
จะได้วัดผลด้วยโค้ดชุดเดียวกับที่รันจริง

รัน: python step3_segment.py   (ต้องรัน step1 ก่อน)
"""

import os
import sys

import cv2
import numpy as np


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "images")
MASK_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "mask_threshold")

# ค่าตัดเริ่มต้น 160 = ค่ากลางระหว่างพื้นหลัง (~130) กับดอก (~185)
# ลองกับทั้ง 50 ภาพแล้ว ค่า 150-180 ไม่มีภาพไหนที่ mask ขาวทั้งภาพหรือดำทั้งภาพ
# ยังไม่ใช่ค่าที่ดีที่สุด ค่าที่ดีที่สุดต้องดูจาก ROC Curve
YELLOW_THRESHOLD = 160


def make_yellow_score(image):
    """
    แปลงภาพสีเป็นภาพเทาที่บอกว่าแต่ละพิกเซลเหลืองแค่ไหน
    รับ     : image ภาพสีจาก cv2.imread (BGR)
    ส่งกลับ : ภาพเทาขนาดเท่าเดิม ค่า 0-255  ยิ่งมากยิ่งเหลือง
    """

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # ช่องที่ 0 = L, 1 = a, 2 = b   เอาช่อง b
    yellow_score = lab_image[:, :, 2]

    return yellow_score


def make_mask(yellow_score, threshold):
    """
    ตัดสินทีละพิกเซลว่าเป็นดอกหรือพื้นหลัง
    รับ     : yellow_score จาก make_yellow_score()   threshold ค่าตัด
    ส่งกลับ : mask ขาวดำ  255 = ดอก (Positive)  0 = พื้นหลัง (Negative)
    """

    # เริ่มจากภาพดำล้วน = ถือว่าทุกพิกเซลเป็นพื้นหลังไว้ก่อน
    mask = np.zeros(yellow_score.shape, np.uint8)

    # เลือกเฉพาะพิกเซลที่คะแนนเกินค่าตัด แล้วระบายเป็นขาว
    mask[yellow_score > threshold] = 255

    return mask


def main():
    # ให้ print ภาษาไทยได้เสมอ (เหตุผลอยู่ใน step1)
    sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(MASK_FOLDER, exist_ok=True)

    # เอาเฉพาะ .jpg กันไฟล์ .part (เน็ตหลุด) และ .DS_Store (Mac) ปนมา
    image_files = []
    for file_name in sorted(os.listdir(IMAGE_FOLDER)):
        if file_name.endswith(".jpg"):
            image_files.append(file_name)

    print("แยกดอกทานตะวันด้วยการตัดค่าสี  ค่าตัด = " + str(YELLOW_THRESHOLD))
    print("")

    for file_name in image_files:
        image = cv2.imread(os.path.join(IMAGE_FOLDER, file_name))

        yellow_score = make_yellow_score(image)
        mask = make_mask(yellow_score, YELLOW_THRESHOLD)

        # เซฟเป็น png เพราะ jpg บีบอัดแล้วค่า 255 อาจเพี้ยน
        mask_name = file_name.replace(".jpg", ".png")
        cv2.imwrite(os.path.join(MASK_FOLDER, mask_name), mask)

        # ภาพนี้ถูกตัดสินว่าเป็นดอกกี่เปอร์เซ็นต์  (ถ้าได้ 0 หรือ 100 แปลว่าค่าตัดไม่เหมาะกับภาพนี้)
        # count_nonzero นับพิกเซลที่ไม่ใช่ 0   mask.size คือจำนวนพิกเซลทั้งหมด
        white_percent = np.count_nonzero(mask) / mask.size * 100
        print(file_name + "   เป็นดอกไม้ " + str(round(white_percent, 1)) + "%")

    print("")
    print("เสร็จแล้ว เขียน mask ไป " + str(len(image_files)) + " ไฟล์")
    print("เปิดโฟลเดอร์ dataset/mask_threshold ดูผลได้เลย")


if __name__ == "__main__":
    main()
