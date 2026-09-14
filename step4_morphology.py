"""
Step 4 — ปรับปรุงผลลัพธ์ด้วย Morphology
โจทย์ข้อ 4 (Lecture 10 สไลด์หน้า 63): ทดลองใช้ Morphology algorithm เพื่อปรับปรุงผล Segmentation
รับ mask จาก step3 มาแต่ง แล้วเขียนลงโฟลเดอร์ใหม่

ปัญหาใน mask ของ step3 และท่าที่ใช้แก้
    1. จุดขาวเล็กๆ กระจายในพื้นหลัง  -> Opening = Erosion แล้ว Dilation (สไลด์หน้า 24)
       "removes completely regions of an object that cannot contain the structuring element"
    2. วงกลีบขาดเป็นตัว C             -> Closing = Dilation แล้ว Erosion (สไลด์หน้า 24)
       "joins narrow breaks, fills long thin gulfs"
    3. เกสรกลางดอกเป็นรูดำ (โดนัท)     -> Region Filling (สไลด์หน้า 34)
       สไลด์หน้า 35 บอกว่า OpenCV ไม่มี operation นี้ ให้ใช้วิธีจากลิงก์
       https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
       (Closing อุดรูเกสรไม่ได้ เพราะอุดได้แค่รูที่เล็กกว่า SE แต่เกสรใหญ่มาก)

ลำดับ Opening -> Closing -> Region Filling
    Closing ต้องมาก่อน Region Filling เพราะ Region Filling อุดได้เฉพาะรูที่ถูกล้อมมิด
    ถ้าวงกลีบยังขาด รูเกสรจะต่อถึงพื้นหลัง ไม่ถือว่าเป็นรู
    ส่วน Opening กับ Closing ถ้าสลับกัน ผลต่างกันจริง (รวม 50 ภาพต่างกัน 29,729 พิกเซล)
    แต่ยังไม่รู้ว่าแบบไหนแม่นกว่า ต้องวัดเทียบ ground truth ก่อน

clean_mask() แยกเป็น def ไว้ ให้ step ถัดไป import ไปใช้ได้

รัน: python step4_morphology.py   (ต้องรัน step3 ก่อน)
"""

import os
import sys

import cv2
import numpy as np


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_MASK_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "mask_threshold")
MORPHOLOGY_MASK_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "mask_morphology")

# ขนาดของ Structuring Element (SE) เป็นรูปวงรี (MORPH_ELLIPSE แบบตัวอย่างสไลด์หน้า 21)
# ดอกทานตะวันกลม SE วงรีให้ขอบโค้งตามดอก SE สี่เหลี่ยมจะให้ขอบเป็นเหลี่ยม
# Opening ใช้ SE เล็ก เพราะ SE ใหญ่จะลบปลายกลีบเรียวๆ ทิ้งไปด้วย
# Closing ใช้ SE ใหญ่กว่า เพราะ SE เล็กเชื่อมรอยขาดระหว่างกลีบไม่ติด
OPENING_SIZE = 5
CLOSING_SIZE = 11


def remove_small_spots(mask):
    """
    Opening: ลบก้อนขาวเล็กๆ ในพื้นหลัง
    รับ     : mask ขาวดำจาก step3
    ส่งกลับ : mask ที่จุดเล็กถูกลบแล้ว
    """

    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPENING_SIZE, OPENING_SIZE))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, structuring_element)


def close_small_gaps(mask):
    """
    Closing: เชื่อมรอยขาดแคบๆ ของวงกลีบให้ต่อกัน
    รับ     : mask ที่ผ่าน Opening แล้ว
    ส่งกลับ : mask ที่วงกลีบต่อกันมากขึ้น
    """

    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_SIZE, CLOSING_SIZE))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structuring_element)


def fill_holes(mask):
    """
    Region Filling: อุดรูที่ถูกสีขาวล้อมรอบมิดชิด (วิธีจากลิงก์ในสไลด์หน้า 35)
    รับ     : mask ที่ผ่าน Closing แล้ว
    ส่งกลับ : mask ที่รูข้างในถูกอุดแล้ว

    วิธีคิด: หารูตรงๆ ไม่ได้ แต่หาพื้นหลังนอกดอกได้ เพราะมันต่อถึงขอบภาพ
             ท่วมสีขาวจากมุมภาพ ส่วนที่ท่วมไม่ถึงและยังดำอยู่ = รู
    """

    # เติมขอบดำหนา 1 พิกเซลรอบภาพ ให้พื้นหลังทุกส่วนต่อถึงกันผ่านขอบนี้
    # ถ้าไม่เติม แล้วดอกชนขอบภาพ ดอกจะตัดพื้นหลังเป็นหลายส่วน ส่วนที่ท่วมจากมุม (0,0) ไปไม่ถึง
    # จะถูกนับเป็น "รู" แล้วโดนอุดเป็นขาว  (204p_0009: ขาว 69% ของภาพ กลายเป็น 85%)
    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # ท่วมสีขาวจากมุมซ้ายบน
    # floodFill บังคับให้ส่งภาพช่วยจำ (helper) ที่ใหญ่กว่าภาพจริงด้านละ 2 พิกเซลเข้าไปด้วย
    flooded = padded_mask.copy()
    padded_height, padded_width = flooded.shape
    helper = np.zeros((padded_height + 2, padded_width + 2), np.uint8)
    cv2.floodFill(flooded, helper, (0, 0), 255)

    # ตอนนี้ทุกอย่างขาว เหลือแค่รูที่ยังดำ  กลับสี -> เหลือแต่รูเป็นสีขาว
    holes_only = cv2.bitwise_not(flooded)

    # รวมรูเข้ากับ mask เดิม (OR = ขาวในอันไหนก็ได้ ให้เป็นขาว)
    filled_mask = cv2.bitwise_or(padded_mask, holes_only)

    # ตัดขอบ 1 พิกเซลที่เติมไว้ออก  [1:-1] = ตัดตัวแรกกับตัวสุดท้ายทิ้ง
    return filled_mask[1:-1, 1:-1]


def clean_mask(mask):
    """
    ทำ Morphology ครบ 3 ขั้นตามลำดับ
    รับ     : mask ดิบจาก step3
    ส่งกลับ : mask ที่ปรับปรุงแล้ว
    """

    mask = remove_small_spots(mask)
    mask = close_small_gaps(mask)
    mask = fill_holes(mask)

    return mask


def main():
    # ให้ print ภาษาไทยได้เสมอ (เหตุผลอยู่ใน step1)
    sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(MORPHOLOGY_MASK_FOLDER, exist_ok=True)

    # เอาเฉพาะ .png กันไฟล์อื่น เช่น .DS_Store (Mac) ปนมา
    mask_files = []
    for file_name in sorted(os.listdir(THRESHOLD_MASK_FOLDER)):
        if file_name.endswith(".png"):
            mask_files.append(file_name)

    print("ปรับปรุง mask ด้วย Morphology")
    print("    Opening  SE วงรีขนาด " + str(OPENING_SIZE))
    print("    Closing  SE วงรีขนาด " + str(CLOSING_SIZE))
    print("    Region Filling")
    print("")

    for file_name in mask_files:
        # 0 = อ่านเป็นภาพเทาช่องเดียว (แบบตัวอย่างสไลด์หน้า 21)
        mask = cv2.imread(os.path.join(THRESHOLD_MASK_FOLDER, file_name), 0)

        cleaned_mask = clean_mask(mask)

        cv2.imwrite(os.path.join(MORPHOLOGY_MASK_FOLDER, file_name), cleaned_mask)

        # พิกเซลขาวเปลี่ยนไปเท่าไหร่ จะได้เห็นว่า morphology ทำอะไรกับภาพนี้
        before = np.count_nonzero(mask)
        after = np.count_nonzero(cleaned_mask)
        print(file_name + "   " + str(before) + " -> " + str(after)
              + "   (เปลี่ยน " + str(after - before) + " พิกเซล)")

    print("")
    print("เสร็จแล้ว เขียน mask ไป " + str(len(mask_files)) + " ไฟล์")
    print("เปิดโฟลเดอร์ dataset/mask_morphology เทียบกับ dataset/mask_threshold ดูได้เลย")


if __name__ == "__main__":
    main()
