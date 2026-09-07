"""
Step 4 — ปรับปรุงผลลัพธ์ด้วย Morphology
วิชา 310-3311 Image Processing / Workshop ท้าย Lecture 10

โจทย์ (สไลด์หน้า 63)
    ข้อ 4  ทดลองใช้ Morphology algorithm เพื่อปรับปรุงผลลัพธ์กระบวนการ Segmentation

    ไฟล์นี้ทำข้อ 4 อย่างเดียว
    รับ mask ที่ step3 ทำไว้ เอามาแต่งให้ดีขึ้น แล้วเขียนลงโฟลเดอร์ใหม่
    ไม่ยุ่งกับการตัดค่าสี ซึ่งเป็นงานของ step3

ปัญหา 3 อย่างที่เห็นจากผล step3 และท่าที่ใช้แก้แต่ละอย่าง

    ปัญหาที่ 1  พื้นหลังมีจุดขาวเล็กๆ กระจาย
                เกิดจากใบไม้หรือดอกอื่นที่บังเอิญเหลืองพอจะผ่านค่าตัด
                แก้ด้วย Opening (สไลด์หน้า 24)
                "removes completely regions of an object that cannot contain
                 the structuring element" คือลบก้อนที่เล็กกว่า SE ทิ้ง
                Opening = Erosion แล้วตามด้วย Dilation

    ปัญหาที่ 2  วงกลีบดอกขาดเป็นตัว C ไม่ครบวง
                เกิดจากกลีบบางกลีบสีเข้มไป ไม่ผ่านค่าตัด
                แก้ด้วย Closing (สไลด์หน้า 24)
                "joins narrow breaks, fills long thin gulfs" คือเชื่อมรอยขาดแคบๆ
                Closing = Dilation แล้วตามด้วย Erosion

    ปัญหาที่ 3  เกสรกลางดอกเป็นรูดำ ทำให้ mask เป็นวงโดนัท
                เกิดจากเกสรสีน้ำตาลเข้ม คะแนนความเหลืองไม่ถึงค่าตัด
                แต่ในเฉลยเราถือว่าเกสรเป็นส่วนหนึ่งของดอก จึงต้องอุดรู
                แก้ด้วย Region Filling (สไลด์หน้า 34)

                หมายเหตุ  เคยลองใช้ Closing อุดรูนี้แล้ว ไม่ได้ผล
                          วัดจริงกับ 204p_0001 ได้ 20,052 -> 21,318 พิกเซล เพิ่มแค่ 6%
                          เพราะ Closing อุดได้เฉพาะรูที่เล็กกว่า SE
                          แต่เกสรใหญ่มาก ถ้าใช้ SE ใหญ่พอจะอุดได้ SE นั้นจะไปกลืนพื้นหลังด้วย
                          ครูเขียนกำกับไว้ในสไลด์หน้า 35 แล้วว่า Region filling ของ Python
                          ให้ตามไปอ่านลิงก์นี้แทน เพราะ OpenCV ไม่มี operation นี้ให้เลือกใช้
                          https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

ลำดับต้องเป็นแบบนี้เท่านั้น
    Opening -> Closing -> Region Filling
    ถ้าสลับเอา Region Filling ขึ้นก่อน จุดขาวเล็กๆ ในพื้นหลังจะยังอยู่
    และถ้าวงกลีบยังขาดอยู่ รูตรงกลางจะทะลุออกนอก ไม่นับเป็นรู อุดไม่ได้
    จึงต้อง Closing ปิดวงให้ครบก่อน แล้วค่อยอุด

ไฟล์นี้มี 6 def แต่ละอันทำงานอย่างเดียว เรียงตามลำดับที่ถูกเรียกใช้จริง
    1. make_structuring_element()  สร้าง SE ที่ Opening กับ Closing ใช้
    2. remove_small_spots()        Opening   ลบจุดเล็กในพื้นหลัง
    3. close_small_gaps()          Closing   เชื่อมรอยขาดของวงกลีบ
    4. fill_holes()                Region Filling  อุดรูเกสร
    5. clean_mask()                เรียก 3 อันบนตามลำดับ
    6. main()                      วนทำทั้ง 50 ภาพ

    step6_roc_curve.py จะเรียก clean_mask() จากไฟล์นี้ไปใช้ต่อ
    เพื่อให้ ROC เส้น "หลัง morphology" ใช้โค้ดชุดเดียวกับที่รันจริงเป๊ะๆ

รัน:  python step4_morphology.py
      (ต้องรัน step3 ให้เสร็จก่อน)
"""

import os
import sys

import cv2
import numpy as np


# บังคับให้ข้อความที่ print ออกไป ใช้ตารางตัวอักษร utf-8 เสมอ
# (เหตุผลเต็มๆ อธิบายไว้ใน step1_download_images.py)
sys.stdout.reconfigure(encoding="utf-8")


# โฟลเดอร์ที่ไฟล์ .py นี้วางอยู่
# (เหตุผลเต็มๆ ว่าทำไมไม่เขียน path ตรงๆ อธิบายไว้ใน step1_download_images.py)
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

THRESHOLD_MASK_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "mask_threshold")
MORPHOLOGY_MASK_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "mask_morphology")

# ขนาดของ Structuring Element
#
# ทำไมต้องแยกขนาดของ Opening กับ Closing
#     Opening ลบก้อนที่เล็กกว่า SE ถ้าใช้ SE ใหญ่ไป จะไปลบปลายกลีบดอกที่เรียวๆ ทิ้งด้วย
#     Closing เชื่อมรอยขาด ถ้าใช้ SE เล็กไป จะเชื่อมช่องว่างระหว่างกลีบไม่ติด
#     งานคนละแบบกัน จึงไม่ควรบังคับให้ใช้เลขเดียวกัน
OPENING_SIZE = 5
CLOSING_SIZE = 11


# ====================================================================
#  1. make_structuring_element() — สร้าง SE
# ====================================================================

def make_structuring_element(size):
    """
    สร้าง Structuring Element รูปวงรี ตามขนาดที่บอก

    รับ     : size ความกว้างและความสูงของ SE เป็นพิกเซล
    ส่งกลับ : SE ที่เอาไปใส่ให้ cv2.morphologyEx ใช้ได้

    ทำไมเลือกวงรี ไม่เลือกสี่เหลี่ยม
        สไลด์หน้า 13 บอกว่ารูปร่างของ SE เป็นตัวกำหนดว่าวัตถุจะโตหรือหดไปทางไหน
        ดอกทานตะวันกลม ถ้าใช้ SE สี่เหลี่ยม ขอบที่ได้จะเป็นเหลี่ยมตามไปด้วย
        วงรีให้ขอบที่โค้งตามรูปดอกมากกว่า
        (โค้ดตัวอย่างที่ครูให้ในสไลด์หน้า 21 ก็ใช้ cv.MORPH_ELLIPSE เหมือนกัน)
    """

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# ====================================================================
#  2. remove_small_spots() — Opening ลบจุดเล็กในพื้นหลัง
# ====================================================================

def remove_small_spots(mask):
    """
    ลบก้อนสีขาวเล็กๆ ในพื้นหลังทิ้ง ด้วย Opening

    รับ     : mask ภาพขาวดำจาก step3
    ส่งกลับ : mask ที่จุดเล็กๆ ถูกลบไปแล้ว

    Opening ทำงานยังไง (สไลด์หน้า 24)
        Erosion ก่อน ทำให้ทุกก้อนหดลง ก้อนที่เล็กกว่า SE จะหายไปเลย
        แล้ว Dilation ตาม ทำให้ก้อนที่รอดกลับมาขนาดใกล้เคียงเดิม
        ผลคือ ก้อนเล็กหาย ก้อนใหญ่อยู่ครบ
    """

    structuring_element = make_structuring_element(OPENING_SIZE)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, structuring_element)


# ====================================================================
#  3. close_small_gaps() — Closing เชื่อมรอยขาดของวงกลีบ
# ====================================================================

def close_small_gaps(mask):
    """
    เชื่อมรอยขาดแคบๆ ของวงกลีบดอกให้ต่อกัน ด้วย Closing

    รับ     : mask ที่ผ่าน Opening มาแล้ว
    ส่งกลับ : mask ที่วงกลีบต่อกันครบวงมากขึ้น

    Closing ทำงานยังไง (สไลด์หน้า 24)
        Dilation ก่อน ทำให้ทุกก้อนพองออก ช่องว่างแคบๆ ถูกปิดสนิท
        แล้ว Erosion ตาม ทำให้ก้อนหดกลับมาขนาดใกล้เคียงเดิม แต่รอยที่ปิดไปแล้วไม่เปิดกลับ

    ขั้นนี้สำคัญกับขั้นถัดไป
        fill_holes() อุดได้เฉพาะรูที่ถูกล้อมรอบมิดชิด
        ถ้าวงกลีบยังขาด รูตรงกลางจะทะลุออกนอกภาพ ไม่ถือว่าเป็นรู อุดไม่ได้
    """

    structuring_element = make_structuring_element(CLOSING_SIZE)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structuring_element)


# ====================================================================
#  4. fill_holes() — Region Filling อุดรูเกสรกลางดอก
# ====================================================================

def fill_holes(mask):
    """
    อุดรูที่ถูกสีขาวล้อมรอบมิดชิด ให้กลายเป็นสีขาวไปด้วย

    รับ     : mask ที่ผ่าน Closing มาแล้ว
    ส่งกลับ : mask ที่รูตรงกลางถูกอุดแล้ว

    วิธีคิด
        เราหา "รู" ตรงๆ ไม่ได้ แต่หา "พื้นหลังที่อยู่นอกดอก" ได้ง่าย
        เพราะพื้นหลังนอกดอกต่อถึงขอบภาพ ส่วนรูข้างในไม่ต่อถึงขอบ
        จึงเริ่มท่วมสีขาวจากมุมภาพ อะไรที่ท่วมถึง = พื้นหลังนอกดอก
        อะไรที่ยังดำอยู่หลังท่วมเสร็จ = รูข้างใน
    """

    # เติมขอบสีดำหนา 1 พิกเซลรอบภาพก่อน
    #
    # ทำไมต้องเติมขอบ
    #     ถ้าดอกไม้ชนขอบภาพพอดี มุม (0,0) จะเป็นสีขาวคือตัวดอก ไม่ใช่พื้นหลัง
    #     พอเริ่มท่วมจากมุมนั้น มันจะไปท่วมตัวดอกแทนที่จะท่วมพื้นหลัง แล้วผลกลับหัวทั้งภาพ
    #     ทดสอบกับ 204p_0009 มาแล้ว ถ้าไม่เติมขอบ จาก 52,716 พิกเซล
    #     กลายเป็น 73,905 พิกเซล คือขาวเกือบทั้งภาพ ใช้ไม่ได้เลย
    #     เติมขอบดำเข้าไปก่อน ทำให้มั่นใจว่ามุมภาพเป็นพื้นหลังแน่นอน
    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # ท่วมสีขาวจากมุมซ้ายบน ซึ่งตอนนี้เป็นขอบดำที่เพิ่งเติมเข้าไป
    flooded = padded_mask.copy()

    # floodFill บังคับว่าต้องส่งภาพช่วยจำเข้าไปด้วย และต้องใหญ่กว่าภาพจริงด้านละ 2 พิกเซล
    padded_height, padded_width = flooded.shape
    helper = np.zeros((padded_height + 2, padded_width + 2), np.uint8)
    cv2.floodFill(flooded, helper, (0, 0), 255)

    # ตอนนี้ flooded มีสีขาวทั้ง พื้นหลังนอกดอก และ ตัวดอก เหลือแค่รูข้างในที่ยังดำ
    # กลับสีขาวดำ จะเหลือเฉพาะรูเป็นสีขาว
    holes_only = cv2.bitwise_not(flooded)

    # เอารูมารวมกับ mask เดิม ด้วย OR = ที่ไหนขาวในอันใดอันหนึ่ง ให้เป็นขาว
    filled_mask = cv2.bitwise_or(padded_mask, holes_only)

    # ตัดขอบ 1 พิกเซลที่เติมเข้าไปตอนแรกออก ให้ภาพกลับมาขนาดเท่าเดิม
    # [1:-1] แปลว่า "เอาตั้งแต่ตัวที่ 1 ถึงตัวสุดท้ายแต่ไม่เอาตัวสุดท้าย" คือตัดหัวตัดท้าย
    return filled_mask[1:-1, 1:-1]


# ====================================================================
#  5. clean_mask() — เรียก 3 ขั้นตอนบนตามลำดับ
# ====================================================================

def clean_mask(mask):
    """
    ทำ Morphology ครบทั้ง 3 ขั้นตามลำดับที่ถูกต้อง

    รับ     : mask ดิบจาก step3
    ส่งกลับ : mask ที่ปรับปรุงแล้ว
    """

    mask = remove_small_spots(mask)
    mask = close_small_gaps(mask)
    mask = fill_holes(mask)

    return mask


# ====================================================================
#  6. main() — วนทำทั้ง 50 ภาพ
# ====================================================================

def main():
    """
    อ่าน mask จาก dataset/mask_threshold แล้วเขียนผลลง dataset/mask_morphology

    รับ     : ไม่รับอะไร
    ส่งกลับ : ไม่ส่งอะไรกลับ
    """

    os.makedirs(MORPHOLOGY_MASK_FOLDER, exist_ok=True)

    mask_files = sorted(os.listdir(THRESHOLD_MASK_FOLDER))

    print("ปรับปรุง mask ด้วย Morphology")
    print("    Opening  SE วงรีขนาด " + str(OPENING_SIZE))
    print("    Closing  SE วงรีขนาด " + str(CLOSING_SIZE))
    print("    Region Filling")
    print("")

    for file_name in mask_files:
        # ใส่ 0 ตัวหลัง แปลว่าอ่านเป็นภาพเทาช่องเดียว ไม่ต้องอ่านเป็นภาพสี 3 ช่อง
        mask = cv2.imread(os.path.join(THRESHOLD_MASK_FOLDER, file_name), 0)

        cleaned_mask = clean_mask(mask)

        cv2.imwrite(os.path.join(MORPHOLOGY_MASK_FOLDER, file_name), cleaned_mask)

        # บอกว่าพิกเซลขาวเปลี่ยนไปเท่าไหร่ จะได้เห็นว่า morphology ทำอะไรกับภาพนี้บ้าง
        before = (mask > 0).sum()
        after = (cleaned_mask > 0).sum()
        change = after - before
        print(file_name + "   " + str(before) + " -> " + str(after)
              + "   (เปลี่ยน " + str(change) + " พิกเซล)")

    print("")
    print("เสร็จแล้ว เขียน mask ไป " + str(len(os.listdir(MORPHOLOGY_MASK_FOLDER))) + " ไฟล์")
    print("เปิดโฟลเดอร์ dataset/mask_morphology เทียบกับ dataset/mask_threshold ดูได้เลย")


if __name__ == "__main__":
    main()
