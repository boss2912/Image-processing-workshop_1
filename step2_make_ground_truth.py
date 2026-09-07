"""
Step 2 — ระบาย Ground truth ด้วยมือ ทีละภาพ
วิชา 310-3311 Image Processing / Workshop ท้าย Lecture 10

โจทย์ (สไลด์หน้า 63)
    ข้อ 2  หา Dataset อย่างน้อย 50 ภาพ และ Ground truth (ถ้าไม่มี ให้สร้างขึ้นมาเอง)
           -> dataset PhotoArt50 มีให้แต่ "กรอบสี่เหลี่ยม" ไม่ใช่ mask ระดับพิกเซล
              เราจึงต้องสร้าง Ground truth เองตามที่โจทย์อนุญาตไว้

Ground truth คืออะไร
    คือ "เฉลย" ที่บอกทีละพิกเซลว่าตรงนั้นเป็นดอกทานตะวันหรือพื้นหลัง
    สีขาว (255) = ดอกทานตะวัน = Positive Class
    สีดำ  (0)   = พื้นหลัง     = Negative Class
    step5 กับ step6 จะเอาเฉลยนี้ไปเทียบกับผลของโปรแกรม เพื่อนับ TP FP FN TN

ทำไมต้องระบายเอง ทำไมไม่ให้โปรแกรมเดาให้ก่อน
    เพราะสิ่งที่เรากำลังจะวัด คือความแม่นของ threshold สีเหลือง
    ถ้าเอา threshold สีเหลืองมาสร้างเฉลยเสียเอง เท่ากับเอาคำตอบไปเป็นเฉลยของตัวเอง
    วัดแล้วได้คะแนนดีปลอมๆ ตอบครูไม่ได้

วิธีใช้
    ลากเมาส์ปุ่มซ้าย   ระบายทับดอกทานตะวัน (เพิ่มพื้นที่สีขาว)
    ลากเมาส์ปุ่มขวา    ลบส่วนที่ระบายเกิน
    กด [ หรือ ]        หัวแปรงเล็กลง / ใหญ่ขึ้น
    กด c               ล้างที่ระบายไว้ทั้งหมดของภาพนี้ เริ่มใหม่
    กด s               เซฟแล้วไปภาพถัดไป
    กด n               ข้ามภาพนี้ไปก่อน ยังไม่เซฟ
    กด q               ออกจากโปรแกรม (ที่เซฟไปแล้วไม่หาย)

    ภาพไหนที่เซฟแล้ว โปรแกรมจะข้ามให้อัตโนมัติในการรันครั้งถัดไป
    ถ้าอยากระบายภาพไหนใหม่ ให้ลบไฟล์ของภาพนั้นใน dataset/ground_truth/ ทิ้งก่อน

ไฟล์นี้มี 6 def แต่ละอันทำงานอย่างเดียว เรียงตามลำดับที่ถูกเรียกใช้จริง
    1. draw_brush()        ระบายวงกลม 1 จุดลงบน mask
    2. handle_mouse()      แปลการกดเมาส์ ว่าจะให้ draw_brush ระบายหรือลบ
    3. make_display()      เอาภาพต้นฉบับกับ mask มาซ้อนกันให้เห็นบนจอ
    4. save_mask()         เซฟ mask ลงไฟล์
    5. label_one_image()   คุมการระบาย 1 ภาพ จนกว่าจะกด s หรือ n
    6. main()              ไล่เปิดทีละภาพจนครบ 50

รัน:  python step2_make_ground_truth.py
      (ต้องรัน step1 ให้เสร็จก่อน ไม่งั้นจะไม่มีภาพให้ระบาย)
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

IMAGE_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "images")
GROUND_TRUTH_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "ground_truth")

WINDOW_NAME = "ground truth"

# ภาพใน dataset สูงแค่ 177-533 px ถ้าแสดงขนาดจริงจะเล็กมาก ระบายไม่ถนัด
# จึงขยายให้สูงเท่านี้ตอนแสดงบนจอ แต่ mask ที่เซฟยังเป็นขนาดจริงของภาพเสมอ
#
# ทำไมเป็น 600 ไม่ใช่ 700
#     ภาพที่เตี้ยที่สุดคือ 204p_0045.jpg สูงแค่ 177 px ถูกขยายมากที่สุด
#     ถ้าตั้ง 700 หน้าต่างจะกลายเป็น 1123x700 บวกแถบหัวหน้าต่างกับ task bar
#     แล้วสูงเกินจอโน้ตบุ๊ก 1366x768 ซึ่งเป็นจอที่เพื่อนอาจใช้
#     ตั้ง 600 หน้าต่างใหญ่สุดเหลือ 963x600 ใส่ได้ทุกจอ
#
# ถ้าจอใครใหญ่และอยากระบายละเอียดกว่านี้ แก้เลขบรรทัดล่างนี้บรรทัดเดียวพอ
DISPLAY_HEIGHT = 600

PAINT_COLOR = 255   # สีขาว = ดอกทานตะวัน
ERASE_COLOR = 0     # สีดำ  = พื้นหลัง

START_BRUSH_SIZE = 12
SMALLEST_BRUSH = 2
BIGGEST_BRUSH = 60

# ให้เริ่มระบายจากภาพหมายเลขไหน
#
# ใช้ตอนแบ่งงานกัน 2 คน
#     คนที่ทำครึ่งแรก  ตั้งเป็น 1   ระบายภาพ 0001 ถึง 0025
#     คนที่ทำครึ่งหลัง ตั้งเป็น 26  ระบายภาพ 0026 ถึง 0050
#
# ทำไมต้องมีเลขนี้ ทำไมไม่กด n ข้ามเอา
#     โปรแกรมข้ามให้อัตโนมัติเฉพาะภาพที่ "มีเฉลยอยู่ในเครื่องนี้แล้ว"
#     แต่เฉลยของอีกคนยังอยู่ในเครื่องเขา ยังไม่ได้ push ขึ้น git มาให้เรา
#     ถ้าไม่มีเลขนี้ คนที่ทำครึ่งหลังต้องกด n รัว 25 ครั้ง ทุกครั้งที่เปิดโปรแกรม
#     แล้วถ้าเผลอกดเกินไป 1 ครั้ง จะข้ามภาพที่ตัวเองต้องทำโดยไม่รู้ตัว
START_FROM_IMAGE = 1


# ตัวแปรส่วนกลาง 5 ตัว
#
# ทำไมต้องใช้ตัวแปรส่วนกลาง
#     OpenCV บังคับว่า function ที่รับเหตุการณ์เมาส์ ต้องมีหน้าตาแบบนี้เป๊ะๆ
#         handle_mouse(event, x, y, flags, param)
#     เราเพิ่มช่องรับ mask เข้าไปเองไม่ได้ เพราะ OpenCV เป็นคนเรียก function นี้ ไม่ใช่เรา
#     จึงต้องเอา mask มาวางไว้ข้างนอกให้ทั้งไฟล์มองเห็นร่วมกันแทน
current_mask = None      # mask ของภาพที่กำลังระบายอยู่ตอนนี้
current_scale = 1.0      # ภาพบนจอ ถูกขยายจากขนาดจริงกี่เท่า
brush_size = START_BRUSH_SIZE
is_painting = False      # ตอนนี้กดเมาส์ซ้ายค้างอยู่ไหม
is_erasing = False       # ตอนนี้กดเมาส์ขวาค้างอยู่ไหม


# ====================================================================
#  1. draw_brush() — ระบายวงกลม 1 จุดลงบน mask
# ====================================================================

def draw_brush(screen_x, screen_y, color):
    """
    ระบายวงกลม 1 วง ลงบน mask ตรงตำแหน่งที่เมาส์ชี้อยู่

    รับ     : screen_x, screen_y ตำแหน่งเมาส์ "บนจอ"
              color สีที่จะระบาย  255 คือเพิ่ม  0 คือลบ
    ส่งกลับ : ไม่ส่งอะไรกลับ แต่ current_mask จะถูกแก้
    """

    # ตำแหน่งเมาส์ที่ได้มา เป็นตำแหน่งบนภาพที่ขยายแล้ว
    # แต่ mask เป็นขนาดจริงของภาพ ต้องหารกลับด้วยอัตราขยายก่อน ไม่งั้นจะระบายผิดที่
    mask_x = int(screen_x / current_scale)
    mask_y = int(screen_y / current_scale)

    # -1 ตรงท้าย แปลว่า "ระบายทึบทั้งวง" ถ้าใส่เลขบวกจะได้เป็นเส้นขอบวงกลมแทน
    cv2.circle(current_mask, (mask_x, mask_y), brush_size, color, -1)


# ====================================================================
#  2. handle_mouse() — แปลการกดเมาส์ ว่าจะให้ระบายหรือลบ
# ====================================================================

def handle_mouse(event, x, y, flags, param):
    """
    ถูกเรียกโดย OpenCV ทุกครั้งที่เมาส์ขยับหรือถูกกด บนหน้าต่างของเรา

    รับ     : event เหตุการณ์ที่เกิด, x y ตำแหน่งเมาส์
              flags กับ param ไม่ได้ใช้ แต่ต้องมีไว้ให้ครบตามที่ OpenCV กำหนด
    ส่งกลับ : ไม่ส่งอะไรกลับ
    """

    global is_painting, is_erasing

    # กดปุ่มซ้ายลง = เริ่มระบาย และระบายจุดแรกทันที
    if event == cv2.EVENT_LBUTTONDOWN:
        is_painting = True
        draw_brush(x, y, PAINT_COLOR)

    # ปล่อยปุ่มซ้าย = หยุดระบาย
    elif event == cv2.EVENT_LBUTTONUP:
        is_painting = False

    # กดปุ่มขวาลง = เริ่มลบ
    elif event == cv2.EVENT_RBUTTONDOWN:
        is_erasing = True
        draw_brush(x, y, ERASE_COLOR)

    elif event == cv2.EVENT_RBUTTONUP:
        is_erasing = False

    # เมาส์ขยับ ต้องเช็คก่อนว่ากำลังกดปุ่มค้างอยู่ไหม
    # ถ้าไม่เช็ค แค่เลื่อนเมาส์ผ่านเฉยๆ ก็จะระบายไปด้วย
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_painting:
            draw_brush(x, y, PAINT_COLOR)
        elif is_erasing:
            draw_brush(x, y, ERASE_COLOR)


# ====================================================================
#  3. make_display() — เอาภาพต้นฉบับกับ mask มาซ้อนกันให้เห็นบนจอ
# ====================================================================

def make_display(image, mask, file_name, image_number, total_images):
    """
    สร้างภาพที่จะเอาไปแสดงบนจอ = ภาพต้นฉบับ + สีเขียวโปร่งแสงตรงที่ระบายไว้

    รับ     : image ภาพต้นฉบับ, mask ที่ระบายไว้ตอนนี้
              file_name, image_number, total_images เอาไว้เขียนบอกบนจอ
    ส่งกลับ : ภาพสีขนาดเท่าที่จะแสดงบนจอ
    """

    display_width = int(image.shape[1] * current_scale)
    display_height = int(image.shape[0] * current_scale)

    big_image = cv2.resize(image, (display_width, display_height))
    big_mask = cv2.resize(mask, (display_width, display_height))

    # ทำภาพสีเขียวล้วนขนาดเท่ากัน แล้วผสมกับภาพจริงครึ่งต่อครึ่ง
    # ที่ต้องโปร่งแสง เพราะถ้าทับทึบจะมองไม่เห็นกลีบดอกข้างใต้ แล้วระบายไม่ตรงขอบ
    green_layer = np.zeros(big_image.shape, np.uint8)
    green_layer[:] = (0, 255, 0)
    mixed = cv2.addWeighted(big_image, 0.5, green_layer, 0.5, 0)

    display = big_image.copy()

    # big_mask > 0 คือการเลือก "เฉพาะพิกเซลที่ระบายไว้"
    # บรรทัดนี้แปลว่า เอาสีที่ผสมแล้ว ไปแปะเฉพาะจุดที่ระบายไว้เท่านั้น จุดอื่นไม่แตะ
    display[big_mask > 0] = mixed[big_mask > 0]

    # วาดวงกลมบอกขนาดหัวแปรงไว้มุมซ้ายบน จะได้รู้ว่าตอนนี้แปรงใหญ่แค่ไหน
    cv2.circle(display, (40, 70), int(brush_size * current_scale), (0, 0, 255), 2)

    # cv2.putText วาดภาษาไทยไม่ได้ เพราะ OpenCV ไม่มีฟอนต์ไทยมาให้
    # ข้อความบนจอจึงเป็นภาษาอังกฤษ ส่วนคำอธิบายภาษาไทยพิมพ์ออกทางเทอร์มินัลแทน
    label = file_name + "   (" + str(image_number) + "/" + str(total_images) + ")   brush=" + str(brush_size)
    cv2.putText(display, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(display, "L=paint  R=erase  [ ]=size  c=clear  s=save  n=skip  q=quit",
                (10, display_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return display


# ====================================================================
#  4. save_mask() — เซฟ mask ลงไฟล์
# ====================================================================

def save_mask(mask, file_name):
    """
    เซฟ mask เป็นไฟล์ png ในโฟลเดอร์ ground_truth ใช้ชื่อเดียวกับภาพต้นฉบับ

    รับ     : mask ที่ระบายเสร็จแล้ว, file_name ชื่อไฟล์ภาพต้นฉบับ เช่น 204p_0001.jpg
    ส่งกลับ : ไม่ส่งอะไรกลับ
    """

    # เปลี่ยนนามสกุลจาก .jpg เป็น .png
    # เซฟเป็น png เพราะ jpg เป็นการบีบอัดแบบมีการสูญเสีย
    # ค่า 255 ที่เซฟลงไป ตอนอ่านกลับมาอาจกลายเป็น 251 หรือ 254 ได้
    # ซึ่งทำให้การนับ TP FP ใน step5 เพี้ยน  ส่วน png เก็บค่าเดิมเป๊ะ
    mask_name = file_name.replace(".jpg", ".png")
    cv2.imwrite(os.path.join(GROUND_TRUTH_FOLDER, mask_name), mask)


# ====================================================================
#  5. label_one_image() — คุมการระบาย 1 ภาพ จนกว่าจะกด s หรือ n
# ====================================================================

def label_one_image(file_name, image_number, total_images):
    """
    เปิดภาพ 1 ภาพขึ้นมาให้ระบาย แล้ววนรอรับปุ่มจนกว่าผู้ใช้จะสั่งจบ

    รับ     : file_name ชื่อไฟล์ภาพ, image_number ภาพที่เท่าไหร่, total_images ทั้งหมดกี่ภาพ
    ส่งกลับ : "saved"   ระบายเสร็จและเซฟแล้ว
              "skipped" ข้ามภาพนี้ไป
              "quit"    ผู้ใช้สั่งออกจากโปรแกรม
    """

    global current_mask, current_scale, brush_size

    image = cv2.imread(os.path.join(IMAGE_FOLDER, file_name))

    # เริ่มจาก mask ดำล้วนขนาดเท่าภาพ  np.uint8 คือชนิดข้อมูลที่เก็บค่า 0-255
    current_mask = np.zeros(image.shape[:2], np.uint8)

    # ขยายภาพให้สูงเท่ากันทุกภาพ ภาพเตี้ยจะถูกขยายมากกว่าภาพสูง
    current_scale = DISPLAY_HEIGHT / image.shape[0]

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    while True:
        display = make_display(image, current_mask, file_name, image_number, total_images)
        cv2.imshow(WINDOW_NAME, display)

        # รอปุ่ม 20 มิลลิวินาที แล้ววาดจอใหม่ ทำให้เห็นรอยระบายตามเมาส์ทันที
        #
        # & 0xFF ต่อท้ายเพื่ออะไร
        #     waitKey คืนค่าเป็นตัวเลขที่บาง OS ใส่ข้อมูลอื่นพ่วงมาข้างหน้าด้วย
        #     & 0xFF คือการตัดให้เหลือแค่ 8 บิตท้าย ซึ่งเป็นรหัสตัวอักษรที่เรากด
        #     ถ้าไม่ใส่ เครื่องบอสอาจใช้ได้ แต่เครื่อง Mac ของเพื่อนกดปุ่มแล้วไม่ตอบสนอง
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            save_mask(current_mask, file_name)
            return "saved"

        if key == ord("n"):
            return "skipped"

        if key == ord("q"):
            return "quit"

        if key == ord("c"):
            current_mask[:] = 0

        if key == ord("["):
            brush_size = max(SMALLEST_BRUSH, brush_size - 2)

        if key == ord("]"):
            brush_size = min(BIGGEST_BRUSH, brush_size + 2)


# ====================================================================
#  6. main() — ไล่เปิดทีละภาพจนครบ
# ====================================================================

def main():
    """
    ไล่เปิดภาพทีละภาพให้ระบาย ข้ามภาพที่เคยเซฟไปแล้ว

    รับ     : ไม่รับอะไร
    ส่งกลับ : ไม่ส่งอะไรกลับ
    """

    os.makedirs(GROUND_TRUTH_FOLDER, exist_ok=True)

    image_files = sorted(os.listdir(IMAGE_FOLDER))
    total_images = len(image_files)

    print("วิธีใช้")
    print("    ลากเมาส์ซ้าย   ระบายทับดอกทานตะวัน")
    print("    ลากเมาส์ขวา    ลบส่วนที่ระบายเกิน")
    print("    [ ]            เปลี่ยนขนาดหัวแปรง")
    print("    c              ล้างภาพนี้เริ่มใหม่")
    print("    s              เซฟแล้วไปภาพถัดไป")
    print("    n              ข้ามภาพนี้ไปก่อน")
    print("    q              ออกจากโปรแกรม")
    print("")

    saved_count = 0

    for image_number, file_name in enumerate(image_files, start=1):

        # ภาพก่อนหมายเลขที่ตั้งไว้ ไม่ใช่ส่วนของเรา ข้ามไปเลย
        if image_number < START_FROM_IMAGE:
            continue

        mask_path = os.path.join(GROUND_TRUTH_FOLDER, file_name.replace(".jpg", ".png"))

        # ภาพที่เคยเซฟแล้ว ข้ามไปเลย จะได้ระบายต่อจากที่ค้างไว้ได้
        if os.path.exists(mask_path):
            saved_count = saved_count + 1
            continue

        result = label_one_image(file_name, image_number, total_images)

        if result == "quit":
            print("ออกจากโปรแกรม")
            break

        if result == "saved":
            saved_count = saved_count + 1
            print("เซฟแล้ว  " + file_name + "   (ทำไปแล้ว " + str(saved_count) + " ภาพ)")

        if result == "skipped":
            print("ข้าม     " + file_name)

    cv2.destroyAllWindows()

    # นับไฟล์จริงในโฟลเดอร์ เพื่อยืนยันว่าเหลืออีกกี่ภาพ
    done = len(os.listdir(GROUND_TRUTH_FOLDER))
    print("")
    print("ตอนนี้มี ground truth แล้ว " + str(done) + " ภาพ จากทั้งหมด " + str(total_images) + " ภาพ")

    if done < total_images:
        print("ยังเหลืออีก " + str(total_images - done) + " ภาพ รันไฟล์นี้ใหม่เพื่อทำต่อได้เลย")


if __name__ == "__main__":
    main()
