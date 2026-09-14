"""
Step 2 — ระบาย Ground truth ด้วยมือ ทีละภาพ
โจทย์ข้อ 2 (Lecture 10 สไลด์หน้า 63): Ground truth (ถ้าไม่มี ให้สร้างขึ้นมาเอง)
PhotoArt50 มีให้แค่กรอบสี่เหลี่ยม ไม่มี mask รายพิกเซล จึงต้องระบายเอง

กติกาของ mask (step2 step3 step4 ใช้ตรงกัน)
    ขาว 255 = ดอกทานตะวัน = Positive Class
    ดำ  0   = พื้นหลัง     = Negative Class
    ให้ระบายทับ "ตัวดอก" รวมเกสรตรงกลางด้วย ห้ามระบายพื้นหลัง

ห้ามให้โปรแกรม threshold ช่วยระบายให้ก่อน
    เพราะเฉลยจะกลายเป็นคำตอบของโปรแกรมเอง แล้ววัดผลได้คะแนนดีปลอมๆ

วิธีใช้
    ลากเมาส์ซ้าย = ระบายดอก     ลากเมาส์ขวา = ลบ     [ ] = ขนาดหัวแปรง
    c = ล้างภาพนี้   s = เซฟแล้วไปภาพถัดไป   n = ข้ามไปก่อน   q = ออก
    ถ้ากดปุ่มแล้วไม่มีอะไรเกิดขึ้น ลองสลับแป้นพิมพ์เป็นภาษาอังกฤษ
    ภาพที่เซฟแล้วจะถูกข้ามในรอบหน้า ถ้าจะระบายใหม่ ให้ลบไฟล์ใน dataset/ground_truth/ ก่อน

รัน: python step2_make_ground_truth.py   (ต้องรัน step1 ก่อน)
"""

import os
import sys

import cv2
import numpy as np


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "images")
GROUND_TRUTH_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "ground_truth")

WINDOW_NAME = "ground truth"

# ภาพจริงสูงแค่ 177-533 px ระบายไม่ถนัด จึงขยายให้สูง 600 px ตอนแสดงบนจอ
# (mask ที่เซฟยังเป็นขนาดจริงของภาพ)  600 ยังใส่จอโน้ตบุ๊ก 1366x768 ได้
DISPLAY_HEIGHT = 600

PAINT_COLOR = 255   # ขาว = ดอกทานตะวัน
ERASE_COLOR = 0     # ดำ  = พื้นหลัง

START_BRUSH_SIZE = 12
SMALLEST_BRUSH = 2
BIGGEST_BRUSH = 60

# แบ่งงาน 2 คน: คนแรกตั้ง 1 (ระบายภาพ 1-25)  คนที่สองตั้ง 26 (ระบายภาพ 26-50)
START_FROM_IMAGE = 1


# ตัวแปรส่วนกลาง 3 ตัว
# OpenCV เป็นคนเรียก handle_mouse(event, x, y, flags, param) เอง เราเพิ่มช่องส่ง mask เข้าไปไม่ได้
# จึงวางไว้ข้างนอก ให้ทุก def ในไฟล์มองเห็นร่วมกัน
current_mask = None      # mask ของภาพที่กำลังระบาย
current_scale = 1.0      # ภาพบนจอขยายจากขนาดจริงกี่เท่า
brush_size = START_BRUSH_SIZE


def draw_brush(screen_x, screen_y, color):
    """
    ระบายวงกลม 1 วงลง current_mask ตรงที่เมาส์ชี้
    รับ     : screen_x, screen_y ตำแหน่งเมาส์บนจอ   color 255 = ระบาย  0 = ลบ
    ส่งกลับ : ไม่มี (current_mask ถูกแก้)
    """

    # ตำแหน่งบนจอเป็นของภาพที่ขยายแล้ว ต้องหารกลับเป็นตำแหน่งบนภาพจริง
    mask_x = int(screen_x / current_scale)
    mask_y = int(screen_y / current_scale)

    # -1 = ระบายทึบทั้งวง
    cv2.circle(current_mask, (mask_x, mask_y), brush_size, color, -1)


def handle_mouse(event, x, y, flags, param):
    """
    OpenCV เรียกฟังก์ชันนี้เองทุกครั้งที่เมาส์ขยับหรือถูกกดบนหน้าต่าง
    รับ     : event เหตุการณ์  x y ตำแหน่ง  flags ปุ่มที่กดค้างอยู่  param ไม่ได้ใช้
    ส่งกลับ : ไม่มี
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        draw_brush(x, y, PAINT_COLOR)

    elif event == cv2.EVENT_RBUTTONDOWN:
        draw_brush(x, y, ERASE_COLOR)

    elif event == cv2.EVENT_MOUSEMOVE:
        # flags เป็นเลขก้อนเดียวที่รวมสถานะทุกปุ่มไว้  & คือถามว่า "ปุ่มซ้ายอยู่ในก้อนนี้ไหม"
        # ห้ามจำสถานะปุ่มไว้ในตัวแปรเอง: ถ้าปล่อยปุ่มนอกหน้าต่าง ตัวแปรจะค้าง แล้วระบายเองทั้งที่ไม่ได้กด
        if flags & cv2.EVENT_FLAG_LBUTTON:
            draw_brush(x, y, PAINT_COLOR)
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            draw_brush(x, y, ERASE_COLOR)


def make_display(image, mask, file_name, image_number, total_images):
    """
    สร้างภาพสำหรับแสดงบนจอ = ภาพจริง + สีเขียวโปร่งแสงตรงที่ระบายไว้
    รับ     : image ภาพจริง  mask ที่ระบายไว้  ที่เหลือเอาไว้เขียนบอกบนจอ
    ส่งกลับ : ภาพสีขนาดที่ขยายแล้ว
    """

    display_width = int(image.shape[1] * current_scale)
    display_height = int(image.shape[0] * current_scale)

    big_image = cv2.resize(image, (display_width, display_height))

    # INTER_NEAREST = ขยายโดยก๊อปค่าเดิม ห้ามเกลี่ย
    # ถ้าเกลี่ย ขอบ mask จะมีค่ากลางๆ โผล่มา แล้วสีเขียวบนจอจะกว้างกว่าที่ระบายจริง
    big_mask = cv2.resize(mask, (display_width, display_height),
                          interpolation=cv2.INTER_NEAREST)

    # ผสมภาพจริงกับสีเขียวครึ่งต่อครึ่ง แล้วแปะเฉพาะจุดที่ระบายไว้ (big_mask > 0)
    green_layer = np.zeros(big_image.shape, np.uint8)
    green_layer[:] = (0, 255, 0)
    mixed = cv2.addWeighted(big_image, 0.5, green_layer, 0.5, 0)

    display = big_image.copy()
    display[big_mask > 0] = mixed[big_mask > 0]

    # วงแดงมุมซ้ายบน = ขนาดหัวแปรงตอนนี้
    cv2.circle(display, (40, 70), int(brush_size * current_scale), (0, 0, 255), 2)

    # cv2.putText เขียนภาษาไทยไม่ได้ ข้อความบนจอจึงเป็นภาษาอังกฤษ
    label = file_name + "   (" + str(image_number) + "/" + str(total_images) + ")   brush=" + str(brush_size)
    cv2.putText(display, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(display, "L=paint FLOWER  R=erase  [ ]=size  c=clear  s=save  n=skip  q=quit",
                (10, display_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return display


def label_one_image(file_name, image_number, total_images):
    """
    เปิดภาพ 1 ภาพให้ระบาย แล้ววนรอปุ่มจนกว่าจะกด s n หรือ q
    รับ     : file_name ชื่อไฟล์ภาพ  image_number ภาพที่เท่าไหร่  total_images ทั้งหมดกี่ภาพ
    ส่งกลับ : "saved" / "skipped" / "quit"
    """

    global current_mask, current_scale, brush_size

    image = cv2.imread(os.path.join(IMAGE_FOLDER, file_name))

    # เริ่มจาก mask ดำล้วนขนาดเท่าภาพ (np.uint8 = เก็บค่า 0-255)
    current_mask = np.zeros(image.shape[:2], np.uint8)

    # ขยายให้ทุกภาพสูงเท่ากันบนจอ
    current_scale = DISPLAY_HEIGHT / image.shape[0]

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    while True:
        display = make_display(image, current_mask, file_name, image_number, total_images)
        cv2.imshow(WINDOW_NAME, display)

        # รอปุ่ม 20 มิลลิวินาที แล้ววนไปวาดจอใหม่
        # & 0xFF ตัดให้เหลือแค่ 8 บิตท้ายที่เป็นรหัสปุ่ม (tutorial ของ OpenCV ก็เขียนแบบนี้)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            # ยังไม่ได้ระบายเลย (mask ดำล้วน ค่ามากสุดคือ 0) ห้ามเซฟ
            # ถ้าเซฟไป รอบหน้าโปรแกรมจะข้ามภาพนี้ไปตลอด ทั้งที่เฉลยว่างเปล่า
            if current_mask.max() == 0:
                print("ยังไม่ได้ระบายดอกใน " + file_name + " จึงยังไม่เซฟ")
                continue

            # เซฟเป็น png เพราะ jpg บีบอัดแล้วค่า 255 อาจเพี้ยนเป็น 254 ทำให้นับ TP FP ผิด
            mask_name = file_name.replace(".jpg", ".png")
            cv2.imwrite(os.path.join(GROUND_TRUTH_FOLDER, mask_name), current_mask)
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


def main():
    # ให้ print ภาษาไทยได้เสมอ (เหตุผลอยู่ใน step1)
    sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(GROUND_TRUTH_FOLDER, exist_ok=True)

    # เอาเฉพาะ .jpg กันไฟล์ .part (เน็ตหลุด) และ .DS_Store (Mac) ปนมา
    # ไฟล์พวกนั้น cv2.imread อ่านไม่ได้ แล้วโปรแกรมจะพัง
    image_files = []
    for file_name in sorted(os.listdir(IMAGE_FOLDER)):
        if file_name.endswith(".jpg"):
            image_files.append(file_name)

    total_images = len(image_files)

    print("วิธีใช้")
    print("    ลากเมาส์ซ้าย   ระบายทับดอกทานตะวัน (ไม่ใช่พื้นหลัง)")
    print("    ลากเมาส์ขวา    ลบส่วนที่ระบายเกิน")
    print("    [ ]            เปลี่ยนขนาดหัวแปรง")
    print("    c              ล้างภาพนี้เริ่มใหม่")
    print("    s              เซฟแล้วไปภาพถัดไป")
    print("    n              ข้ามภาพนี้ไปก่อน")
    print("    q              ออกจากโปรแกรม")
    print("")

    saved_count = 0
    image_number = 0

    for file_name in image_files:
        image_number = image_number + 1

        # ภาพก่อนหมายเลขที่ตั้งไว้ เป็นส่วนของอีกคน ข้ามไป
        if image_number < START_FROM_IMAGE:
            continue

        # ภาพที่เคยเซฟแล้ว ข้ามไป จะได้ระบายต่อจากที่ค้างไว้
        mask_path = os.path.join(GROUND_TRUTH_FOLDER, file_name.replace(".jpg", ".png"))
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

    # นับไฟล์เฉลยจริงในโฟลเดอร์ (เฉพาะ .png) ว่าเหลืออีกกี่ภาพ
    done = 0
    for file_name in os.listdir(GROUND_TRUTH_FOLDER):
        if file_name.endswith(".png"):
            done = done + 1

    print("")
    print("ตอนนี้มี ground truth แล้ว " + str(done) + " ภาพ จากทั้งหมด " + str(total_images) + " ภาพ")

    if done < total_images:
        print("ยังเหลืออีก " + str(total_images - done) + " ภาพ รันไฟล์นี้ใหม่เพื่อทำต่อได้เลย")


if __name__ == "__main__":
    main()
