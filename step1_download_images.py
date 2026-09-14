"""
Step 1 — โหลดภาพดอกทานตะวัน 50 ภาพจาก dataset PhotoArt50
โจทย์ข้อ 2 (Lecture 10 สไลด์หน้า 63): หา Dataset อย่างน้อย 50 ภาพ

ที่มา: https://github.com/BathVisArtData/PhotoArt50  คลาส 204.sunflower
ภาพติดลิขสิทธิ์ ให้ใช้เพื่อการศึกษาเท่านั้น จึงไม่เอาขึ้น git ใครอยากได้ให้รันไฟล์นี้

รัน: python step1_download_images.py
     ถ้าเน็ตหลุดกลางทาง รันซ้ำได้เลย ภาพที่โหลดแล้วจะถูกข้าม
"""

import os
import sys
import urllib.request


# raw.githubusercontent.com ให้ไฟล์ภาพตรงๆ (ถ้าใช้ github.com จะได้หน้าเว็บ HTML แทน)
BASE_URL = "https://raw.githubusercontent.com/BathVisArtData/PhotoArt50/master/204.sunflower"

# หาโฟลเดอร์ที่ไฟล์ .py นี้วางอยู่ ภาพจะได้ลงที่เดิมเสมอ ไม่ว่าจะรันจากโฟลเดอร์ไหน
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(PROJECT_FOLDER, "dataset", "images")

HOW_MANY_IMAGES = 50


def download_one_image(image_number):
    """
    โหลดภาพดอกทานตะวัน 1 ภาพ
    รับ     : image_number เลขภาพ 1 ถึง 50
    ส่งกลับ : ไม่มี (ได้ไฟล์ใหม่ใน dataset/images)
    """

    # zfill(4) เติม 0 ข้างหน้าให้ครบ 4 หลัก   เลข 1 -> "0001" -> 204p_0001.jpg
    file_name = "204p_" + str(image_number).zfill(4) + ".jpg"
    image_url = BASE_URL + "/" + file_name
    save_path = os.path.join(IMAGE_FOLDER, file_name)

    if os.path.exists(save_path):
        print("ข้าม   " + file_name + "  (มีอยู่แล้ว)")
        return

    # โหลดลงชื่อ .part ก่อน โหลดจบแล้วค่อยเปลี่ยนเป็นชื่อจริง
    # ถ้าเน็ตหลุดกลางทาง ไฟล์ครึ่งๆ จะค้างอยู่แค่ในชื่อ .part
    # ถ้าโหลดลงชื่อจริงตรงๆ รอบหน้าบรรทัด os.path.exists จะข้ามภาพเสียนั้นไปตลอด
    temp_path = save_path + ".part"
    urllib.request.urlretrieve(image_url, temp_path)
    os.replace(temp_path, save_path)

    print("โหลด   " + file_name)


def main():
    # ให้ print ภาษาไทยได้ แม้ส่ง output ลงไฟล์ (ตอนนั้น Windows ใช้ cp1252 ที่ไม่มีตัวอักษรไทย)
    sys.stdout.reconfigure(encoding="utf-8")

    # exist_ok=True = ถ้ามีโฟลเดอร์อยู่แล้วไม่ต้อง error
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    print("กำลังโหลดภาพลงโฟลเดอร์ " + IMAGE_FOLDER)
    print("")

    # range(1, 51) ได้เลข 1 ถึง 50
    for image_number in range(1, HOW_MANY_IMAGES + 1):
        download_one_image(image_number)

    # นับเฉพาะ .jpg ไม่นับไฟล์ .part ที่อาจค้างอยู่
    image_count = 0
    for file_name in os.listdir(IMAGE_FOLDER):
        if file_name.endswith(".jpg"):
            image_count = image_count + 1

    print("")
    print("เสร็จแล้ว มีภาพในโฟลเดอร์ทั้งหมด " + str(image_count) + " ภาพ")


if __name__ == "__main__":
    main()
