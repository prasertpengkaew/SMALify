# เพิ่ม path ไปยังโฟลเดอร์ root เพื่อให้สามารถ import modules ได้จากโฟลเดอร์แม่
import sys
sys.path.append('../')

# นำเข้าไลบรารีพื้นฐานที่ใช้ในงานประมวลผลภาพและ deep learning
import numpy as np
import cv2
import torch
import imageio
from tqdm import tqdm  # แสดง progress bar ขณะวนลูป

# ฟังก์ชันตัดภาพเฉพาะบริเวณ silhouette
from utils import crop_to_silhouette
import os
import json

# สำหรับอ่านไฟล์ .csv แบบ dictionary
from csv import DictReader

# ไลบรารีจัดการภาพแบบ PIL
from PIL import Image, ImageFilter

# ไลบรารีของ COCO สำหรับ decode segmentation ที่เก็บในรูปแบบ RLE
from pycocotools.mask import decode as decode_RLE

# ไลบรารีสำหรับ copy object
from copy import copy

# ค่าคงที่ config ต่าง ๆ เช่น joint index, path
import config


def load_badja_sequence(BADJA_PATH, sequence_name, crop_size, image_range = None):
    # ประกาศตัวแปรเก็บข้อมูลทั้งหมด
    file_names = []
    rgb_imgs = []
    sil_imgs = []
    joints = []
    visibility = []

    # path ไปยังโฟลเดอร์ annotations ของ BADJA dataset
    annotations_path = os.path.join(BADJA_PATH, "joint_annotations")
    json_path = os.path.join(annotations_path, "{0}.json".format(sequence_name))

    # โหลดข้อมูล annotation จากไฟล์ .json
    with open(json_path) as json_data:
        sequence_annotation = np.array(json.load(json_data))

        # หากมีการกำหนดช่วงภาพ (image_range) ให้ตัดเฉพาะช่วงนั้น
        if image_range is not None:
            sequence_annotation = sequence_annotation[image_range]

        # วนลูปผ่านแต่ละภาพใน sequence
        for image_annotation in tqdm(sequence_annotation):
            # path ไปยังภาพ RGB และ segmentation mask
            file_name = os.path.join(BADJA_PATH, image_annotation['image_path'])
            seg_name = os.path.join(BADJA_PATH, image_annotation['segmentation_path'])

            # ตรวจสอบว่ามีไฟล์ครบหรือไม่
            if os.path.exists(file_name) and os.path.exists(seg_name):
                # ดึงตำแหน่ง joint ที่กำหนดใน config
                landmarks = np.array(image_annotation['joints'])[config.BADJA_ANNOTATED_CLASSES]

                # ดึงข้อมูล visibility (ความมั่นใจว่ามี joint นั้นจริง)
                visibility.append(np.array(image_annotation['visibility'])[config.BADJA_ANNOTATED_CLASSES])

                # โหลดภาพ RGB และ mask (segment) แล้ว normalize
                rgb_img = imageio.imread(file_name) / 255.0
                sil_img = imageio.imread(seg_name)[:, :, 0] / 255.0  # ช่องแรกของ mask

                # ปรับขนาด segmentation mask ให้ตรงกับ RGB
                rgb_h, rgb_w, _ = rgb_img.shape
                sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)

                # ครอปภาพให้อยู่ในบริเวณที่มี silhouette และ update joint
                sil_img, rgb_img, landmarks = crop_to_silhouette(sil_img, rgb_img, landmarks, crop_size)

                # เพิ่มข้อมูลเข้า list
                rgb_imgs.append(rgb_img)
                sil_imgs.append(sil_img)
                joints.append(landmarks)
                file_names.append(os.path.basename(image_annotation['image_path']))

            # แจ้งเตือนหากไม่มีไฟล์ segmentation หรือภาพ
            elif os.path.exists(file_name):
                print ("BADJA SEGMENTATION file path: {0} is missing".format(seg_name))
            else:
                print ("BADJA IMAGE file path: {0} is missing".format(file_name))

    # แปลงรายการทั้งหมดเป็น Tensor สำหรับการเทรนใน PyTorch
    rgb = torch.FloatTensor(np.stack(rgb_imgs, axis = 0)).permute(0, 3, 1, 2)  # [N, C, H, W]
    sil = torch.FloatTensor(np.stack(sil_imgs, axis = 0))[:, None, :, :]      # [N, 1, H, W]
    joints = torch.FloatTensor(np.stack(joints, axis = 0))                    # [N, num_joints, 2]
    visibility = torch.FloatTensor(np.stack(visibility, axis = 0).astype(np.float))  # [N, num_joints]

    # ปรับค่า visibility เป็น 0 สำหรับ joint ที่ไม่ได้ใช้ (index == -1)
    invalid_joints = np.array(config.BADJA_ANNOTATED_CLASSES) == -1
    visibility[:, invalid_joints] = 0.0
    
    return (rgb, sil, joints, visibility), file_names


def load_stanford_sequence(STANFORD_EXTRA, image_name, crop_size):
    # โฟลเดอร์เก็บภาพในชุดข้อมูล StanfordExtra
    img_dir = os.path.join(STANFORD_EXTRA, "sample_imgs")

    # ไฟล์ annotation json
    json_loc = os.path.join(STANFORD_EXTRA, "StanfordExtra_sample.json")

    # โหลด json เข้า memory
    with open(json_loc) as infile:
        json_data = json.load(infile)

    # แปลงเป็น dictionary ที่ใช้ path เป็น key เพื่อเข้าถึงข้อมูลได้เร็ว
    json_dict = {i['img_path']: i for i in json_data}
    
    def get_seg_from_entry(entry):
        """แปลงค่า RLE segmentation จาก json เป็น mask image (binary)"""
        rle = {
            "size": [entry['img_height'], entry['img_width']],
            "counts": entry['seg']
        }
        decoded = decode_RLE(rle)
        return decoded

    def get_dog(name):
        """โหลดข้อมูลภาพและ mask จากชื่อภาพ"""
        data = json_dict[name]
        img_data = imageio.imread(os.path.join(img_dir, data['img_path']))
        seg_data = get_seg_from_entry(data)
        data['img_data'] = img_data
        data['seg_data'] = seg_data
        return data

    # โหลดข้อมูลของภาพที่ต้องการ
    loaded_data = get_dog(image_name)

    # เพิ่ม dummy joint สำหรับ tail_mid (เพราะในชุดนี้ไม่มี)
    raw_joints = np.concatenate([
        np.array(loaded_data['joints']), [[0.0, 0.0, 0.0]]], axis = 0)

    # ครอปภาพและ segment พร้อมแปลง joint ให้อยู่ในกรอบใหม่
    sil_img, rgb_img, landmarks = crop_to_silhouette(
        loaded_data['seg_data'], loaded_data['img_data'] / 255.0, 
        raw_joints[:, [1, 0]], crop_size)

    # แปลงข้อมูลทั้งหมดเป็น Tensor สำหรับ PyTorch
    rgb = torch.FloatTensor(rgb_img)[None, ...].permute(0, 3, 1, 2)  # [1, 3, H, W]
    sil = torch.FloatTensor(sil_img)[None, None, ...]                # [1, 1, H, W]
    joints = torch.FloatTensor(landmarks)[:, :2].unsqueeze(0)        # [1, num_joints, 2]
    visibility = torch.FloatTensor(raw_joints)[:, -1].unsqueeze(0)   # [1, num_joints]
    file_names = [os.path.basename(loaded_data['img_path'])]         # ชื่อไฟล์ภาพ

    return (rgb, sil, joints, visibility), file_names

    
    