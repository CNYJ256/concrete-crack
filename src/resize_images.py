import argparse
import os
from PIL import Image

def center_crop_to_square(img, target_size=1024):
    """
    等比缩放使最短边不小于 target_size，然后中心裁剪为 target_size x target_size
    """
    w, h = img.size
    # 如果任一边小于目标尺寸，先等比放大
    if w < target_size or h < target_size:
        scale = target_size / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h

    # 中心裁剪
    left = (w - target_size) // 2
    top = (h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    return img.crop((left, top, right, bottom))

def main():
    parser = argparse.ArgumentParser(
        description="将输入目录中的所有图片等比缩放并中心裁剪为 1024×1024 正方形"
    )
    parser.add_argument('--input_dir', required=True, help='输入图片目录')
    parser.add_argument('--output_dir', required=True, help='输出图片目录')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f'错误：输入目录不存在: {input_dir}')
        return

    os.makedirs(output_dir, exist_ok=True)

    # 支持的常见图片格式
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    for fname in os.listdir(input_dir):
        file_lower = fname.lower()
        if file_lower.endswith(extensions):
            input_path = os.path.join(input_dir, fname)
            try:
                img = Image.open(input_path).convert('RGB')
                img_cropped = center_crop_to_square(img, 1024)
                save_path = os.path.join(output_dir, fname)
                img_cropped.save(save_path)
                print(f'已处理: {fname}')
            except Exception as e:
                print(f'处理失败: {fname} —— {e}')

    print(f'\n完成！所有图片已保存到: {output_dir}')

if __name__ == '__main__':
    main()