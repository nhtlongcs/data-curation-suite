# /mnt/10TBHDD/allie/Deakin/
# | ID*/
# |--20**
#    |--mm/
#       |--dd/
#          |--ID*.jpg

# out -> ID mm dd.txt - list of all image paths
import os
from pathlib import Path
from typing import List


if __name__ == "__main__":
    base_dir = Path("/mnt/10TBHDD/allie/Deakin/")
    # person_dir: ID***
    # year_dir: 20**
    # month_dir: ** (int, prefix 0)
    # day_dir: ** (int, prefix 0)
    all_image_paths: List[Path] = []
    for person_dir in base_dir.glob("ID*"):

        if not person_dir.is_dir():
            continue
        
        for year_dir in person_dir.glob("20*"):
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir(): 
                if not month_dir.is_dir():
                    continue
                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    image_paths = list(day_dir.glob("ID*.jpg"))
                    all_image_paths.extend(image_paths)

    # save to ID*_yyyy_mm_dd.txt -> all paths
    # ID101_20220310_172608_000.jpg -> ID101_20220310.txt
    # sort path by name
    all_image_paths.sort()
    out_dir = 'deakin_txt_lists'
    os.makedirs(out_dir, exist_ok=True)
    for image_path in all_image_paths:
        image_name = image_path.stem  # ID101_20220310_172608_000
        parts = image_name.split("_")
        if len(parts) < 2:
            continue
        id_part = parts[0]  # ID101
        date_part = parts[1]  # 20220310
        year = date_part[:4]
        month = date_part[4:6]
        day = date_part[6:8]
        out_filename = f"{id_part}_{year}{month}{day}.txt"
        out_filepath = os.path.join(out_dir, out_filename)
        with open(out_filepath, "a") as f:
            f.write(str(image_path) + "\n")