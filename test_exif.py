import exifread
import subprocess
import json
# https://github.com/OpenDroneMap/ODM/blob/master/opendm/photo.py

file=r"C:\Users\emili\Documents\THESE\Images_drone\100_0001\100_0001_0043.JPG"
exiftool_path = r"C:\Users\emili\exiftool-13.45_64\exiftool.exe"

# exif_data = subprocess.check_output([exiftool_path,"-json","-Model","-EXIF:all",file])
exif_data = subprocess.check_output([
    exiftool_path,
    "-json",
    "-G",
    "-n",
    "-EXIF:all",
    "-XMP:all",
    "-MakerNotes:all",
    "-Composite:all",
    file
])

json_exif = json.loads(exif_data)
# print(f"{json_exif}")
print(json.dumps(json_exif, indent=4))




exit()



with open(file, 'rb') as f:
        tags = exifread.process_file(f, details=True, extract_thumbnail=False)
        for tag in tags:
            print(f"{tag}")

        try:
            if 'Image Make' in tags:
                try:
                    camera_make = tags['Image Make'].values
                    camera_make = camera_make.strip()
                except UnicodeDecodeError:
                    camera_make = "unknown"
            if 'Image Model' in tags:
                try:
                    camera_model = tags['Image Model'].values
                    camera_model = camera_model.strip()
                except UnicodeDecodeError:
                    camera_model = "unknown"
        except Exception as e:
            print(f"Cannot read extended EXIF tags for {file}: {str(e)}")

        print(f"{camera_make=} {camera_model=}")