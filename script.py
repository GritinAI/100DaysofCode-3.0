import os
import shutil

# nums = list(range(26, 100))
#
# nums = [str(i) for i in nums]
#
# nums = [('0'+i if len(i) < 3 else i) for i in nums]
#
# print(nums)
#
# for f in nums:
#     try:
#         os.makedirs(f'Days/Day_{f}')
#     except:
#         pass


with open("assets/100DaysofCodeContent.md") as file:
    content = file.read()

lines = content.split("---\n")

# print(lines)

def format_line(line):
    line = line.strip()
    line = "# " + line
    split_line = line.split("\n\n")
    split_line[1] = "## " + split_line[1]

    day = split_line[0].strip().split(" ")[-1].strip()

    prefix = f"![100 days of code Day {day}](../../Images/Day{day}.png)\n\n"

    split_line[2] = prefix + split_line[2]

    # split_line[2] = split_line[2].replace("\!", "\!\n").replace(
    #     f"Welcome to Day {day} of the 100 Days of Code challenge!",
    #     f"Welcome to Day {day} of the 100 Days of Code challenge!" + "\n\n\n"
    # )

    split_line[2] = split_line[2].replace("\\!", "!\n")

    return day, "\n\n".join(split_line) + "\n\n"

def save_content_for_day(day, content):
    day = str(day)
    formatted_day = '0' + day if len(day) < 3 else day

    with open(f"Days/Day_{formatted_day}/README.md", "w") as file:
        file.write(content)

    print(f"Content saved for day {day}!")

    return


def extract_images(path_to_extract_to="Images/", path_to_extract_from="assets/canva_images.zip"):
    import zipfile

    file_names = os.listdir(path_to_extract_to)
    file_names = list(filter(lambda p: os.path.splitext(p)[0].isnumeric(), file_names))

    for file_name in file_names:
        if not file_name.startswith("Day"):
            print(file_name)
            continue
        else:
            file_path = path_to_extract_to + file_name
            os.remove(file_path)

    with zipfile.ZipFile(path_to_extract_from, mode="r") as file:
        file.extractall(path_to_extract_to)

    print("All images extracted!")

    file_names = os.listdir(path_to_extract_to)
    file_names = list(filter(lambda p: os.path.splitext(p)[0].isnumeric(), file_names))

    img_file_paths = [path_to_extract_to + f for f in file_names]
    new_img_file_paths = [path_to_extract_to + "Day"+ f for f in file_names]

    for old, new in zip(img_file_paths, new_img_file_paths):
        os.rename(old, new)

    return


if __name__ == "__main__":
    # index = 0
    # day, content = format_line(lines[index])
    #
    # print(content)

    # print(lines[index])
    # print(lines[index].split("\n\n"))

    for line in lines:
        day, content = format_line(line)
        save_content_for_day(day, content)