import os
import shutil


def set_new_file_path(new_file_path):
    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)


def get_all_files(file_path):
    all_files = os.listdir(file_path)
    return all_files


def set_new_path_list(new_file_path):
    new_path_list = []
    for i in range(3):
        path = os.path.join(new_file_path, str(i))
        if not os.path.exists(path):
            os.mkdir(path)
        new_path_list.append(path)
    return new_path_list


def remove_small_files(all_files, file_path, new_path_list):
    file_num = len(all_files)
    count = 0

    for file_name in all_files:
        size = os.path.getsize(os.path.join(file_path, file_name)) / 1024
        # print('{:.2f} KB'.format(size))
        if size <= 100:
            os.remove(os.path.join(file_path, file_name))
            print('{} is washed'.format(file_name))
            count += 1
        else:
            label = file_name[-5]
            if label == '0':
                shutil.move(os.path.join(file_path, file_name), new_path_list[0])
                count += 1
                print('{} is moved to 0'.format(file_name))
            elif label == '1':
                shutil.move(os.path.join(file_path, file_name), new_path_list[1])
                count += 1
                print('{} is moved to 1'.format(file_name))
            elif label == '2':
                shutil.move(os.path.join(file_path, file_name), new_path_list[2])
                count += 1
                print('{} is moved to 2'.format(file_name))

    fail_num = file_num - count
    print('\nall:{}\nsuccess:{}\nfailed:{}'.format(file_num, count, fail_num))


if __name__ == '__main__':
    """
    total_file_path : original data's location, like 0083,0085
    root_new_file_path: washed data's location
    """

    total_file_path = r'F:\experiment\data_original'
    root_new_file_path = r'F:\experiment\data_wash'
    file_paths = get_all_files(total_file_path)  # 得到类似0085的单独文件夹

    for file_path in file_paths:
        new_file_path = os.path.join(root_new_file_path, file_path)
        set_new_file_path(new_file_path)

        file_path_temp = os.path.join(total_file_path,file_path)
        all_files = get_all_files(file_path_temp)

        new_path_list = set_new_path_list(new_file_path)
        remove_small_files(all_files, file_path, new_path_list)
