import shutil
import os
import random


def generate_cv_folder(cv_i, root, save_path, train_subjects, val_subjects, test_subjects):

    cv_folder = os.path.join(save_path, "cross_val" + str(cv_i))

    if not os.path.exists(cv_folder):
        os.mkdir(cv_folder)
        for split in ["train", "val", "test"]:
            os.mkdir(os.path.join(cv_folder, split))
            for folder in ["images", "labels"]:
                os.mkdir(os.path.join(cv_folder, split, folder))

    save_path = os.path.join(cv_folder)

    subjects_names = os.listdir(root)
    for subject in subjects_names:

        if subject in train_subjects:
            save_folder = os.path.join(save_path, "train")
        elif subject in val_subjects:
            save_folder = os.path.join(save_path, "val")
        elif subject in test_subjects:
            save_folder = os.path.join(save_path, "test")
        else:
            raise ValueError("Subject not associated to any split " + subject)

        label_list = os.listdir(os.path.join(root, subject, "label"))

        for i, label in enumerate(label_list):

            subject_id = label.split("-labels")[0]
            img_number = label.split("-labels")[1].strip(".png")

            image_path = os.path.join(root, subject, "US",
                                      subject_id + img_number + ".png")
            image_save_path = os.path.join(save_folder, "images", subject_id + "_" + img_number + ".png")

            label_path = os.path.join(root, subject, "label", label)
            label_save_path = os.path.join(save_folder, "labels", subject_id + "_" + img_number + "_label.png")

            shutil.copyfile(image_path, image_save_path)  # copyfile(src, dst)
            shutil.copyfile(label_path, label_save_path)  # copyfile(src, dst)


def main():
    root = "D:\\NAS\\images\\images"
    save_path = "D:\\NAS\\BoneSegmentation\\linear_probe"
    test_subjects = ["Elisa", "MatthiasS"]
    test_split_percentage = 0.8
    cross_val_folders = 5

    subject_list = os.listdir(root)
    subject_list = [item for item in subject_list if item not in test_subjects]
    num_train_subjects = int(test_split_percentage * len(subject_list))

    for i in range(cross_val_folders):
        train_subjects = random.sample(population=subject_list,
                                       k=num_train_subjects)

        val_subjects = [item for item in subject_list if item not in train_subjects]

        generate_cv_folder(cv_i=i,
                           root=root,
                           save_path=save_path,
                           train_subjects=train_subjects,
                           val_subjects=val_subjects,
                           test_subjects=test_subjects)


if __name__ == '__main__':
    main()
