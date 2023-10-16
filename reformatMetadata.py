import json
import os
import numpy as np
import pickle as pkl
import shutil

local = True
if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/KITS19/"
else:
    root_folder = "/rds/general/user/kc2322/home/data/KITS19"

input_data_dir = os.path.join(root_dir, "data")
output_data_dir = os.path.join(root_dir, "FullDataset", "imagesTr")
output_label_dir = os.path.join(root_dir, "FullDataset", "labelsTr")


def reformatMetadata():
    # Reformat metadata from json to pickle
    f = open(os.path.join(root_dir, 'kits.json'))

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    case_id = []
    age = []
    gender = []
    bmi = []

    for sub in data:
        case_id.append(sub["case_id"])
        age.append(sub["age_at_nephrectomy"])
        gender.append(sub["gender"])
        bmi.append(sub["body_mass_index"])

    case_id = np.array(case_id)
    age = np.array(age)
    gender = np.array(gender)
    bmi = np.array(bmi)

    # convert gender to binary indicator
    gender_bin = np.zeros(gender.shape)
    gender_bin[gender == "female"] = 1

    print("Number of females: {}".format(np.sum(gender_bin)))
    print("Number of males: {}".format(gender_bin.shape[0] - np.sum(gender_bin)))

    metadata = {"id": case_id,
                "age": age,
                "gender": gender_bin,
                "bmi": bmi}

    # save
    f = open(os.path.join(root_dir, "metadata.pkl"), "wb")
    pkl.dump(metadata, f)
    f.close()

    print("Done")


def reformatFolderStructure():
    folders = os.listdir(input_data_dir)

    # rename and copy across the image and the label
    for folder in folders:
        id = folder[6:]
        image_name = "case_" + id + "0000.nii.gz"
        label_name = "case_" + id + ".nii.gz"

        # copy across the image to its new destination
        src = os.path.join(input_data_dir, folder, "imaging.nii.gz")
        dest = os.path.join(output_data_dir, image_name)
        shutil.copy(src, dest)

        # copy across the label to its new destination
        src = os.path.join(input_data_dir, folder, "segmentation.nii.gz")
        dest = os.path.join(output_label_dir, label_name)
        shutil.copy(src, dest)


def main():
    reformatMetadata()
    reformatFolderStructure()


if __name__ == "__main__":
    main()