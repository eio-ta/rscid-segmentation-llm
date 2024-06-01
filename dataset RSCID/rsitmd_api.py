# Update in 2021.10.28
# Several sentences were updated to correct blank

import json

def load_from_json(filename):
    with open(filename) as f:
        infos = json.load(f)
    return infos

def analyse_infos(infos):
    # keys: ['images', 'dataset']
    print("===============================================")
    keys = infos.keys()
    print("All_keys: {}\n".format(keys))

    # key - dataset
    print("===============================================")
    print("Dataset name: {}\n".format(infos['dataset']))

    # key - images
    print("===============================================")
    print("Lens of images: {}".format(len(infos['images'])))
    train_num = 0
    for i in range(len(infos['images'])):
        if infos['images'][i]['split'] == "train":
            train_num += 1
    print("Lens of train images: {}\n".format(train_num))

    # example
    print("===============================================")
    print("Example:")
    for k,v in infos['images'][0].items():
        print("key:",k,"   value:",v)

if __name__=="__main__":
    infos = load_from_json("dataset_RSITMD.json")
    analyse_infos(infos)