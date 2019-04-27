import pandas as pd
from fastai.vision import *
from fastai.metrics import accuracy

path = "data/nabirds/"

images = pd.read_csv(path + '/images.txt', sep=" ",
                     header=None, names=['file', 'path'])
images['cat_num'] = images['path'].str.split('/').str[0]

# import classes to get common_names
classes = pd.read_table(path + '/classes.txt', delimiter=None)
classes.columns = ['code']
classes[['cat_num', 'common_name']
        ] = classes['code'].str.split(" ", 1, expand=True)
classes = classes.drop(['code'], axis=1)
classes['cat_num'] = classes['cat_num'].str.zfill(
    4)  # fill missing leading zeros

# merge common_name from classes into images df
images = pd.merge(images, classes,  how='left', on='cat_num')

# import train_test_split and merge with images to get the train_test split provided with dataset
split = pd.read_csv(path + '/train_test_split.txt', sep=" ",
                    header=None, names=['file', 'train_test'])
images = pd.merge(images, split,  how='left', on='file')

tfms = get_transforms(flip_vert=True, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.1)
data = ImageList.from_df(images, path=path + '/images', cols='path').split_by_rand_pct(
).label_from_df(cols='common_name').transform(tfms, size=224).databunch(bs=64)

learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(5)

learn.lr_find()
learn.recorder.plot()

learn.save('birdv1')
learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-7, 3e-6))

learn.save('birdv2')
learn.export('birdwatchin.pkl')

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(6, figsize=(16, 16))
