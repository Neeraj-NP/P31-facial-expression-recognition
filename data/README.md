# Data layout

Datasets are not committed. Put them here:

```
data/
├── fer2013/                 # Kaggle "msambare/fer2013" (image-folder version)
│   ├── train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg
│   └── test/{...same...}/*.jpg
└── rafdb/                   # RAF-DB "basic" (request access from the authors)
    ├── Image/aligned/*_aligned.jpg
    ├── EmoLabel/list_patition_label.txt
    └── Annotation/manual/*_manu_attri.txt   # gender / race / age
```

Download FER-2013:
```
kaggle datasets download -d msambare/fer2013 -p data/fer2013 --unzip
```

RAF-DB: http://www.whdeng.cn/raf/model1.html — email the authors for the password.
