# SAFE
This code implements [SAFE: Similarity-Aware Multi-modal Fake News Detection](https://www.researchgate.net/publication/339873393_SAFE_Similarity-Aware_Multi-Modal_Fake_News_Detection) model.

## Resource 
### Dataset
We use [FakeNewsNet](https://arxiv.org/abs/1809.01286) dataset and provide our data in [this link](https://drive.google.com/drive/folders/1gSx4S9i6Haul4TQRkoNQtj3sRHVwGFQ3?usp=sharing). For the latest verision of FakeNewsNet, please directly check out: https://github.com/KaiDMML/FakeNewsNet.
### Image captioning tool
We use [Show and Tell](https://github.com/nikhilmaram/Show_and_Tell) to abstract the content of images.
### Word2vec embedding
We embed words use pre-trained word vectors [glove.840B.300d](https://github.com/stanfordnlp/GloVe) and the embedding tool [SIF](https://github.com/PrincetonML/SIF). The computation of glove.840B.300d word map is time-consuming, in order to provide more convenience we upload [(words, We)](https://drive.google.com/drive/folders/1yJSwmx7kpmEHvJ5OTt5mdF9FtFxs4Mqd?usp=sharing), which is the result of `data_io.getWordmap(wordfile)` in SIF. Please check **embedding** branch for the modified code and embedding results.


## Requirements
- Python 3.7
- TensorFlow 2.2
- xlwt
- nltk

## Getting Started

### Install requirements
```
pip install -r requirements.txt
```

### Train
```
python3 helper.py
python3 train.py
```


### Test
```
python3 test.py
```

## Citation
If you use this code for your research, please cite our [paper](https://www.researchgate.net/publication/339873393_SAFE_Similarity-Aware_Multi-Modal_Fake_News_Detection):
```
@inproceedings{zhou2020multimodal,
  title={SAFE: Similarity-Aware Multi-modal Fake News Detection},
  author={Zhou, Xinyi and Wu, Jindi and Zafarani, Reza},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={354--367},
  year={2020},
  organization={Springer}
}
```

## Contact
If you have any question, please contact zhouxinyi@data.syr.edu.


