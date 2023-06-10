# Cross-modality Data Augmentation for Sign Language Translation

This repo contains the training and evaluation code for the paper [XmDA: Cross-modality Data Augmentation for End-to-End
Sign Language Translation](https://arxiv.org/pdf/2305.11096.pdf). 

* This code is based on [Sign Language Transformers](https://github.com/neccam/slt) but modified to realize Cross-modality KD and  Cross-modality mix-up. 
* For baseline end-to-end SLT, you can use the [Sign Language Transformers](https://github.com/neccam/slt). 
* For gloss-to-text tearchers model, you can follow the [PGen](https://github.com/Atrewin/PGen) or use the original text-to-text [Joey NMT](https://github.com/joeynmt/joeynmt) framework. 
* For put them to one, we expend Sign Language Transformers framework with Joey NMT and allow the new one can forward gloss-to-text and mix-to-text (i.e., ``forward_type`` in [sign, gloss, mixup]).


## Requirements
* Create environment follow [Sign Language Transformers](https://github.com/neccam/slt).
* Reproduce [PGen](https://github.com/Atrewin/PGen) to obtain multi-references as sentence-level guidance from gloss-to-text teachers (or using ``forward_type`` = gloss).
* Reproduce [SMKD](https://github.com/ycmin95/VAC_CSLR) to pre-process the sign video.
* Pre-process dataset and put them into ``./data/DATA-NAME/`` (ref the format to https://github.com/neccam/slt)


## Usage

  `python -m signjoey train_XmDA configs/Sign_XmDA.yaml` 

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
## ToDo:

- [X] *Initial code release.*
- [ ] Release Pre-process dataset.
- [ ] Share extensive qualitative and quantitative results & config files to generate them.


## Reference

Please cite the paper below if you use this code in your research:

    @article{ye2023cross,
      title={Cross-modality Data Augmentation for End-to-End Sign Language Translation},
      author={Ye, Jinhui and Jiao, Wenxiang and Wang, Xing and Tu, Zhaopeng and Xiong, Hui},
      journal={arXiv preprint arXiv:2305.11096},
      year={2023}
    }


