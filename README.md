# adversarial_seizure_detection
## Title: Adversarial representation learning for robust patient-independent epileptic seizure detection 

**PDF: [J-BHI](https://ieeexplore.ieee.org/abstract/document/8994148), [arXiv](https://arxiv.org/abs/1909.10868)**

**Authors: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Lina Yao](https://www.linayao.com/) (lina.yao@unsw.edu.au), Manqing Dong, Zhe Liu, Yu Zhang, Yong Li**

## Overview
This repository contains reproducible codes for the proposed adversarial_seizure_detection model.  
In this paper, we propose a novel and generic deep learning framework aiming at patient-independent epileptic seizure diagnosis. The proposed approach refines the seizure-specific representation by eliminating the inter-subject noise through adversarial training. Moreover, we involve the attention mechanism to learn the contribution of each EEG channel in the epileptic seizure detection, which empowers our method with great explainability. Please check our paper for more details on the algorithm.


## Citing
If you find our work useful for your research, please consider citing this paper:

    @article{zhang2020adversarial,
      title={Adversarial representation learning for robust patient-independent epileptic seizure detection},
      author={Zhang, Xiang and Yao, Lina and Dong, Manqing and Liu, Zhe and Zhang, Yu and Li, Yong},
      journal={IEEE journal of biomedical and health informatics},
      volume={24},
      number={10},
      pages={2852--2859},
      year={2020},
      publisher={IEEE}
    }

## Datasets
The cleaned dataset all_14sub.p (894 Mb) is too large to be uploaded here. Please download the original dataset at the TUH Corpus website (https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) and clean by yourself.



## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang.alan.zhang@gmail.com>.


## License

This repository is licensed under the MIT License.
