# SSMamba: Superpixel Segmentation with Mamba

Abstract—Deep convolutional networks have achieved remarkable success in superpixel segmentation. However, they only focus on local features ignoring global attributes. The visual Mamba demonstrates an exceptional capability to capture long-range dependencies and offers a lower computational cost compared to the Transformer. Building on this inspiration, we propose a novel superpixel segmentation with Mamba, termed SSMamba. In SSMamba, Mamba is integrated with a global-local architecture, enabling efficient interaction between global attributes and local features to produce high-quality superpixels. The designed activation function further enhances the effectiveness of SSMamba. Extensive experiments demonstrate that SSMamba achieves state-of-the-art performance across multiple public datasets, in terms of quantitative metrics and visual comparison.
 
 Index Terms—Convolutional neural network, Mamba, super
pixel segmentation

![Fig1](https://github.com/user-attachments/assets/08485668-68ac-48b4-8153-9b8e14d4f6a5)
Overall framework of the proposed SSMamba. In the first stage, we employ CNN and Mamba to compute global attention. In the second stage, similar to the first stage, we use CNN and Mamba to generate local attention. The superpixel head, guided by the edge head, generates the final superpixels by integrating features from both stages using the FFU. In FFU, w_1 and w_2 are learnable weights.

✅ We have submitted the paper to the IEEE Signal Processing Letters (2024/12/08)

✅ We have updated the code (2025/03/18)

# ✨ Getting Start

# Environment Installation

Reference to VMamba (https://github.com/MzeroMiko/VMamba) and SCN (https://github.com/fuy34/superpixel_fcn)

# Preparing Dataset
1. BSDS500: Following this link: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
2. NYUDv2: Following this link: http://vcl.ucsd.edu/hed/nyu/
3. KITTI: Following this link: http://www.cvlibs.net/datasets/kitti/
4. DRIVE: Following this link: https://drive.grand-challenge.org/

Furthermore, Preprocessing of BSDS500 training data Following SCN (https://github.com/fuy34/superpixel_fcn)

# Training
1. Stage 1 Global Modeling was first trained using VMamba's pre-training weights.
2. Next, Stage 2 Local Modeling is trained using Stage 1 Global Modeling and VMamba's pre-training weights.

    Run `python main.py` to start the program.

   ✨ It's worth mentioning that SSMamba is trained exclusively on the BSDS500 training set and directly generates superpixels for NYUv2, KITTI, and DRIVE without requiring fine-tuning.

# Testing
1. Test BSDS500: Please run `test_bsds.py`
2. Test NYUDv2: Please run `test_nyu.py`
3. Test KITTI: Please run `test_kitti.py`
4. Test DRIVE: Please run `test_drive.py`

# Weights
We have placed each of the three weights (VMamba, Global Modeling, and Local Modeling) in the https://pan.baidu.com/s/1f0BU5w4NP0TI_iecK3Zr1g password: 94y8

# Result

We tested the ASA, BR-BP, CO, and UE metrics (refer: https://doi.org/10.1016/j.cviu.2017.03.007) on the four datasets and their CSV results are in the https://pan.baidu.com/s/1f0BU5w4NP0TI_iecK3Zr1g password: 94y8

And, to facilitate the comparison of other good work with our approach, we also publish eval files for four datasets in the `./eval_result`

## 📚 Cite Us

Please cite us if this work is helpful to you ✨

```bibtex
@ARTICLE{10960337,
  author={Jia, Xiaohong and Li, Yonghui and Jiao, Jianjun and Zhao, Yao and Xia, Zhiwei},
  journal={IEEE Signal Processing Letters}, 
  title={SSMamba: Superpixel Segmentation With Mamba}, 
  year={2025},
  volume={32},
  number={},
  pages={1715-1719},
  keywords={Feature extraction;Head;Convolutional neural networks;Image edge detection;Computational modeling;Visualization;Transformers;Training;Data mining;Computational efficiency;Convolutional neural network;mamba;superpixel segmentation},
  doi={10.1109/LSP.2025.3559425}
}


# Acknowledgments

The basic code is partially from the below repos.
1. VMamba (https://github.com/MzeroMiko/VMamba)
2. SCN (https://github.com/fuy34/superpixel_fcn)
3. ESNet (DOI: 10.1109/TCSVT.2023.3347402)
