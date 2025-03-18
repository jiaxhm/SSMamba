# SSMamba: Superpixel Segmentation with Mamba

Abstract—Deep convolutional networks have achieved remark
able success in superpixel segmentation. However, they only focus
 on local features ignoring global attributes. The visual Mamba
 demonstrates an exceptional capability to capture long-range
 dependencies and offers a lower computational cost compared
 to the Transformer. Building on this inspiration, we propose a
 novel superpixel segmentation with Mamba, termed SSMamba.
 In SSMamba, Mamba is integrated into a global-local archi
tecture, enabling efficient interaction between global attributes
 and local features to produce high-quality superpixels. The
 designed activation function further enhances the effectiveness
 of SSMamba. Extensive experiments demonstrate that SSMamba
 achieves superior performance on multiple public datasets, in
 terms of quantitative metric and visual comparison.
 
 Index Terms—Convolutional neural network, Mamba, super
pixel segmentation

🔖 We have submitted the paper to the IEEE Signal Processing Letters

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

# Cite US

Please cite us if this work is helpful to you.

# Acknowledgments

The basic code is partially from the below repos.
1. VMamba (https://github.com/MzeroMiko/VMamba)
2. SCN (https://github.com/fuy34/superpixel_fcn)
3. ESNet (DOI: 10.1109/TCSVT.2023.3347402)
