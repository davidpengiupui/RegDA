# RegDA

## Simple Version of - [Regressive Domain Adaptation for Unsupervised Keypoint Detection (RegDA, CVPR 2021)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/regressive-domain-adaptation-cvpr21.pdf)

## Experiment and Results



```shell script
# Train a RegDA on RHD -> H3D task using PoseResNet.
# Assume you have put the datasets under the path `data/RHD` and  `data/H3D_crop`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python regda.py data/RHD data/H3D_crop -s RenderedHandPose -t Hand3DStudio --seed 0 --debug --log logs/regda/rhd2h3d
```

## Citation


```latex
@misc{dalib,
  author = {Junguang Jiang, Baixu Chen, Bo Fu, Mingsheng Long},
  title = {Transfer-Learning-library},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thuml/Transfer-Learning-Library}},
}
```

```
@InProceedings{RegDA,
    author    = {Junguang Jiang and
                Yifei Ji and
                Ximei Wang and
                Yufeng Liu and
                Jianmin Wang and
                Mingsheng Long},
    title     = {Regressive Domain Adaptation for Unsupervised Keypoint Detection},
    booktitle = {CVPR},
    year = {2021}
}

```
