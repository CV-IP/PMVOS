# PMVOS: Pixel-Level Matching-Based Video Object Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pmvos-pixel-level-matching-based-video-object/video-object-segmentation-on-youtube-vos)](https://paperswithcode.com/sota/video-object-segmentation-on-youtube-vos?p=pmvos-pixel-level-matching-based-video-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pmvos-pixel-level-matching-based-video-object/visual-object-tracking-on-davis-2016)](https://paperswithcode.com/sota/visual-object-tracking-on-davis-2016?p=pmvos-pixel-level-matching-based-video-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pmvos-pixel-level-matching-based-video-object/visual-object-tracking-on-davis-2017)](https://paperswithcode.com/sota/visual-object-tracking-on-davis-2017?p=pmvos-pixel-level-matching-based-video-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pmvos-pixel-level-matching-based-video-object/semi-supervised-video-object-segmentation-on-1)](https://paperswithcode.com/sota/semi-supervised-video-object-segmentation-on-1?p=pmvos-pixel-level-matching-based-video-object)

#### Suhwan Cho, Heansung Lee, Sungmin Woo, Sungjun Jang, Sangyoun Lee


---
### Download
[[arXiv_paper]](https://arxiv.org/pdf/2009.08855.pdf)


[[DAVIS 2017]](https://davischallenge.org/davis2017/code.html)


[[pre-trained model]](https://drive.google.com/file/d/189Vx0ow8LZ3bQowH3Tgiu3WIe7swur1f/view?usp=sharing)


[[pre-computed results]](https://drive.google.com/file/d/1Tl7XVjY0SlAVUdhIx9inRLRmkgwUuwxs/view?usp=sharing)


---
### Usage
1. Modify 'davis_path' in 'local_config.py'.

2. Check fps.
```
python test.py --fps
```

3. Save the results.
```
python test.py --save
```

---
### Others
DAVIS_2016.txt, DAVIS_2017.txt, DAVIS_fps.txt: Accuracies and speeds on DAVIS datasets.

We use a single GeForce RTX 2080 Ti GPU.


---
### Citation
```
@article{cho2020pmvos,
  title={PMVOS: Pixel-Level Matching-Based Video Object Segmentation},
  author={Cho, Suhwan and Lee, Heansung and Woo, Sungmin and Jang, Sungjun and Lee, Sangyoun},
  journal={arXiv preprint arXiv:2009.08855},
  year={2020}
}
```

---
### Acknowledgement
This code is built on [joakimjohnander/agame-vos](https://github.com/joakimjohnander/agame-vos).
