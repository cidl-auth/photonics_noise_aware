# photonics_noise_aware
Noise-aware training for PNNs

# Introduction
This repository contains code that replicates the proposed noise-aware training described in "Noise-resilient and high-speed deep learning with coherent silicon photonics".

You can find a example experiment in [run_experiment.py](run_experiment.py), which demonstrates that taking into account hardware noise during the training can improve the inference results.

# Requirements

The code has been tested on a Ubuntu 22.04 system with Pytorch v1.11.0+cu102 and Python 3.10.4.

# Citation 

If you use this code in your work please cite the following paper:

<pre>
@article{dain,
  title={Noise-resilient and high-speed deep learning with coherent silicon photonics},
  author={G. Mourgias-Alexandris and M. Moralis-Pegios and A. Tsakyridis and S. Simos and G. Dabos and A.Totovic and N. Passalis and M. Kirtas and T. Rutirawut and F. Y. Gardes and A. Tefas and N. Pleros},
  journal={Nature Communications},
  year={2022}
}
</pre>

