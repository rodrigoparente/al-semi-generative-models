# Active Learning + Semi-supervised Generative Models

This repository contains experiments performed by combining the active learning labeling technique with several semi-supervised generative models, such as Variational Autoencoder [(ZHANG et al., 2019)](https://arxiv.org/abs/1905.02361v2), Auxiliary Deep Generative Model [(MAALØE et al., 2016)](https://arxiv.org/abs/1602.05473) and the combination of models of latent variables with discriminant models [(KINGMA et al., 2014)](https://arxiv.org/abs/1406.5298), better known such as M1 + M2. Our aim was to see if the use of such models would bring any benefit to the risk-based vulnerability management problem. However, none of these techniques showed a significant gain in the performance in relation to the supervised learning technique.

For more information about the active learning technique and the vulnerability risk classification problem, read the paper: [A Vulnerability Risk Assessment Methodology Using Active Learning](https://link.springer.com/chapter/10.1007/978-3-031-28451-9_15), published at the international conference on Advanced Information Networking and Applications. Also, with you would like to know more about the security dataset used in the experiments, you can check the [CVEJoin Security Dataset](https://github.com/rodrigoparente/cvejoin-security-dataset) repository.

## Requirements

Install requirements using the following command

```bash
$ pip install -r requirements.txt
```

## License

This project is [GNU GPLv3 licensed](./LICENSE).