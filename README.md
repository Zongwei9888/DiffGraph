# HDL
The code will coming soon!
>Heterogeneous Diffusion Graph Learning
![model](./HDL.jpg)
## Abstract
Social recommendation has emerged as a valuable means to improve personalized recommendation for its strength to enhance user representation learning by leveraging the social connections among users, such as following and friend relations observed in online social platforms. The key assumption of social recommendation is the homophily in preference patterns for socially-connected users. That is, users connected by social ties share similar taste in user-item activities such as rating and purchasing. However, this assumption does not always hold due to the presence of irrelevant and false social ties, which can instead pollute the user embeddings and harm the recommendation accuracy.
To address this issue, we propose a novel diffusion-based social denoising framework, called Social Diffusion-based Recommender model. A simple yet effective hidden-space diffusion is designed for noise removal in the compressed and dense representation space. Through the multi-step noise diffusion and removal, our model is endowed with a strong noise identification and elimination ability for the encoded user representations with different noise degrees. The diffusion module is optimized in a downstream task-aware manner to maximally facilitate the recommendation target.
Extensive experiments have been conducted for evaluating the efficacy of this framework. Results demonstrate its superiority in recommendation accuracy, training efficiency, and denoising effectiveness.
## Environment
- python=3.8
- torch=1.12.1
- numpy=1.23.1
- scipy=1.9.1
- dgl=1.0.2+cu113
