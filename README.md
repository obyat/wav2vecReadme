<h1 align="center">Welcome to WavBERT: Exploiting Semantic and Non-semantic Speech using Wav2vec and BERT for Dementia Detection 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="http://www.homepages.ed.ac.uk/sluzfil/ADReSSo-2021/" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://github.com/kefranabg/readme-md-generator/graphs/commit-activity" target="_blank">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>
</p>

> In this project, we exploit semantic and non-semantic information from patient’s speech data usingWav2vec and Bidirectional Encoder Representations from Transformers (BERT) for dementia detection. We first propose a basic WavBERT model by extracting semantic information from speech data using Wav2vec, and analyzing the semantic information using BERT for dementia detection. While the basic model discards the non-semantic information, we propose extended WavBERT models that convert the output ofWav2vec to the input to BERT for preserving the non-semantic information in dementia detection. Specifically, we determine the locations and lengths of inter-word pauses using the number of blank tokens from Wav2vec where the threshold for setting the pauses is automatically generated via BERT. We further design a pre-trained embedding conversion network that converts the output embedding of Wav2vec to the input embedding of BERT, enabling the fine-tuning of WavBERT with non-semantic information. Our evaluation results using the ADReSSo dataset showed that the WavBERT models achieved the highest accuracy of 83.1% in the classification task, the lowest Root-Mean-Square Error (RMSE) score of 4.44 in the regression task, and a mean F1 of 70.91% in the progression task. We confirmed the effectiveness of WavBERT models exploiting both semantic and non-semantic speech.

### 🏠 [Homepage](https://github.com/billzyx/wav2vec)

## Run tests

```sh
.....
```

## Author

👤 **Youxiang Zhu**

* Website: http://www.faculty.umb.edu/xiaohui.liang/mobcp.html
* GitHub: [@ billzyx ](https://github.com/billzyx )

## Author

👤 **Abdelrahman Obyat**

* Website: https://www.linkedin.com/in/abdelrahman-obyat-52065b173/
* GitHub: [@ obyat ](https://github.com/obyat)

## Author

👤 **Xiaohui Liang**

* Website: http://faculty.umb.edu/xiaohui.liang/
* Website: https://www.linkedin.com/in/xiaohui-liang-7622a419/


## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
