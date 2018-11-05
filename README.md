This uses Guntenberg books as source data, and generated the next words. In this example I used ~60 Erotic novels from last century.

The model is a pytorch implementation of OpenAI's Finetuned Transformer Language Model, with pretrained weights.

Example of some of the tamer outputs:


    Base: i want you to want 

    Result: i want you to want to comply with an object which strictly dictates its action. I must have you to myself for a few moments this evening. understand that anything must be accomplished with a bare minimum of intimacy. go to your room and get some sleep. tomorrow morning I will notify you of my orders. " 
    I obeyed. the old baw d, who was beginning to look at me strangely, made an impulsive appeal to comfort and lust, declaring that she had been dreaming. mabel thus roused me up like a ravenous beast, and swelling with lust, surged forward, said that she must have had a dream, and begged me to lie,


    Base: i will never 

    Result: i will never lay hold of any of it for all eternity. " 
    his feelings were real and complete as he meditated and communicated his thoughts to his solicitor, for in spite of his original feeling of despair when he had lost mary, there was a renewed kind of love of the old man and he was willing to not allow him to be bought. there was now a separation between him and her ; she would be moved to florence and he would be able to help her. 
    dominating himself with all the frustrations and struggles of the moment, which still dared not abandon him, he furiously tore into his possession an antique watch which had belonged to

    Base: i will never 

    Result: i will never give him back, that the memory of that ro bot like monster has wrought so much on his daily life, I can not countenance a mere indiscretion that can be forgotten ; and then I refuse to deprive him of the enjoyment of being my own, even if I shall have of putting it out of my mind. 
    " after the present absence everybody can scarce wait until the day after next, which follows -- the full moon rising, the young man who shares my bed with miss frank land and others, and who has then, in a scheme of male gratification, seized and animated my person and got into bed with me.

    Base: panting, he ripped at her bodice 

    Result: panting, he ripped at her bodice. then suddenly, as if the fiend had torn off his own shirt, he covered her eyes with it ! 
    and suddenly, in a rage, he went downstairs and left her to her admirers. 
    it is not difficult to recall dur tal's last terrifying scene in the drawing - room. it was about fifty years ago, when his sympathies were exhausted and he had to sell his property and his house for the bulk of it. now, six years later, he was in a state of complete dread of any change in her character. if he would only give her an ounce of this plain ness...? " any god kind sort. -- - hardly too hardly rarely hardly on al, all the's, on on they heavy. 

    Base: an extraordinarily long neck 

    Result: an extraordinarily long neck began and ended in a little rounded nostrils, and plat ea thed a pleasing shape to meet, in fact, the north fair travelers, who were, as gar ru lous and stag ger ingly lewd as their riders were lewd. miss frank land made an effort not to notice them and made a polite but discreet curtsy, which, although not without distinction, were some measure of its graceful sway. at the same time, she seemed intensely interested in its movement. mrs. eth eri dge, too, had, indeed, the same curious air. when their company, most of them young, and some quite young, took to one

    Base: an extraordinarily long neck  

    Result: an extraordinarily long neck, a broad bust, a well made bust, which he placed on his desk without ceremony : every part of me ar dently desiring it should tremendously fall into the hands of some men, who were of the imagination themselves most interested in the beauties before them. but they were evidently careful not to leave my room. so excit able was the curiosity of them, as they did not wish to compromise my repose, or to disturb the repose which they were made to be in. I was no understand of taste either. yet the idea aroused my inner inquisiti veness to the utmost extent that frustrated me all the more by the appalling retreat hardly about


# PyTorch implementation of OpenAI's Finetuned Transformer Language Model

This is a PyTorch implementation of the [TensorFlow code](https://github.com/openai/finetune-transformer-lm) provided with OpenAI's paper ["Improving Language Understanding by Generative Pre-Training"](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.

This implementation comprises **a script to load in the PyTorch model the weights pre-trained by the authors** with the TensorFlow implementation.

![Transformer Language Model](assets/ftlm.png)

The model classes and loading script are located in [model_pytorch.py](model_pytorch.py).

The names of the modules in the PyTorch model follow the names of the Variable in the TensorFlow implementation. This implementation tries to follow the original code as closely as possible to minimize the discrepancies.

This implementation thus also comprises a modified Adam optimization algorithm as used in OpenAI's paper with:
- fixed weights decay following the work of [Loshchilov et al.](https://arxiv.org/abs/1711.05101), and
- scheduled learning rate as [commonly used for Transformers](http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer).

## Requirements
To use the model it-self by importing [model_pytorch.py](model_pytorch.py), you just need:
- PyTorch (version >=0.4)

To run the classifier training script in [train.py](train.py) you will need in addition:
- tqdm
- sklearn
- spacy
- ftfy
- pandas

You can download the weights of the OpenAI pre-trained version by cloning [Alec Radford's repo](https://github.com/openai/finetune-transformer-lm) and placing the `model` folder containing the pre-trained weights in the present repo.

```bash
python -m spacy download en
```

## Fine-tuning the pre-trained model on a classification task

Use the train.ipynb notebook

