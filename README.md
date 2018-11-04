This uses Guntenberg books as source data, and generated the next words. In this example I used ~60 Erotic novels from last century.

The model is a pytorch implementation of OpenAI's Finetuned Transformer Language Model, with pretrained weights.

Example outputs:


    Base: rosy buttocks 

    Result: rosy buttocks but noted that his fingers had been already working him into a post mor tem, a replacement for his earlier foray with that unspeakable je ering creature. 
    " vince ? " 
    it was his niece, jessica, who had just reached lunch time. she eyed vince warily when she arrived, quickly averting her gaze when he glanced her way. he moved to victoria's side, but she stopped him with a slight gesture. 
    " eat with me in costume, darling, " she urged, and then recalled the phone call this morning from emily and all the others at the supreme court. nobody had was meeting here after the inquisition, according to the files on the table. there was no way to avoid revealing her visit. 
    vince nodded his acknowledgment



    Base: I want you 

    Result: i want you to know in case i'm wrong and you'll show up soon. " 
    " why do you say that ? " he said. " do you, and everybody in the world, think there's something wrong ? " 
    she took a drink. " i'll get to that later. listen, i'm sorry for taking you away from your father, " she said. " colleen's a great person, but it's a lot to put on your shoulders, and we already know that. okay, she's a tough lady, you know her. she struggles, she's worried about the ranch, but her love for them gives them what they need. they let her go after a long absence... in her own way, anyway



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

