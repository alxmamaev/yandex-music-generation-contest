# Transformer ABC notes genenerator
This reposetory contains solution for Yandex music generation contest. 
More about this solution you can read in my blog at medium.


[Link to article](https://alxmamaev.medium.com/generating-music-with-ai-or-transformers-go-brrrr-3a3ac5a04126)


# Data
You may to download data with abc notes from [google drive](https://drive.google.com/drive/folders/15rNfd10B2yEab-67CG5VAyVjvolJN-E4?usp=sharing), and unpack in project directory. 
## Training
Firstly wee need to train tokenizer.

```
python3 train_tokenizer trainset/abc abc.yttm
```

Then make some cleaning of the data

```
python3 clean_data.py trainset/abc cleaned_data
```

And starts training the tokenizer

```
python3 train.py cleaned_data
```
You may to setup some parameters, like gradient accamulation, batch size and e.t.c.

## Generation 
```
python3 generate.py testset/abc ABCModel/checkpoint-3/pytorch_model.bin
``` 

After that you gets a dirrectory with generated abc notes. You can convert abc to midi with [abc2midi tool](https://www.systutorials.com/docs/linux/man/1-abc2midi/), or in [web service](https://www.abcjs.net/abcjs-editor.html).

