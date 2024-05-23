# Text Classification by fine-tuning the GPT model

We are trying to fine-tune the GPT-2 model from open source for some financial summary classification. The task wanted to figure out a way to identify the respective fund type of a certain fund after processing the fund prospectus, which is obviously an NLP task, but we thought of trying to test the result with some large language models too. 

## Data Source
We were given data as follows, with some basic information of the 472 different funds. We were also given the fund prospectus of each fund, which we had combined with the basic information to form the following dataframe.

![image](https://github.com/ThomasK1018/GPT-Classification/assets/69462048/4abbcbd9-5ee5-4036-94aa-88696cf29f6f)

## Task
The original task was to train an NLP model to read the 'summary' column and make classification prediction regarding the 'Ivestment Strategy', but out of curiousity, we tried to pull off the Language Model trick because it makes the task look a lot fancier! And after some online research, we saw the GPT-2 model being available in open source, so we were able to fine-tune that with our current data and classify with it. 
