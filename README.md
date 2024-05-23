# Text Classification by fine-tuning the GPT model

We are trying to fine-tune the GPT-2 model from open source for some financial summary classification. The task wanted to figure out a way to identify the respective fund type of a certain fund after processing the fund prospectus, which is obviously an NLP task, but we thought of trying to test the result with some large language models too. 

## Data Source
We were given data as follows, with some basic information of the 472 different funds. We were also given the fund prospectus of each fund, which we had combined with the basic information to form the following dataframe.

![image](https://github.com/ThomasK1018/GPT-Classification/assets/69462048/4abbcbd9-5ee5-4036-94aa-88696cf29f6f)

## Task
The original task was to train an NLP model to read the 'summary' column and make classification prediction regarding the 'Ivestment Strategy', but out of curiousity, we tried to pull off the Language Model trick because it makes the task look a lot fancier! And after some online research, we saw the GPT-2 model being available in open source, so we were able to fine-tune that with our current data and classify with it. 

## Procedures
Because the task is all about predicting the type with summary, we extracted the two columns out. 
![image](https://github.com/ThomasK1018/GPT-Classification/assets/69462048/0f94a8ec-1c65-4f77-a69b-395b694f2d19)

We then built the GPT-2 neural network:
```
model = TFGPT2Model.from_pretrained("gpt2", use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
model.training = True
model.resize_token_embeddings(len(tokenizer))

for layer in model.layers:
    layer.trainable = False

input = tf.keras.layers.Input(shape=(None,), dtype='int32')
mask = tf.keras.layers.Input(shape=(None,), dtype='int32')
x = model(input, attention_mask=mask)
x = tf.reduce_mean(x.last_hidden_state, axis=1)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(3, activation='softmax')(x)

clf = tf.keras.Model([input, mask], output)

base_learning_rate = 0.0005
optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
loss=tf.keras.losses.SparseCategoricalCrossentropy()

clf.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", verbose=1, patience=3, restore_best_weights=True)

history = clf.fit([X_train_in, X_train_mask], y_train_in, epochs=30, batch_size=32, validation_split=0.2, callbacks=callbacks)
```

## Result
![image](https://github.com/ThomasK1018/GPT-Classification/assets/69462048/80f4286a-2056-48c8-917a-97a663934a29)
The overall F1-score looks fine, but it's mainly due to a large ratio in the equity class. If we look into the confusion table, we will see the following:
![image](https://github.com/ThomasK1018/GPT-Classification/assets/69462048/2c6b3268-1565-46f8-bced-226b1255cd74)
It looks like the GPT-2 model needs further fine-tuning for the classification task.


