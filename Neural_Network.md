# Over view 
### 1. model 
<pre>
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 16 , input_shape = (1,) , activation =  'relu')) # 1 of shape mean our data is 1 dimensional
model.add(tf.keras.layers.Dense(units=32, activation='relu') )
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
</pre>
### 2. compile
<pre>
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
</pre>
### 3. fit
<pre>
model.fit(
      x=scaled_train_samples
    , y=train_labels
    , batch_size=10
    , epochs=30
    , verbose=2 # how many detail process have
)
</pre>

## To set validation, There are two method 
#### (1) put actual data as tuple of samples_array and labels_array
<pre>
model.fit(
  , validation_data=(validation_samples,validation_labels)
)
</pre>

#### (2) put just validation__split
<pre>
model.fit(
  ,validation__split = 0.1
)
</pre>

## Show a infer with images 

It returns a list of possibilities for each label
<pre>
prediction_labels = model.predict(scaled_test_samples)
</pre>
<pre>
 [0.96872485 0.03127513]
 ...
 [0.05153604 0.9484639 ]
</pre>
You can see the most possible label using a argmax
<pre>
for i in np.argmax(prediction_labels, axis=-1):
    print(i)
</pre>

## Saving And Loading The Model In Its Entirety
Check the model architecture and weights
<pre>
model.summary()
print( model.weights )
</pre>

#### (1) Save The architecture and weights same as prior model
<pre>
model.save('models/medical_trial_model.h5')
new_model = tf.keras.models.load_model('models/medical_trial_model.h5')
</pre>

#### (2) Save Only The Architecture Of The Model
<pre>
json_string = model.to_json()
new_model =  tf.keras.models.model_from_json(json_string)
</pre>

#### (3) Save Only The weights Of The Model

<pre>
model.save_weights('models/my_model_weights.h5')
new_model.load_weights('models/my_model_weights.h5')
</pre>
