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
<pre>
prediction_labels  = model.predict(scaled_test_samples)
</pre>
