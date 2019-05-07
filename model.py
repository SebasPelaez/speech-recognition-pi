import tensorflow as tf

class ModelArchitecture(tf.keras.models.Model):

  def __init__(self,num_classes, **kwargs):
    super(ModelArchitecture, self).__init__(**kwargs)

    self.num_classes = num_classes

    self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,padding='same')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.activation1 = tf.keras.layers.Activation('relu')
    self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')
    
    self.conv2 = tf.keras.layers.Conv2D(filters=192,kernel_size=3,strides=1,padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.activation2 = tf.keras.layers.Activation('relu')
    self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')

    self.conv3 = tf.keras.layers.Conv2D(filters=192,kernel_size=3,strides=1,padding='same')
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.activation3 = tf.keras.layers.Activation('relu')
    
    self.global_average_pool = tf.keras.layers.GlobalAveragePooling2D()
    self.fully_connected = tf.keras.layers.Dense(units = self.num_classes)
    self.activation_softmax = tf.keras.layers.Activation('softmax')
        
  def call(self, inputs, training=None):

    first_block = self.conv1(inputs)
    first_block = self.bn1(first_block, training=training)
    first_block = self.activation1(first_block)
    first_block = self.pool1(first_block)
    
    second_block = self.conv2(first_block)
    second_block = self.bn2(second_block, training=training)
    second_block = self.activation2(second_block)
    second_block = self.pool2(second_block)

    third_block = self.conv3(second_block)
    third_block = self.bn3(third_block, training=training)
    third_block = self.activation3(third_block)

    gap = self.global_average_pool(third_block)

    output = self.fully_connected(gap)
    output = self.activation_softmax(output)

    return output