import tensorflow as tf

class ModelArchitecture(tf.keras.models.Model):

  def __init__(self,num_classes, **kwargs):
    super(ModelArchitecture, self).__init__(**kwargs)

    self.num_classes = num_classes

    self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,padding='same')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.activation1 = tf.keras.layers.Activation('relu')
    #self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same')
    self.avg_pool1 = tf.keras.layers.AveragePooling2D(pool_size=5, padding='same')

    """
    self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=5,strides=2,padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.activation2 = tf.keras.layers.Activation('relu')
    self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same')
    """

    self.flatten = tf.keras.layers.Flatten()

    
    self.fully_connected_1 = tf.keras.layers.Dense(units = 128)
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.activation_relu = tf.keras.layers.Activation('relu')
    
    self.fully_connected_output = tf.keras.layers.Dense(units = self.num_classes)
    self.activation_softmax = tf.keras.layers.Activation('softmax')
        
  def call(self, inputs, training=None):

    first_block = self.conv1(inputs)
    first_block = self.bn1(first_block, training=training)
    first_block = self.activation1(first_block)
    pool1 = self.avg_pool1(first_block)

    """
    second_block = self.conv2(pool1)
    second_block = self.bn2(second_block, training=training)
    second_block = self.activation2(second_block)
    pool2 = self.max_pool2(second_block)
    """

    flat = self.flatten(pool1)
    
    
    third_block = self.fully_connected_1(flat)
    third_block = self.bn3(third_block, training=training)
    third_block = self.activation_relu(third_block)
    

    output = self.fully_connected_output(third_block)
    output = self.activation_softmax(output)

    return output