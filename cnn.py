'''
    # %pip install tensorflow
    # %pip install scikit-learn
    # %pip install scipy
    # %pip install matplotlib
'''
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t
from tensorflow.keras.datasets import mnist


def CNNFashionMnist():

    #https://github.com/guilhermedom/cnn-fashion-mnist/blob/main/notebooks/1.0-gdfs-cnn-fashion-mnist.ipynb
    #https://github.com/zubairsamo/Fashion-Mnist-Using-CNN


    # CNN on MNIST Fashion Original Resolution
    # **Referenced https://en.wikipedia.org/wiki/LeNet for LeNet Architecture**

    # Commented out IPython magic to ensure Python compatibility.


    # Load the MNIST Fashion dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape images to 3D tensors (height, width, channels)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    '''
    Here I defined the CNN model with reference to LeNet architecture
    - The model has two convolutional layers followed by max pooling layers
    - The output of the second pooling layer is flattened and passed through two dense layers
    - The final dense layer has 10 units with softmax activation

    Reason I chose to implement LeNet architecture is because it is a simple and effective architecture for image classification
    It has been used for MNIST dataset and has shown good performance
    '''

    # Define the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    epochs = 10
    history = model.fit(train_images, train_labels, epochs, validation_data=(test_images, test_labels), verbose=2)
    train_acc = history.history['accuracy']
    print('Train accuracy:', train_acc[-1])

    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f'Test accuracy: {test_acc:.3f}')

    def get_margin_of_error(test_acc, accuracy_results, n_iterations):
        # Compute confidence intervals
        p_value = (np.sum(np.array(accuracy_results) >= test_acc) + 1) / (n_iterations + 1)

        print("p-value:", p_value)

        alpha = 0.05
        z_score = stats.norm.ppf(1 - alpha / 2)

        # Compute the standard error of the permutation test accuracies
        std_error = np.std(accuracy_results) / np.sqrt(n_iterations)

        # Compute the margin of error
        margin_of_error = z_score * std_error

        # Calculate the lower and upper bounds of the confidence interval
        lower_bound = test_acc - margin_of_error
        upper_bound = test_acc + margin_of_error

        print("p-value:", p_value)
        print("95% Confidence Interval:", (lower_bound, upper_bound))
        return margin_of_error, (lower_bound, upper_bound)

    # Mini-batch Bootstrap resampling for confidence interval estimation
    n_iterations = 10
    batch_size = 128  # Adjust batch size as needed
    accuracy_results = []

    print()
    print("Performing Permutation test...")
    
    for _ in range(n_iterations):
        #print(f'Iteration {_ + 1}/{n_iterations}')
        # Randomly select indices for the mini-batch
        indices = np.random.choice(train_images.shape[0], size=100, replace=True)
        indices_tensor = tf.convert_to_tensor(indices)  # Convert numpy array to tensor
        resampled_images = tf.gather(train_images, indices_tensor)
        resampled_labels = tf.gather(train_labels, indices_tensor)

        permuted_train_labels = np.random.permutation(resampled_labels)

        model.fit(resampled_images, permuted_train_labels, epochs=10, verbose=0)

        # Evaluate model on test data
        _, acc = model.evaluate(test_images, test_labels, verbose=0)
        accuracy_results.append(acc)  # Append accuracy value to the list
        #print(f'Accuracy: {acc:.3f}')

    train_margin_of_error, train_confidence_interval = get_margin_of_error(train_acc, accuracy_results, n_iterations)

    test_margin_of_error, test_confidence_interval = get_margin_of_error(test_acc, accuracy_results, n_iterations)

    print('train_margin_of_error:', train_margin_of_error)
    print('test_margin_of_error:', test_margin_of_error)

    # Plot training and test accuracy
    '''
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.3, 1])
    plt.legend(loc='lower right')


    # Plot confidence intervals
    for epoch, (train_acc, test_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
        plt.errorbar(epoch, train_acc, yerr=train_margin_of_error, fmt='none', ecolor='gray', capsize=5)
        plt.errorbar(epoch, test_acc, yerr=test_margin_of_error, fmt='none', ecolor='gray', capsize=5)

    plt.title('Training and Test Accuracy with Confidence Intervals')
    plt.show()
    '''
    # plot both train and test confidence intervals
    plt.errorbar(x=0, y=train_acc, yerr=train_margin_of_error, fmt='o', label='Train Accuracy')
    plt.errorbar(x=1, y=test_acc, yerr=test_margin_of_error, fmt='o', label='Test Accuracy')
    plt.xticks([0, 1], ['Train', 'Test'])
    plt.ylabel('Accuracy')
    plt.title('95% Confidence Interval for Train and Test Accuracy after ' + str(epochs) + ' Epochs')
    plt.legend()

    plt.savefig('CNN_Fashion_MNIST.png')
    plt.clf()
    return test_confidence_interval

def CNNDigitMnist():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Print dimensions of the original images
    #print("Original Train Images Shape:", train_images.shape)
    #print("Original Test Images Shape:", test_images.shape)

    # Display some original images
    '''
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(train_images[i], cmap='gray')
        plt.title(f"Label: {train_labels[i]}")
        plt.axis('off')
    plt.show()
    ''' 
    """## I'm using MAX pooling instead of average pooling because
    - **I found that average pooling made the images too blurry**
    - **MAX pooling retains the edges which is important for digit recognition**
    """

    # Reshape images to 4D tensors (height, width, channels)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # Define the pooling layer
    pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

    # Apply max pooling to the images
    train_images_pooled = pooling_layer(train_images)
    test_images_pooled = pooling_layer(test_images)

    # Normalize pixel values to be between 0 and 1
    train_images_pooled = train_images_pooled / 255.0
    test_images_pooled = test_images_pooled / 255.0

    #print("Pooled Train Images Shape:", train_images_pooled.shape)

    train_images = train_images_pooled
    test_images = test_images_pooled

    # Display some original images
    '''
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(train_images_pooled[i], cmap='gray')
        plt.title(f"Label: {train_labels[i]}")
        plt.axis('off')
    plt.show()
    '''
    """## Here I defined the CNN model with reference to LeNet architecture
    - **The model has two convolutional layers followed by max pooling layers**
    - **The output of the second pooling layer is flattened and passed through two dense layers**
    - **The final dense layer has 10 units with softmax activation**

    **Reason I chose to implement LeNet architecture is because it is a simple and effective architecture for image classification**
    **It has been used for MNIST dataset and has shown good performance**
    """

    # Define the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(7, 7, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    """**I chose to use the Adam optimizer because it is computationally efficient and popular for CNNs**

    """

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    """**I chose to use 10 epochs because I found that the model converges to a good accuracy within 10 epochs**

    **With more time, I would have trained for more epochs**
    """

    epochs = 10
    # Train the model
    history = model.fit(train_images, train_labels, epochs, validation_data=(test_images, test_labels), verbose=2)
    train_acc = history.history['accuracy']
    print("Train accuracy:", train_acc)

    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f'Test accuracy: {test_acc*100:.2f}%')

    def get_margin_of_error(test_acc, accuracy_results, n_iterations):
        # Compute confidence intervals
        p_value = (np.sum(np.array(accuracy_results) >= test_acc) + 1) / (n_iterations + 1)

        print("p-value:", p_value)

        alpha = 0.05
        z_score = stats.norm.ppf(1 - alpha / 2)

        # Compute the standard error of the permutation test accuracies
        std_error = np.std(accuracy_results) / np.sqrt(n_iterations)

        # Compute the margin of error
        margin_of_error = z_score * std_error

        # Calculate the lower and upper bounds of the confidence interval
        lower_bound = test_acc - margin_of_error
        upper_bound = test_acc + margin_of_error

        print("p-value:", p_value)
        print("95% Confidence Interval:", (lower_bound, upper_bound))
        return margin_of_error, (lower_bound, upper_bound)

    # Permutation test
    n_iterations = 10
    permutation_test_accuracies = []
    print()
    print("Performing Permutation test...")
    for i in range(n_iterations):
        #print(f'Iteration: {i+1}')

        indices = np.random.choice(train_images.shape[0], size=100, replace=True)
        indices_tensor = tf.convert_to_tensor(indices)  # Convert numpy array to tensor
        resampled_images = tf.gather(train_images, indices_tensor)
        resampled_labels = tf.gather(train_labels, indices_tensor)

        permuted_train_labels = np.random.permutation(resampled_labels)

        model.fit(resampled_images, permuted_train_labels, epochs=10, verbose=0)

        # Evaluate model on test data
        _, acc = model.evaluate(test_images, test_labels, verbose=0)
        permutation_test_accuracies.append(acc)  # Append accuracy value to the list
        #print(f'Accuracy: {acc*100:.2f}%')



    train_margin_of_error, train_confidence_interval = get_margin_of_error(train_acc, permutation_test_accuracies, n_iterations)

    test_margin_of_error, test_confidence_interval = get_margin_of_error(test_acc, permutation_test_accuracies, n_iterations)

    # Plot training and test accuracy
    '''
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.3, 1])
    plt.legend(loc='lower right')

    # Plot confidence intervals
    for epoch, (train_acc, test_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
        plt.errorbar(epoch, train_acc, yerr=train_margin_of_error, fmt='none', ecolor='gray', capsize=5)
        plt.errorbar(epoch, test_acc, yerr=test_margin_of_error, fmt='none', ecolor='gray', capsize=5)

    plt.title('Training and Test Accuracy with Confidence Intervals')
    plt.show()
    '''

    plt.title('95% Confidence Interval for Train and Test Accuracy after ' + str(epochs) + ' Epochs')
    # plot both train and test confidence intervals
    plt.errorbar(x=0, y=train_acc, yerr=train_margin_of_error, fmt='o', label='Train Accuracy')
    plt.errorbar(x=1, y=test_acc, yerr=test_margin_of_error, fmt='o', label='Test Accuracy')
    plt.xticks([0, 1], ['Train', 'Test'])
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('CNN_Digit_MNIST.png')
    plt.clf()
    
    return test_confidence_interval

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print('---------------CNN Fashion Mnist----------------')
    print()
    fashion_test_accuracy_confidence_interval = CNNFashionMnist()
    print()

    print('---------------CNN Digit Mnist----------------')
    print()
    digit_test_accuracy_confidence_interval = CNNDigitMnist()
    print()
    # Calculate means
    fashion_mean = sum(fashion_test_accuracy_confidence_interval) / 2
    digit_mean = sum(digit_test_accuracy_confidence_interval) / 2

    # Plot confidence intervals
    plt.errorbar(x=0, y=fashion_mean, yerr=fashion_test_accuracy_confidence_interval[1]-fashion_mean, fmt='o', label='Fashion Test Accuracy')
    plt.errorbar(x=1, y=digit_mean, yerr=digit_test_accuracy_confidence_interval[1]-digit_mean, fmt='o', label='Digit Test Accuracy')
    plt.xticks([0, 1], ['Fashion', 'Digit'])
    plt.ylabel('Accuracy')
    plt.title('95% Confidence Interval for Fashion and Digit Test Accuracy')
    plt.legend()
    plt.savefig('CNN_Fashion_vs_Digit_MNIST.png')
    plt.show()
    plt.clf()