import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import kagglehub

# Helper to load external Kaggle data
def load_kaggle_data(part):
    try:
        path = kagglehub.dataset_download("pkdarabi/bone-fracture-detection-computer-vision-project")
        print(f"Kaggle data found at: {path}")
        
        kaggle_dataset = []
        # Class mapping based on data.yaml
        # 0: elbow positive, 1: fingers positive, 2: forearm fracture, 
        # 3: humerus fracture, 4: humerus (normal), 5: shoulder fracture, 6: wrist positive
        class_to_part = {
            0: "Elbow", 1: "Hand", 2: "Wrist", 
            3: "Shoulder", 4: "Shoulder", 5: "Shoulder", 6: "Wrist"
        }
        
        # We only care about the requested 'part'
        # Note: the project parts are ["Elbow", "Hand", "Shoulder"]
        # We map Wrist/Hand to Hand, and Shoulder to Shoulder.
        
        for split in ['train', 'test']:
            img_dir = os.path.join(path, "BoneFractureYolo8", split, "images")
            label_dir = os.path.join(path, "BoneFractureYolo8", split, "labels")
            
            if not os.path.exists(img_dir): continue
            
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                label_name = os.path.splitext(img_name)[0] + ".txt"
                label_path = os.path.join(label_dir, label_name)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Use the first detected object's class
                            class_idx = int(lines[0].split()[0])
                            detected_part = class_to_part.get(class_idx, "Unknown")
                            
                            # Filter for the requested part
                            # Normalize Wrist/Hand to Hand for the project's models if needed
                            target_part = part
                            if part == "Hand" and detected_part in ["Hand", "Wrist"]:
                                pass # Match
                            elif part == detected_part:
                                pass # Match
                            else:
                                continue # Skip
                                
                            label = "fractured"
                            if class_idx == 4: # humerus (normal)
                                label = "normal"
                                
                            kaggle_dataset.append({
                                'body_part': detected_part,
                                'patient_id': 'kaggle_' + img_name.split('_')[0],
                                'label': label,
                                'image_path': img_path
                            })
        
        print(f"Loaded {len(kaggle_dataset)} external images for {part}")
        return kaggle_dataset
    except Exception as e:
        print(f"Kaggle parsing failed: {e}")
        return []

def load_path(path, part):
    """
    load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = path + '/' + str(folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                if body == part:
                    body_part = body
                    path_p = folder + '/' + str(body)
                    for id_p in os.listdir(path_p):
                        patient_id = id_p
                        path_id = path_p + '/' + str(id_p)
                        for lab in os.listdir(path_id):
                            if lab.split('_')[-1] == 'positive':
                                label = 'fractured'
                            elif lab.split('_')[-1] == 'negative':
                                label = 'normal'
                            path_l = path_id + '/' + str(lab)
                            for img in os.listdir(path_l):
                                img_path = path_l + '/' + str(img)
                                dataset.append(
                                    {
                                        'body_part': body_part,
                                        'patient_id': patient_id,
                                        'label': label,
                                        'image_path': img_path
                                    }
                                )
    return dataset


# this function get part and know what kind of part to train, save model and save plots
def trainPart(part):
    # categories = ['fractured', 'normal']
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = THIS_FOLDER + '/Dataset/'
    
    # Load primary dataset
    data = load_path(image_dir, part)
    
    # Load external Kaggle dataset for improved accuracy
    external_data = load_kaggle_data(part)
    data.extend(external_data)
    
    labels = []
    filepaths = []

    # add labels for dataframe for each category 0-fractured, 1- normal
    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    # split all dataset 10% test, 90% train (after that the 90% train will split to 20% validation and 80% train
    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

    # each generator to process and convert the filepaths into image arrays,
    # and the labels into one-hot encoded labels.
    # The resulting generators can then be used to train and evaluate a deep learning model.

    # now we have 10% test, 72% training and 18% validation
    # MEDICAL PERFORMANCE PRIORITY: Augmentation
    # Enhance robustness to subtle variations (zoom, shift, rotation, flips)
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Added vertical flip for X-ray orientation robustness
        fill_mode='nearest'
    )

    # use ResNet50 architecture
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    # we use rgb 3 channels and 224x224 pixels images, use feature extracting , and average pooling
    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')

    # fine-tune the top layers of ResNet50 for medical specificity
    pretrained_model.trainable = True
    # Freeze bottom layers (first 100 out of ~175) to keep core features
    for layer in pretrained_model.layers[:100]:
        layer.trainable = False

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(256, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dropout(0.3)(x) # Added dropout for better generalization
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Added dropout for better generalization
    x = tf.keras.layers.Dense(50, activation='relu')(x)

    # outputs Dense '2' because of 2 classes, fratured and normal
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print("-------Training " + part + "-------")

    # Calculate class weights to handle imbalance and prioritize fractures
    unique_classes = np.unique(train_df['Label'])
    weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=train_df['Label'])
    class_weights_dict = dict(enumerate(weights))
    print(f"Class Weights: {class_weights_dict}")

    # Adam optimizer with lower learning rate for fine-tuning stability
    model.compile(optimizer=Adam(learning_rate=0.00005), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

    # callbacks: EarlyStopping and Learning Rate Scheduler for precision
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    ]
    history = model.fit(train_images, validation_data=val_images, epochs=30, callbacks=callbacks, class_weight=class_weights_dict)

    # save model to this path
    model.save(THIS_FOLDER + "/weights/ResNet50_" + part + "_frac.h5")
    results = model.evaluate(test_images, verbose=0)
    print(part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

    # create plots for accuracy and save it
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    figAcc = plt.gcf()
    my_file = os.path.join(THIS_FOLDER, "./plots/FractureDetection/" + part + "/_Accuracy.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

    # create plots for loss and save it
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    figAcc = plt.gcf()
    my_file = os.path.join(THIS_FOLDER, "./plots/FractureDetection/" + part + "/_Loss.jpeg")
    figAcc.savefig(my_file)
    plt.clf()


# run the function and create model for each parts in the array
categories_parts = ["Elbow", "Hand", "Shoulder"]
for category in categories_parts:
    trainPart(category)
