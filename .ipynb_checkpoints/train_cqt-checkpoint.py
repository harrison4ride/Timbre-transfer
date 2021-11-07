import argparse
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import CHECKPOINT_DIR, IMG_DIM, OUTPUT_PATH, TEST_AUDIOS_PATH, DEFAULT_SAMPLING_RATE, amin
from data import DataGenerator, forward_cqt, join_magnitude_slices, slice_magnitude,inverse_cqt
from losses import discriminator_loss, generator_loss, l1_loss
from model import Discriminator, Generator

tf.enable_eager_execution()

def generate_audio(prediction, phase, output_name,sr = DEFAULT_SAMPLING_RATE):
    mag_db = join_magnitude_slices(prediction, phase.shape)
    mag_db *= 20*np.log1p(1/(1/(2**16)))
    mag = librosa.db_to_amplitude(mag_db)
    audio = inverse_cqt(mag, phase,sr)
    librosa.output.write_wav(output_name, audio, sr=DEFAULT_SAMPLING_RATE, norm=True)

def train(data, epochs, batch_size=1, gen_lr=1e-4, disc_lr=1e-4, epoch_offset=0):
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(gen_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(disc_lr)

    model_name = data.origin+'_2_'+data.target+'_cqt'
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, model_name)
    
    # Pretrained model
    generator_weights = os.path.join(checkpoint_prefix, 'generator.h5')
    discriminator_weights = os.path.join(checkpoint_prefix, 'discriminator.h5')

    if(not os.path.isdir(checkpoint_prefix)):
        os.makedirs(checkpoint_prefix)
    else:
        if(os.path.isfile(generator_weights)):
            generator.load_weights(filepath=generator_weights, by_name=True)
            print('Generator weights restorred from ' + generator_weights)

        if(os.path.isfile(discriminator_weights)):
            discriminator.load_weights(discriminator_weights, by_name=True)
            print('Discriminator weights restorred from ' + discriminator_weights)

    epoch_size = data.__len__()

    print()
    print("Started training with the following parameters: ")
    print("\tCheckpoints: \t", checkpoint_prefix)
    print("\tEpochs: \t", epochs)
    print("\tgen_lr: \t", gen_lr)
    print("\tdisc_lr: \t", disc_lr)
    print("\tBatchSize: \t", batch_size)
    print("\tnBatches: \t", epoch_size)
    print()

    # Precompute the test input and target for validation
    audio_input,_ = librosa.load( TEST_AUDIOS_PATH+'//'+data.origin+'.wav', DEFAULT_SAMPLING_RATE)
    mag_input, phase = forward_cqt(audio_input, DEFAULT_SAMPLING_RATE)
    mag_input = librosa.amplitude_to_db(mag_input, ref=np.min, amin=amin) # amplitude to db
    mag_input /= 20*np.log1p(1/amin ) # normalize
    test_input = slice_magnitude(mag_input, mag_input.shape[0])
    test_input = (test_input * 2) - 1

    audio_target,_ = librosa.load( TEST_AUDIOS_PATH+'//'+data.target+'.wav', DEFAULT_SAMPLING_RATE)
    mag_target, _ = forward_cqt(audio_target,DEFAULT_SAMPLING_RATE)
    mag_target = librosa.amplitude_to_db(mag_target, ref=np.min, amin=amin) # amplitude to db
    mag_target /= 20*np.log1p(1/amin ) # normalize
    test_target = slice_magnitude(mag_target, mag_target.shape[0])
    test_target = (test_target * 2) - 1

    gen_mae_list,gen_loss_list,disc_loss_list  = [], [], []
    
    test_input,test_target = tf.cast(test_input,tf.float32),tf.cast(test_target,tf.float32) # transfer to float32
    
    for epoch in range(epochs):
        gen_mae_total, gen_loss_total, disc_loss_total= 0, 0, 0

        print('Epoch {}/{}'.format((epoch+1)+epoch_offset, epochs+epoch_offset))
        # Progress Bar
        progbar = tf.keras.utils.Progbar(epoch_size)
        for i in range(epoch_size):
            # Get the data from the DataGenerator
            input_image, target = data.__getitem__(i) 
            # transfer to float32
            input_image,target = tf.cast(input_image,tf.float32),tf.cast(target,tf.float32)
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Generate a fake image
                gen_output = generator(input_image, training=True)
                
                # Train the discriminator
                disc_real_output = discriminator([input_image, target], training=True)
                disc_generated_output = discriminator([input_image, gen_output], training=True)
                
                # Compute the losses
                gen_mae = l1_loss(target, gen_output)
                gen_loss = generator_loss(disc_generated_output, gen_mae)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
                
                # Compute the gradients
                generator_gradients = gen_tape.gradient(gen_loss,generator.trainable_variables)
                discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                
                # Apply the gradients
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

                # Update the progress bar and the totals
                gen_mae = gen_mae.numpy()
                gen_loss = gen_loss.numpy()
                disc_loss = disc_loss.numpy()
                
                
                gen_mae_total += gen_mae
                gen_loss_total += gen_loss
                disc_loss_total += disc_loss

                progbar.add(1, values=[
                                        ("gen_mae", gen_mae), 
                                        ("gen_loss", gen_loss), 
                                        ("disc_loss", disc_loss)
                                    ])
        
        # Write training history 
        gen_mae_list.append(gen_mae_total/epoch_size)
        gen_loss_list.append(gen_loss_total/epoch_size)
        disc_loss_list.append(disc_loss_total/epoch_size)

        history = pd.DataFrame({
                                'gen_mae': gen_mae_list, 
                                'gen_loss': gen_loss_list,
                                'disc_loss': disc_loss_list
                                })
        
        history.to_csv(os.path.join(checkpoint_prefix, 'history.csv'), header='column_names')

        # Generate audios and save spectrograms for the entire audios
        epoch_output = os.path.join(OUTPUT_PATH, model_name, str((epoch+1)+epoch_offset).zfill(3))
        # init directory
        if(not os.path.isdir(epoch_output)):
            os.makedirs(epoch_output)

        prediction = generator(test_input, training=False)
        prediction = (prediction + 1) / 2

        # save image
        plt.imsave(os.path.join(epoch_output, 'spectrogram')+'_'+'input'+'.png', np.flip((((test_input + 1) / 2)[0,:,:,0] + 1) / 2, axis=0)) 
        plt.imsave(os.path.join(epoch_output, 'spectrogram')+'_'+'true'+'.png', np.flip((((test_target + 1) / 2)[0,:,:,0] + 1) / 2, axis=0)) 
        plt.imsave(os.path.join(epoch_output, 'spectrogram')+'_'+'pred'+'.png', np.flip((prediction[0,:,:,0] + 1) / 2, axis=0))

        generate_audio(prediction, phase, os.path.join(epoch_output, 'audio.wav'))
        print('Epoch outputs saved in ' + epoch_output)

        # Save the weights
        generator.save_weights(generator_weights)
        discriminator.save_weights(discriminator_weights)
        print('Weights saved in ' + checkpoint_prefix)

        # Callback at the end of the epoch for the DataGenerator
        data.shuffle()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=False,default='.//data//cqt_features//')
    ap.add_argument('--origin', required=False,default='keyboard_acoustic')
    ap.add_argument('--target', required=False,default='string_acoustic')
    ap.add_argument('--gpu', required=False, default='0')
    ap.add_argument('--epochs', required=False, default=20)
    ap.add_argument('--epoch_offset', required=False, default=0)
    ap.add_argument('--batch_size', required=False, default=5)
    ap.add_argument('--gen_lr', required=False, default=5e-6)
    ap.add_argument('--disc_lr', required=False, default=5e-7)
    ap.add_argument('--validation_split', required=False, default=0.9)
    args = ap.parse_args()
    
    dataset = DataGenerator(origin=args.origin, 
                            target=args.target,
                            base_path=args.dataset_path,
                            batch_size=int(args.batch_size),
                            img_dim=IMG_DIM)
    
    train(
            data=dataset, 
            epochs=int(args.epochs), 
            batch_size=int(args.batch_size), 
            gen_lr=float(args.gen_lr), 
            disc_lr=float(args.disc_lr), 
            epoch_offset=int(args.epoch_offset)
        )
