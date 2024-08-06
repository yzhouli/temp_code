from model.u2a_msd import User2Audio_MultiFeatureSpammerDetection
from train.u2a_train import User2AudioTrain


def main():
    # Model Predict Dimension 
    predict_size = 2
    # Self-behavior Event Embedding Dimension 
    event_embedding = 768
    # Self-behavior Embedding Dimension 
    self_embedding = 8192
    # Self-behavior and Participant-behavior Liner Layer Dimension 
    h_dim = 256
    # U2A-MSD Model
    u2a_msd = User2Audio_MultiFeatureSpammerDetection(self_size=self_embedding, embedding_size=event_embedding,
                                                      out_size=predict_size, h_dim=h_dim)
    # Model Train or Test Dataset Path
    dataset_path = '/Users/yangzhou/Desktop/database/KDD/AAAI2025/datasets/weibo/dataset'  # please replace dataset path
    # Model Learning Rate
    learning_rate = 0.001
    # Model Training Times
    epochs = 100
    # Simple Number on a Batch for Model Training Process
    batch_size = 2
    # Effective Time Span for User Behavior
    time_interval = -1
    # Threshold for Feature A
    event_threshold = 1000
    user_threshold = 1000
    # Decay Factor for Feature A
    decay_factor = 0.62
    # Sampling Number of Impulse Signal for Audio-like
    audio_sampling_num = 2500
    # Period of Impulse Signal for Audio-like
    hz = 4800
    # Model Training Process
    u2a_train = User2AudioTrain(path_ref=dataset_path, learning_rate=learning_rate, epochs=epochs,
                                batch_size=batch_size, time_interval=time_interval,
                                event_threshold=event_threshold, user_threshold=user_threshold,
                                decay_factor=decay_factor,
                                audio_sampling_num=audio_sampling_num, hz=hz, model=u2a_msd,
                                event_size=event_embedding)
    acc_max, auc_max = u2a_train.train()


if __name__ == '__main__':
    main()
