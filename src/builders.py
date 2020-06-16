import gin

@gin.configurable
def build_chimera_with_mel(num_features, hidden_size, num_layers, bidirectional,
                           dropout, embedding_size, embedding_activation, num_sources,
                           mask_activation, num_mels, sample_rate, chunk_size=100, 
                           hop_size=50, num_audio_channels=1, rnn_type='lstm', 
                           normalization_class='BatchNorm', normalization_args=None, 
                           mix_key='mix_magnitude'):
    """
    Builds a config for a Chimera network that can be passed to SeparationModel. 
    Chimera networks are so-called because they have two "heads" which can be trained
    via different loss functions. In traditional Chimera, one head is trained using a
    deep clustering loss while the other is trained with a mask inference loss. 
    This Chimera network uses a recurrent neural network (RNN) to process the input 
    representation.
    
    Args:
        num_features (int): Number of features in the input spectrogram (usually means
          window length of STFT // 2 + 1.)
        hidden_size (int): Hidden size of the RNN.
        num_layers (int): Number of layers in the RNN.
        bidirectional (int): Whether the RNN is bidirectional.
        dropout (float): Amount of dropout to be used between layers of RNN.
        embedding_size (int): Embedding dimensionality of the deep clustering network. 
        embedding_activation (list of str): Activation of the embedding ('sigmoid', 'softmax', etc.). 
          See ``nussl.ml.networks.modules.Embedding``. 
        num_sources (int): Number of sources to create masks for. 
        mask_activation (list of str): Activation of the mask ('sigmoid', 'softmax', etc.). 
          See ``nussl.ml.networks.modules.Embedding``. 
        num_audio_channels (int): Number of audio channels in input (e.g. mono or stereo).
          Defaults to 1.
        rnn_type (str, optional): RNN type, either 'lstm' or 'gru'. Defaults to 'lstm'.
        normalization_class (str, optional): Type of normalization to apply, either
          'InstanceNorm' or 'BatchNorm'. Defaults to 'BatchNorm'.
        normalization_args (dict, optional): Args to normalization class, optional.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: A recurrent Chimera network configuration that can be passed to
          SeparationModel.
    """
    normalization_args = {} if normalization_args is None else normalization_args
    recurrent_stack = {
        'class': 'RecurrentStack',
        'args': {
            'num_features': num_mels,
            'hidden_size': hidden_size,
            'num_layers': 1,
            'bidirectional': bidirectional,
            'dropout': 0.0,
            'rnn_type': rnn_type,
            'batch_first': True
        }
    }
    
    # define the building blocks
    modules = {
        mix_key: {},
        'normalization': {
            'class': normalization_class,
            'args': normalization_args,
        },
        'mel_forward': {
            'class': 'MelProjection',
            'args': {
                'sample_rate': sample_rate,
                'num_frequencies': num_features,
                'num_mels': num_mels,
                'direction': 'forward',
            }
        },
        'dual_path': {
            'class': 'DualPath',
            'args': {
                'num_layers': num_layers,
                'chunk_size': chunk_size,
                'hop_size': hop_size,
                'skip_connection': False,
                'in_features': num_mels,
                'bottleneck_size': num_mels,
                # rest are args to DualPathBlock
                'hidden_size': 2 * hidden_size,
                'intra_processor': recurrent_stack,
                'inter_processor': recurrent_stack,
            }
        },
        'embedding': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': num_mels,
                'embedding_size': embedding_size,
                'activation': embedding_activation,
                'num_audio_channels': num_audio_channels,
                'dim_to_embed': [2, 3],
            }
        },
        'mask': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': num_mels,
                'embedding_size': num_sources,
                'activation': mask_activation,
                'num_audio_channels': num_audio_channels,
                'dim_to_embed': [2, 3],
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    }    

    # define the topology
    connections = [
        ['mel_forward', ['mix_magnitude', ]],
        ['normalization', ['mel_forward', ]],
        ['dual_path', ['normalization', ]],
        ['embedding', ['dual_path', ]],
        ['mask', ['dual_path', ]],
        ['estimates', ['mask', 'mix_magnitude']]
    ]

    # define the outputs
    output = ['embedding', 'estimates', 'mask']

    # put it together
    config = {
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config