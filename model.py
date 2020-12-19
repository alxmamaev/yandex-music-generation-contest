from transformers import T5Config, T5ForConditionalGeneration

def get_model(vocab_size=30000):
    config_encoder = BertConfig()
    config_decoder = BertConfig()

    config_encoder.vocab_size = vocab_size
    config_decoder.vocab_size = vocab_size

    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True

    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = EncoderDecoderModel(config=config)
    
    return model