from transformers import T5Config, T5ForConditionalGeneration

def get_model(vocab_size=30000):
    config = T5Config()
    config.vocab_size = vocab_size

    model = T5ForConditionalGeneration(config)
    
    return model