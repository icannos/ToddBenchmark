GENERATION_CONFIGS = {
    "sampling": {
        "num_beams": 4,
        "num_return_sequences": 16,
        "temperature": 1.0,
        "max_length": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "deterministic": {
        "num_beams": 16,
        "num_return_sequences": 16,
        "num_beam_groups": 4,
        "temperature": 1.0,
        "max_length": 150,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
}
