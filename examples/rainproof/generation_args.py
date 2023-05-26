GENERATION_CONFIGS = {
    "deterministic": {
        "num_beams": 4,
        "num_return_sequences": 4,
        "temperature": 1.0,
        "max_new_tokens": 150,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling": {
        "num_beams": 2,
        "num_return_sequences": 16,
        "temperature": 1.0,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_16": {
        "num_beams": 2,
        "num_return_sequences": 16,
        "temperature": 1.0,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_temp2": {
        "num_beams": 2,
        "num_return_sequences": 16,
        "temperature": 2.0,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_4": {
        "num_beams": 2,
        "num_return_sequences": 4,
        "temperature": 1.0,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_8": {
        "num_beams": 2,
        "num_return_sequences": 8,
        "temperature": 1.0,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_32": {
        "num_beams": 2,
        "num_return_sequences": 32,
        "temperature": 1.0,
        "max_new_tokens": 150,
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
        "max_new_tokens": 150,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_16_1": {
        "num_beams": 2,
        "num_return_sequences": 16,
        "temperature": 1,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_16_1.5": {
        "num_beams": 2,
        "num_return_sequences": 16,
        "temperature": 1.5,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_16_2": {
        "num_beams": 2,
        "num_return_sequences": 16,
        "temperature": 2.0,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "sampling_16_3": {
        "num_beams": 1,
        "num_return_sequences": 16,
        "temperature": 3.0,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_k": 1000,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "samplingnucleus_16_1.5": {
        "num_beams": 1,
        "num_return_sequences": 16,
        "temperature": 1.5,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_p": 0.9,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "samplingnucleus_32_1.5": {
        "num_beams": 1,
        "num_return_sequences": 32,
        "temperature": 1.5,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_p": 0.9,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "samplingnucleus_64_1.5": {
        "num_beams": 1,
        "num_return_sequences": 32,
        "temperature": 1.5,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_p": 0.9,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
    "samplingnucleus_8_1.5": {
        "num_beams": 1,
        "num_return_sequences": 32,
        "temperature": 1.5,
        "max_new_tokens": 150,
        "do_sample": True,
        "top_p": 0.9,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    },
}
