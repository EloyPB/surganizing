time_constant = 20
time_constant_inhibition = time_constant/10
time_constant_error = time_constant/10
neg_error_delay = 3*time_constant

learning_rate = 0.0005

dendrite_threshold = 0.9
activation_function_slope = 3
head_external_threshold = 0.5

max_neg_error_drive = 0.5
max_pos_error_drive = 1.0

noise_max_amplitude = 0.15
noise_period = 6*time_constant
noise_rise_rate = 0.0000002
noise_fall_rate = 0.0002
noise_smoothing_factor = 0.98
