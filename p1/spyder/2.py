# NI USB-6210 specs

TIMING_ACCURACY = 50e-6    # 50 ppm of sample rate
TIMING_RESOLUTION = 50e-9  # 50 nanoseconds

GAIN_TEMP_COEF = 7.3e-6  # ppm / degree celsius
REF_TEMP_COEF = 5e-6     # ppm / degree celsius
INL_ERROR = 76e-6        # ppm of range

TEMP_CHANGE_INTERNAL = 1
TEMP_CHANGE_EXTERNAL = 10

RESIDUAL_GAIN_ERROR = {
    10:  75e-6,
    5:   85e-6,
    1:   95e-6,
    0.2: 135e-6
}

RESIDUAL_OFFSET_ERROR = {
    10:  20e-6,
    5:   20e-6,
    1:   25e-6,
    0.2: 40e-6
}

OFFSET_TEMP_COEF = {
    10:  34e-6,
    5:   36e-6,
    1:   49e-6,
    0.2: 116e-6
}

COVERAGE_FACTOR = 3  # Number of sigmas to cover

# sigma
RANDOM_NOISE = {
    10:  229e-6,
    5:   118e-6,
    1:   26e-6,
    0.2: 12e-6
}

# This is calculated assuming 1 C (celsius) change from the last internal
# calibration and 10 C change from the last external calibration
GAIN_TEMP_ERROR = GAIN_TEMP_COEF * TEMP_CHANGE_INTERNAL + \
    REF_TEMP_COEF * TEMP_CHANGE_EXTERNAL


def list_devices():
    system = nidaqmx.system.System.local()

    for dev in system.devices:
        print(dev)


def _error_calc(airange):
    gain_error = RESIDUAL_GAIN_ERROR[airange] + GAIN_TEMP_ERROR
    offset_error = RESIDUAL_OFFSET_ERROR[airange] + \
        OFFSET_TEMP_COEF[airange] * TEMP_CHANGE_INTERNAL + INL_ERROR

    n_points = 100
    noise_uncertainty = COVERAGE_FACTOR * \
        RANDOM_NOISE[airange] / np.sqrt(n_points)

    return gain_error, offset_error, noise_uncertainty


def measure(
    channel_name: str,
    duration,            # in seconds
    sampling_freq: int,  # in hertz
    min_val=-10,
    max_val=10
) -> pd.DataFrame:
    n_samples = int(duration * sampling_freq)
    duration = n_samples / sampling_freq

    with nidaqmx.Task() as task:
        # Set analog channel
        ai_channel = task.ai_channels.add_ai_voltage_chan(
            channel_name,
            min_val=min_val,
            max_val=max_val,
            terminal_config=TerminalConfiguration.DIFF
        )

        # Set sampling configuration
        task.timing.cfg_samp_clk_timing(
            sampling_freq,
            samps_per_chan=n_samples,
            sample_mode=AcquisitionType.FINITE
        )

        data = task.read(
            number_of_samples_per_channel=READ_ALL_AVAILABLE,
            timeout=duration+0.1
        )

        airange = ai_channel.ai_max

        # Para checkear
        print(f"Analog input range: {airange}")

    t = np.linspace(0, duration, n_samples)
    y = np.asarray(data)

    # Uncertainty calculation
    t_err = TIMING_ACCURACY / sampling_freq + TIMING_RESOLUTION

    gain_error, offset_error, noise_uncertainty = _error_calc(airange)
    y_err = y * gain_error + airange * offset_error + noise_uncertainty

    df = pd.DataFrame({
        "Tiempo [s]": pd.Series(t),
        "Error X [s]": pd.Series(t_err),
        "Voltaje [V]": pd.Series(y),
        "Error Y [V]": pd.Series(y_err)
    })

    return df
