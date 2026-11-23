import unittest
from skate_sim.validation.observation05 import (numeric_oracle, native_contract_probe,
    mjlab_cpu_mock_probe, test_noise_preserves_actions_and_context)

class Observation05Tests(unittest.TestCase):
    def test_numeric_oracle(self):
        numeric_oracle()

    def test_native_128_state_oracle(self):
        numeric_oracle(backend_entry=True)

    def test_tracking_reference(self):
        from skate_sim.validation.observation05 import tracking_reference_probe
        tracking_reference_probe()

    def test_sensor_noise(self):
        test_noise_preserves_actions_and_context()

    def test_native_contract(self):
        native_contract_probe()

    def test_gpu_adapter_cpu_mock(self):
        mjlab_cpu_mock_probe()

if __name__ == '__main__':
    unittest.main()
