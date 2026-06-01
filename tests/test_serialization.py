import os
import shutil
import tempfile
import unittest
import torch
from flowstitch.core.serialization import load_tensors, save_tensors

class TestSerialization(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_load_single_tensor(self):
        # Save a single tensor
        tensor = torch.randn(2, 5)
        path = os.path.join(self.test_dir, "test_tensor.pt")
        save_tensors(tensor, path)
        
        # Verify the saved file name has .safetensors extension
        expected_safetensors_path = os.path.join(self.test_dir, "test_tensor.safetensors")
        self.assertTrue(os.path.exists(expected_safetensors_path))
        
        # Load and verify content
        loaded = load_tensors(path)
        self.assertTrue(torch.equal(tensor, loaded))
        
        # Load from path with .safetensors extension directly
        loaded_direct = load_tensors(expected_safetensors_path)
        self.assertTrue(torch.equal(tensor, loaded_direct))

    def test_save_load_dict_tensors(self):
        # Save a dictionary of tensors
        data = {
            "tensor_a": torch.randn(3, 3),
            "tensor_b": torch.randn(4, 1)
        }
        path = os.path.join(self.test_dir, "test_dict.pt")
        save_tensors(data, path)
        
        # Verify it saved as .safetensors
        expected_safetensors_path = os.path.join(self.test_dir, "test_dict.safetensors")
        self.assertTrue(os.path.exists(expected_safetensors_path))
        
        # Load and verify
        loaded = load_tensors(path)
        self.assertIn("tensor_a", loaded)
        self.assertIn("tensor_b", loaded)
        self.assertTrue(torch.equal(data["tensor_a"], loaded["tensor_a"]))
        self.assertTrue(torch.equal(data["tensor_b"], loaded["tensor_b"]))

    def test_legacy_pt_disabled(self):
        # Create a dummy non-safetensors file
        legacy_path = os.path.join(self.test_dir, "legacy_tensor.pt")
        with open(legacy_path, "wb") as f:
            f.write(b"PK\x03\x04dummy_zip_or_pickle_data")
        
        # Verify it raises FileNotFoundError because pickle loading is disabled
        with self.assertRaises(FileNotFoundError):
            load_tensors(legacy_path)

if __name__ == "__main__":
    unittest.main()
