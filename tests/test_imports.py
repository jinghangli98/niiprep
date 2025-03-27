import unittest

class TestImports(unittest.TestCase):
    """Test that the package can be imported correctly."""
    
    def test_import_package(self):
        """Test importing the main package."""
        try:
            import niiprep
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import niiprep package")
    
    def test_import_functions(self):
        """Test importing the main functions."""
        try:
            from niiprep import resample, register, nii_to_mp4
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import main functions from niiprep")

if __name__ == "__main__":
    unittest.main()
