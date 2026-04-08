import unittest
import argparse
import builtins

# unittest discovery only picks up:
# files starting with test_
# classes starting with Test
# methods starting with test_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--ref_data", required=True, help="Path to reference data")
    args, remaining = parser.parse_known_args()

    builtins.TEST_DATA_DIR = args.test_data
    builtins.REF_DATA_DIR = args.ref_data

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir="spoqc/unittests",
        pattern="test_*.py",
        top_level_dir="."
    )
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)