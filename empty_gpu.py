import os

test_dir = self.args.data_test

if not os.path.exists(test_dir):
    print(f"Error: Test directory does not exist: {test_dir}")
elif len(os.listdir(test_dir)) == 0:
    print(f"Warning: Test directory is empty: {test_dir}")
else:
    print(f"Test directory exists and contains {len(os.listdir(test_dir))} files")
