import pickle

try:
    with open("/home/preston/Public/flat_horizontal/labels.pkl", "rb") as file:
        data = pickle.load(file)
except FileNotFoundError:
    print("Pickle file not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

if isinstance(data, dict):
    if "key1" in data:
        print("Key 'key1' exists.")
    else:
        print("Key 'key1' does not exist.")

    keys = data.keys()
    key_to_print = "joint3d"
    print("All keys:", list(keys))
    if key_to_print in data:
        print(f"The value for key '{key_to_print}' is: {data[key_to_print][0]}")
        diff = data[key_to_print][0] - data[key_to_print][1]
        print(f"The value for key '{key_to_print}' is: {diff}")
    else:
        print(f"Key '{key_to_print}' not found in the data.")

else:
    print("Pickle file does not contain a dictionary.")
