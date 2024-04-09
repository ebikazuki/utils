import json

def read_json(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == '__main__':
    # Example usage
    file_path = './data/config.json'
    config = read_json(file_path)
    print(config['var1'])
    print(config['var3']['var4'])
