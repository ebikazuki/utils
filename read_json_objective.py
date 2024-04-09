import json

class JSONConfig:
    def __init__(self, file_path):
        with open(file_path) as f:
            self.data = json.load(f)

    def __getattr__(self, name):
        if name in self.data:
            value = self.data[name]
            if isinstance(value, dict):
                return JSONConfigDict(value)
            else:
                return value
        else:
            raise AttributeError(f"'JSONConfig' object has no attribute '{name}'")

class JSONConfigDict:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        if name in self.data:
            value = self.data[name]
            if isinstance(value, dict):
                return JSONConfigDict(value)
            else:
                return value
        else:
            raise AttributeError(f"'JSONConfigDict' object has no attribute '{name}'")

    
if __name__ == '__main__':
    # Example usage
    config_path = './data/config.json'
    config = JSONConfig(config_path)
    print(config.var1)
    print(config.var3)
    print(config.var3.var4)
    print(config.var3.var5.var7)
    
    
