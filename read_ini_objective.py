import configparser

class ConfigReader:
    '''iniファイルを読み込んで設定変数オブジェクトを作る'''
    def __init__(self, file_path):
        self.config = configparser.ConfigParser()
        self.config.read(file_path, encoding='utf-8')

    def __getattr__(self, section):
        return SectionWrapper(self.config[section])

class SectionWrapper:
    def __init__(self, section):
        self.section = section

    def __getattr__(self, variable):
        return self.section[variable]

if __name__ == '__main__':
    # Example usage
    config_path = './data/config.ini'
    config = ConfigReader(config_path)
    print(config.DEFAULT.User)
    print(config.DEFAULT.Path)


