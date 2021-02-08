from analyzer import SequenceAnalyzer
from utils.yaml_wrapper import YamlHandler


if __name__ == '__main__':

    args = YamlHandler('./settings.yaml').read_yaml()
    analyzer=SequenceAnalyzer(args)