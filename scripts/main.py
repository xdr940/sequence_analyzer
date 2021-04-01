from analyzer import SequenceAnalyzer
from utils.yaml_wrapper import YamlHandler


if __name__ == '__main__':

    args = YamlHandler('./settings.yaml').read_yaml()
    analyzer=SequenceAnalyzer(args)
    analyzer.photometric_error_map()
    # analyzer.photometric_hist()
    # analyzer.draw()
    # analyzer.photometric_stat()
    # analyzer.corr_test()




    # analyzer.draw()