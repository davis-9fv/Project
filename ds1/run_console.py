import ds1.run as run
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='App description')

    """" For merging wifi with atm"""
    parser.add_argument('--dummy',
                        default=False,
                        help='Dummy information')
    parser.add_argument('--elasticnet',
                        default=False,
                        help='Use Elasticnet algorithm')
    parser.add_argument('--lasso',
                        default=False,
                        help='Use Lasso algorithm')
    parser.add_argument('--knn',
                        default=False,
                        help='Use KNN algorithm')
    parser.add_argument('--sgd',
                        default=False,
                        help='Use SGD algorithm')
    parser.add_argument('--lstm',
                        default=False,
                        help='Use LSTM algorithm')

    FLAGS = parser.parse_args()

    use_dummy = False
    use_elasticnet = False
    use_lasso = False
    use_knn = False
    use_sgd = True
    use_lstm = False
    run.main(FLAGS.dummy, FLAGS.elasticnet, FLAGS.lasso, FLAGS.knn, FLAGS.sgd, FLAGS.lstm)
