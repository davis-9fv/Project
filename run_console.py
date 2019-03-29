import run_ds1
import run_ds2
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--dataset',
                        default='',
                        help='ds1 or ds2')
    parser.add_argument('--noprediction',
                        default=False,
                        help='Dont predict')
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
    parser.add_argument('--mlp',
                        default=False,
                        help='Use LSTM algorithm')

    parser.add_argument('--winsizes',
                        type=int,
                        nargs="+",
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='Choose Window Size')

    parser.add_argument('--btcsizes',
                        type=int,
                        nargs="+",
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='Choose BTC Size')

    FLAGS = parser.parse_args()

    use_no_prediction = False
    use_dummy = False
    use_elasticnet = False
    use_lasso = False
    use_knn = False
    use_sgd = False
    use_mlp = False

    print('Dataset:')
    print(FLAGS.dataset)
    print('Window sizes:')
    print(FLAGS.winsizes)



    if FLAGS.dataset != 'ds1' and FLAGS.dataset != 'ds2':
        raise Exception("Argument --dataset is not well set. Either choose ds1 or ds2.")

    if FLAGS.dataset == 'ds1':
        run_ds1.main(FLAGS.winsizes, FLAGS.noprediction, FLAGS.dummy, FLAGS.elasticnet, FLAGS.lasso, FLAGS.knn, FLAGS.sgd, FLAGS.mlp)
    if FLAGS.dataset == 'ds2':
        print('BTC sizes:')
        print(FLAGS.btcsizes)
        run_ds2.main(FLAGS.winsizes,FLAGS.btcsizes,
                     FLAGS.noprediction, FLAGS.dummy, FLAGS.elasticnet, FLAGS.lasso, FLAGS.knn, FLAGS.sgd, FLAGS.mlp)
