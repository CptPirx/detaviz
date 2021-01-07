from TABL.tabl_train_tf import train_evaluate, get_average_metrics
from tensorflow import keras
from TABL.tabl_models_tf import TABL_model
import meta


def run_model(model, mode, train_epochs=meta.tabl_epochs, window=10, normalization=None):
    """
    Method that runs the passed model and prints the end metrics.
    :param model: the model to train and evaluate
    :param mode: string name of the run
    :param train_epochs: number of epochs to train the model for
    :param window: the prediction window
    :param normalization: the data normalisation to be used
    :return: --
    """

    results1, trained_model = train_evaluate(model,
                                             train_epochs=train_epochs,
                                             horizon=0,
                                             n_runs=meta.num_runs)

    print("----------")
    print("Mode: ", mode)
    metrics_1 = get_average_metrics(results1)
    print(metrics_1)

    return trained_model


def main():
    """
    Create the network model and run its training.
    """
    # get Bilinear model
    projection_regularizer = None
    projection_constraint = keras.constraints.max_norm(3.0, axis=0)
    attention_regularizer = None
    attention_constraint = keras.constraints.max_norm(5.0, axis=1)

    # model = BL_model(template, dropout, projection_regularizer, projection_constraint)
    model = TABL_model(meta.template, meta.dropout, projection_regularizer, projection_constraint,
                       attention_regularizer, attention_constraint, loss=keras.losses.KLDivergence())
    model.summary()

    run_model(model, 'TABL', window=10, normalization='std')


if __name__ == '__main__':
    main()
