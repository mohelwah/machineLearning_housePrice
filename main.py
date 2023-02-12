import preprocess_Dataset
import Decision_Tree
import Neural_network
def preprocess_():
    path_china = 'C:/Users/root/Desktop/notebook/1- Freelancer work/proj/china.csv'
    path_ksa = 'C:/Users/root/Desktop/notebook/1- Freelancer work/proj/SA_Aqar.csv'
    #path_china = 'C:/Users/Senpai/Documents/SP2/china.csv'
    #path_ksa = 'C:/Users/Senpai/Documents/SP2/SA_Aqar.csv'
    data_ = preprocess_Dataset.Dataset_(path_china, path_ksa)
    data_.read_()
    data_.drop_unwanted_cols()
    data_.encoding_()
    data_.handling_missing_values()
    data_.handling_outlier()
    data_.Data_normalize()
    data_.merage_Dataset()
    data_.split_train_test()
    return data_.X_train, data_.y_train, data_.X_test, data_.y_test


def main():
    # 1- preprocess dataset
    x_train, y_train, x_test, y_test = preprocess_()

    # 2 - model
    #   2.1 -  Initialize the model and select the initial parameters
    ml_ = Decision_Tree.machine_learning(random_state=0)
    nn_ = Neural_network.neural_network()
    #   2.2 -  build model
    ml_.Build_m(x_train, y_train)
    ml_.eval_m(x_test, y_test)
    ml_.visual_()

    nn_.build_model(x_train, y_train)
    nn_.evaluate_model(x_test, y_test)
    nn_.visual_()


main()
