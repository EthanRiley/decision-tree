import pandas as pd
import decision_tree as dt
def main():
    data = pd.read_csv('spotify_train.csv')
    tree = dt.dtree(data, dt.gini, class_col='track_genre')
    test = pd.read_csv('spotify_test.csv')
    predictions = dt.predict(tree, test)
    print(predictions)

if __name__ == '__main__':
    main()