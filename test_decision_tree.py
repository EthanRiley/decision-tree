import pandas as pd
import decision_tree as dt
def main():
    data = pd.read_csv('spotify_train.csv')
    tree = dt.dtree(data, dt.gini, class_col='track_genre')

if __name__ == '__main__':
    main()