import pandas as pd

class Transformation:
    def __init__(self, data):
        self.data = data

    def drop_na(self):
        self.data = self.data.dropna()

    def drop_columns(self, columns):
        self.data = self.data.drop(columns = columns)

    def drop_rows(self, column, value):
        self.data = self.data[self.data[column] != value]

    def cat_to_num(self, column, x1, x2):
        self.data[column] = self.data[column].map({x1: 0, x2: 1})

    def one_hot_encoding(self, column, col_to_drop):
        self.data = pd.get_dummies(self.data, columns = [column], prefix = column)
        self.data = self.data.drop(columns = [f"{column}_{col_to_drop}"])

        for col in self.data.columns:
            if self.data[col].dtype == 'bool':
                self.data[col] = self.data[col].astype(float)

    def minutes_to_hours(self, columns):
        for col in columns:
            self.data[col] = self.data[col] / 60

    def rename_columns(self, columns_dict):
        self.data = self.data.rename(columns = columns_dict)

    def one_hot_encoding_map(self, column, mapping, col_to_drop):
        self.data[column] = self.data[column].map(mapping)
        self.one_hot_encoding(column, col_to_drop)

    def normalize(self, scaler):
        self.data = scaler.transform(self.data)