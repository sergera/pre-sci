import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import tkinter

matplotlib.use("TkAgg")

class Plot():
    def discrete_feature_continuous_target(self, data, var, target):
        title = "Discrete ("+var+"): Target distribution per feature value"
        y_label = "Target ("+target+") value"
        x_label = var
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        self.continuous_x_discrete(data[var], data[target], subtitles)

    def discrete_feature_non_continuous_target(self, data, var, target):
        title = "Discrete ("+var+"): Ratio of label value per target"
        y_label = "Ratio of label value"
        x_label = var+" value"
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        self.discrete_x_discrete(data[var], data[target], subtitles)
        title = "Discrete ("+var+"): Ratio of target value per label"
        y_label = "Ratio of target value"
        x_label = "target("+target+") value"
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        self.discrete_x_discrete(data[target], data[var],subtitles)

    def continuous_feature_continuous_target(self, data, var, target):
        line_title = "Continuous ("+var+"): Mean target per value"
        line_y_label = "Median Target ("+target+")"
        line_x_label = var
        line_subtitles = {"title": line_title, "y_label": line_y_label, "x_label": line_x_label}
        scatter_title = "Continuous ("+var+"):"
        scatter_y_label = "Target ("+target+")"
        scatter_x_label = var
        scatter_subtitles = {"title": scatter_title, "y_label": scatter_y_label, "x_label": scatter_x_label}
        subtitles = {"line": line_subtitles, "scatter": scatter_subtitles}
        self.continuous_x_continuous(data[var], data[target], subtitles)

    def continuous_feature_non_continuous_target(self, data, var, target):
        title = "Continuous ("+var+"): Feature distribution per target value"
        y_label = var
        x_label = "Target ("+target+") value"
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        self.continuous_x_discrete(data[target], data[var], subtitles)

    def categorical_feature_continuous_target(self, data, var, target):
        title = "Categorical ("+var+"): Target distribution per feature value"
        y_label = "Target ("+target+") value"
        x_label = var
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        self.continuous_x_discrete(data[var], data[target], subtitles)

    def categorical_feature_non_continuous_target(self, data, var, target):
        title = "Categorical ("+var+"): Ratio of feature value per target"
        y_label = "Ratio of feature value"
        x_label = var+" value"
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        self.discrete_x_discrete(data[var], data[target], subtitles)
        title = "Categorical ("+var+"): Ratio of target value per label"
        y_label = "Ratio of target value"
        x_label = "target("+target+") value"
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        self.discrete_x_discrete(data[target], data[var],subtitles)

    def na_continuous_target(self, data, var, target):
        title = "Features With Missing("+var+"): Mean target value per feature presence"
        y_label = "Median target ("+target+")"
        x_label = var+": 0=present 1=missing"
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        var_missing = data[var].isnull().replace({0: "filled", 1:"missing"})
        self.continuous_x_discrete(var_missing, data[target], subtitles)

    def na_non_continuous_target(self, data, var, target):
        title = "Features With Missing("+var+"): Ratio of feature presence per target value"
        y_label = "Ratio"
        x_label = "Presence per target ("+target+") value"
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}
        var_missing = data[var].isnull().replace({0: "filled", 1:"missing"})
        self.discrete_x_discrete(var_missing, data[target], subtitles)

    def continuous_distribution(self, data, var):
        title = "Continuous ("+var+"): Feature Distribution"
        y_label = "Number of Observations"
        x_label = var+" value"
        subtitles = {"title": title, "y_label": y_label, "x_label": x_label}     
        self.histogram(data[var], subtitles)           

    def discrete_x_discrete(self, row, column, subtitles):
        join = pd.concat([column, row], axis=1)
        unique_row_values = row.dropna().unique()
        unique_row_values.sort()
        unique_column_values = column.dropna().unique()
        unique_column_values.sort()
        relation = {}

        for column_value in unique_column_values:
            column_name = column.name+"="+str(column_value)
            relation[column_name] = {}
            for row_value in unique_row_values:
                row_value_size_for_column_value = len(join[(join[column.name] == column_value) & (join[row.name] == row_value)].index)
                row_value_size = len(join[join[row.name] == row_value].index)
                ratio = row_value_size_for_column_value / row_value_size
                relation[column_name][row_value] = ratio

        df_to_plot = pd.DataFrame(relation)
        self.bar(df_to_plot, subtitles)

    def continuous_x_discrete(self, x, y, subtitles):
        df = pd.concat([x, y], axis=1)
        discrete_values = x.dropna().unique()
        discrete_values.sort()
        values_per_discrete_value = {}

        for discrete_value in discrete_values:
            rows_discrete_value = df[df[x.name] == discrete_value]
            continuous_only = rows_discrete_value[y.name]
            mean = continuous_only.mean()
            column_name = str(discrete_value)+"\nmean:\n"+str(round(mean,2))
            values_per_discrete_value[column_name] = continuous_only

        df_to_plot = pd.DataFrame(values_per_discrete_value)
        self.box(df_to_plot, subtitles)

    def continuous_x_continuous(self, x, y, subtitles):
        join = pd.concat([x, y], axis=1)
        df_to_line = join.groupby([x.name])[y.name].mean()
        self.line(df_to_line, subtitles["line"])
        self.scatter(x, y, subtitles['scatter'])

    def bar(self, df, subtitles):
        df.plot.bar()
        self.plot(subtitles)

    def line(self, df, subtitles):
        df.plot()
        self.plot(subtitles)
    
    def scatter(self, x, y, subtitles):
        plt.scatter(x,y)
        self.plot(subtitles)

    def histogram(self, df, subtitles, bins=40):
        df.plot.hist(bins=bins)
        self.plot(subtitles)

    def box(self, df, subtitles):
        df.boxplot()
        self.plot(subtitles)

    def plot(self, subtitles):
        if "title" in subtitles:
            plt.title(subtitles['title'])
        if "y_label" in subtitles:
            plt.ylabel(subtitles['y_label'])
        if "x_label" in subtitles:
            plt.xlabel(subtitles['x_label'])
        plt.show()
