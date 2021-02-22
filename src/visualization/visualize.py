import pandas as pd
import paths
import ast

credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv')

k=credits['cast'][0]
res = ast.literal_eval(k)
print(res[0])