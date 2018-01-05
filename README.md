# datasetVisualizationTool
A simple csv dataset visualization tool using [plotly](https://plot.ly/) which displays either the data types of the values of the dataset
(from `missing`, `numerical`, `string`, `date` and `boolean`) or a feature-wise repartition of the values.

Here are images of the tool on the [Game Of Thrones dataset](https://www.kaggle.com/mylesoneill/game-of-thrones/data) from Kaggle :
![1](https://github.com/wwatkins42/datasetVisualizationTool/blob/master/resources/images/features-repartition.png?raw=true)
![2](https://github.com/wwatkins42/datasetVisualizationTool/blob/master/resources/images/data-type.png?raw=true)

Run `python visualize.py [csvfile]` to use, run `python visualize.py --help` to see the options.
