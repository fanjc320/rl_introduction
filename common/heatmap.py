import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)

# Plot a heatmap for data centered on 0 with a diverging colormap:

normal_data = np.random.randn(10, 12)
ax = sns.heatmap(normal_data, center=0)



# https://seaborn.pydata.org/generated/seaborn.heatmap.html