import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
df = pd.read_csv('medical_examination.csv')

# 2. Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalize cholesterol and glucose
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Define draw_cat_plot function
def draw_cat_plot():
    # 5. Create DataFrame for catplot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6. Group and reformat data for catplot
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7. Draw the catplot
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='bar', height=4, aspect=1.2)
    
    # 8. Save the catplot
    fig.savefig('catplot.png')
    return fig

# 9. Define draw_heat_map function
def draw_heat_map():
    # 10. Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # 11. Calculate correlation matrix
    corr = df_heat.corr()
    
    # 12. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 13. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 14. Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', center=0, square=True, linewidths=0.5, ax=ax)
    
    # 15. Save the heatmap
    fig.savefig('heatmap.png')
    return fig

# Run the functions
draw_cat_plot()
draw_heat_map()
