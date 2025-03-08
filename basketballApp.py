import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA Player Stats Explorer')

st.markdown(""""
This app performs simple webscraping of NBA players stats data!
* **Data Source:** [Basketball-reference.com](https://www.basketball-reference.com)    
""")

st.sidebar.header('User Input Features')
selectedYear = st.sidebar.selectbox('Year',list(reversed(range(1950,2024))))

@st.cache_data
def loadData(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw['Awards'] = raw['Awards'].fillna('')
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    playerstats = playerstats[playerstats.Player != "League Average"]


    return playerstats



playerStats = loadData(selectedYear)

sortedUniqueTeam = sorted(playerStats.Team.unique())
selectedTeam = st.sidebar.multiselect('Team',sortedUniqueTeam,sortedUniqueTeam)

uniquePos = playerStats.Pos.unique()
selectedPos = st.sidebar.multiselect('Position',uniquePos,uniquePos)

dfSelectedTeam = playerStats[(playerStats.Team.isin(selectedTeam)) & (playerStats.Pos.isin(selectedPos))]

st.header('Display Player Stats of Selected Team(s)')
st.dataframe(dfSelectedTeam)


if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    dfSelectedTeam.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    corr = df.select_dtypes(include=['number']).corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f,ax = plt.subplots(figsize=(7,5))
        ax = sns.heatmap(corr,mask=mask,vmax=1,square=True)
    st.pyplot(f)
    
