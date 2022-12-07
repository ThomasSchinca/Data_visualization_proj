# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:23:41 2022

@author: thoma
"""

import pandas as pd 
import numpy as np 
import fiona
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import math
from shapely.ops import cascaded_union
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
import plotly.io as pio
import webbrowser


########################################################################################################################################################################################
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
#                                   Preprocess of the data                                                                                                                             #
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
########################################################################################################################################################################################

ind_all=pd.date_range(start='01/01/2020',end='28/02/2022',freq='M')
df_migra=pd.read_csv('Migra_2.csv',index_col=0)
df_migra.index=ind_all
df_migra_plt=df_migra

############### Selection

h=[]
for col in range(len(df_migra.iloc[0,:])):
    h.append(df_migra.iloc[:,col].value_counts())
h_1=pd.DataFrame(h)    
h_1=h_1.fillna(0)
l_group=h_1[h_1.iloc[:,0:1].sum(axis=1)>13].index.to_list()

######## Group with neighbors 

df_test= gpd.read_file('SS_adm2.shp')
names_shp=df_test[['geometry']]
names_shp=names_shp.transpose()
names_shp.columns=df_test['ADM2_EN']
lat_long=[]
for i in df_migra.columns:
    lat_long.append(names_shp[i].centroid.bounds.values[0][0:2])

ind_out=[]
for i in l_group:
    ind_out.append(df_migra.columns.to_list().index(i))

ind_d=[]
for i in l_group: 
    dist= 10000000000
    cont=0
    for j in lat_long:
        n_d=math.dist(lat_long[df_migra.columns.to_list().index(i)],j)
        if (n_d < dist) and (cont not in ind_out): 
            dist=n_d
            h_c=cont
        cont=cont+1 
    ind_d.append([df_migra.columns.to_list().index(i),h_c])
ind_d=pd.DataFrame(ind_d)      
ind_d.index=ind_d[1] 
ind_d=ind_d.loc[:,0]    
group=[]
for i in ind_d.index.unique():
    if type(ind_d.loc[i].tolist())==int: 
        group.append([i]+[ind_d.loc[i].tolist()])
    else :
        group.append([i]+ind_d.loc[i].tolist())
        
group_tot=[]
for i in group :
    group_tot=group_tot+i
    
####### Group table

df_gr=[]
for i in range(len(df_migra.iloc[0,:])):
    if i not in group_tot:
        df_gr.append(df_migra.iloc[:,i])
df_gr=pd.DataFrame(df_gr)        
df_gr_1=pd.DataFrame(np.array(df_gr.T))
df_gr_1.columns=df_gr.index
df_gr_1.index=df_migra.index

for i in group : 
    df_gr_1=pd.concat([df_gr_1.reset_index(drop=True),df_migra.iloc[:,i].sum(axis=1).reset_index(drop=True)],axis=1)
l_nom=[]
for i in range(1,len(group)+1):
    l_nom.append('Group '+str(i))
df_gr_1.columns=df_gr_1.columns[0:len(df_gr)].tolist()+l_nom
df_gr_1.index=ind_all

####### Group shape 
df_test= gpd.read_file('SS_adm2.shp')
names_shp=df_test[['geometry']]
names_shp=names_shp.transpose()
names_shp.columns=df_test['ADM2_EN']

union=[]
for i in group:
    list_u=[]
    for j in range(len(i)):
        list_u.append(names_shp.loc[:,df_migra.columns[i[j]]][0])
    union.append(cascaded_union(list_u))

shp_gr=[]
for i in range(len(df_migra.iloc[0,:])):
    if i not in group_tot:
        shp_gr.append(names_shp.loc[:,df_migra.columns[i]][0])
shp_gr=shp_gr+union
shp_gr=pd.DataFrame(shp_gr,index=df_gr_1.columns)
lat_long=[]
for i in range(len(shp_gr.index)):
    lat_long.append([shp_gr.iloc[i,0].centroid.bounds[0:2][0],shp_gr.iloc[i,0].centroid.bounds[0:2][1]])

gdf_group = gpd.GeoDataFrame(geometry=shp_gr.iloc[:,0])
gdf_group['centroid']=gdf_group.centroid
df_gr_1 = df_gr_1.T
df_gr_1.columns = [str(ind_all[count])[0:7] for count in range(len(ind_all))]
gdf_group = gdf_group.merge(df_gr_1, left_index=True, right_index=True)


#### Prediction dataframe
pred_train = pd.read_csv('Pred_train.csv',index_col=(0))
pred_test = pd.read_csv('Pred_test.csv',index_col=(0))
pred_tot = pd.concat([pred_train,pred_test])


########################################################################################################################################################################################
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
#                                   Plots                                                                                                                           #
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
########################################################################################################################################################################################


pio.templates.default = "ggplot2"
app = Dash(__name__)
app.layout = html.Div([html.H1(children='Migration Prediction in South Sudan',style={'color': '#00008B',
                               'justify-content': 'center','display': 'flex','font-family':'Trebuchet MS'}),
    dcc.Dropdown(['All']+list(gdf_group.sum(axis=1).sort_values(ascending=False).index)
                 ,'All',id='region-input',style={'width': '600px','justify-content': 'center','font-family':'Trebuchet MS'}),
    html.Div([html.Div(children=[dcc.Graph(id='graph-with-slider')]),
              html.Div(children=[dcc.Graph(id='table1',style={'width': '90vh'}),html.Div([html.Div(children=[dcc.RadioItems(['Fit','Test','All'], 'Test',id='fit_test_all',style = {'width': '100%', 'display': 'flex','align-items': 'center', 'justify-content': 'center','font-family':'Trebuchet MS'})]),
                                           html.Div(children=[dcc.RadioItems(['RF','RF+Cov','RF+Cov+Dyn'], 'RF+Cov+Dyn',id='model_sel',style = {'width': '100%', 'display': 'flex','align-items': 'center', 'justify-content': 'center','font-family':'Trebuchet MS'})]),
                                           ])]),
              ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div(id='text', style={'whiteSpace': 'pre-line','font-family':'Trebuchet MS'}),
    dcc.Graph(id='plot')])

@app.callback(
    Output('graph-with-slider', 'figure'),
    Output('plot', 'figure'),
    Output('table1', 'figure'),
    Output('text','children'),
    Input('region-input','value'),
    Input('fit_test_all','value'),
    Input('model_sel','value')
    )

def update_figure(region,fit_test,model):
    if region == 'All':
        fig_1 = px.choropleth_mapbox(gdf_group,
                       geojson=gdf_group.geometry,
                       locations=gdf_group.index,
                       color=gdf_group.sum(axis=1),height=375,width=680,zoom=0,
                       labels={"index": "Region","color": "Number of emigrants"})
        fig_1.update_layout(mapbox_style="stamen-terrain",margin=dict(l=10, r=4, t=10, b=4),showlegend=False,hovermode="y")
        fig_1.update_layout(mapbox_bounds={"west": 22, "east": 38, "south": 3.3, "north": 12.3})
        fig_1.update_coloraxes(showscale=False)
        fig_2 = px.line(x=df_gr_1.columns, y=df_gr_1.sum(),labels={
                     "x": "Date",
                     "y": "Total of emigrants"})
        fig_2.update_traces(line_color='black')
        fig_2.update_layout(hovermode="x unified")
        if fit_test=='Fit':
            fig_3 = px.imshow(confusion_matrix(pred_train['Observed'],pred_train[model],labels=[0,1,2]),
                labels=dict(x="Predicted Values", y="Observed Values"),
                x=['Decrease', 'Stable', 'Increase'],
                y=['Decrease', 'Stable', 'Increase'],
                text_auto=True)
            v=[]
            for modelo in ['RF','RF+Cov','RF+Cov+Dyn']:
                v.append(accuracy_score(pred_train['Observed'],pred_train[modelo]))
                v.append(precision_score(pred_train['Observed'],pred_train[modelo],average='macro'))
                v.append(recall_score(pred_train['Observed'],pred_train[modelo],average='macro'))
                v.append(f1_score(pred_train['Observed'],pred_train[modelo],average='macro'))
            
        elif fit_test=='Test':
            fig_3 = px.imshow(confusion_matrix(pred_test['Observed'],pred_test[model],labels=[0,1,2]),
                labels=dict(x="Predicted Values", y="Observed Values"),
                x=['Decrease', 'Stable', 'Increase'],
                y=['Decrease', 'Stable', 'Increase'],
                text_auto=True)
            v=[]
            for modelo in ['RF','RF+Cov','RF+Cov+Dyn']:
                v.append(accuracy_score(pred_test['Observed'],pred_test[modelo]))
                v.append(precision_score(pred_test['Observed'],pred_test[modelo],average='macro'))
                v.append(recall_score(pred_test['Observed'],pred_test[modelo],average='macro'))
                v.append(f1_score(pred_test['Observed'],pred_test[modelo],average='macro'))
        else : 
            fig_3 = px.imshow(confusion_matrix(pred_tot['Observed'],pred_tot[model],labels=[0,1,2]),
                labels=dict(x="Predicted Values", y="Observed Values"),
                x=['Decrease', 'Stable', 'Increase'],
                y=['Decrease', 'Stable', 'Increase'],
                text_auto=True)
            v=[]
            for modelo in ['RF','RF+Cov','RF+Cov+Dyn']:
                v.append(accuracy_score(pred_tot['Observed'],pred_tot[modelo]))
                v.append(precision_score(pred_tot['Observed'],pred_tot[modelo],average='macro'))
                v.append(recall_score(pred_tot['Observed'],pred_tot[modelo],average='macro'))
                v.append(f1_score(pred_tot['Observed'],pred_tot[modelo],average='macro'))
        text_acc= 'RF =========> {0} => Accuracy: {1:.3f}, Precision : {2:.3f}, Recall : {3:.3f}, F1-score : {4:.3f}\n RF+Cov =====> {5} => Accuracy: {6:.3f}, Precision : {7:.3f}, Recall : {8:.3f}, F1-score : {9:.3f}\n RF+Cov+Dyn => {10} => Accuracy: {11:.3f}, Precision : {12:.3f}, Recall : {13:.3f}, F1-score : {14:.3f}\n'.format(
            fit_test,v[0],v[1],v[2],v[3],fit_test,
            v[4],v[5],v[6],v[7],fit_test,v[8],v[9],v[10],v[11])  
    else:
        col=[0]*len(df_gr_1)
        col[list(df_gr_1.index).index(region)]=1
        fig_1 = px.choropleth_mapbox(gdf_group,
                       geojson=gdf_group.geometry,
                       locations=gdf_group.index,
                       color=pd.Series(col),height=375,width=680,zoom=0)
        fig_1.update_layout(mapbox_style="stamen-terrain",margin=dict(l=10, r=4, t=10, b=4),showlegend=False,hovermode="y")
        fig_1.update_layout(mapbox_bounds={"west": 22, "east": 38, "south": 3.3, "north": 12.3})
        fig_1.update_traces(hoverinfo='skip',hovertemplate=None)
        fig_1.update_coloraxes(showscale=False)
        fig_1.update_layout(mapbox_style="stamen-terrain",margin=dict(l=10, r=4, t=10, b=4),showlegend=False,hovermode="y",mapbox_bounds={"west": 22, "east": 38, "south": 3.3, "north": 12.3})
        fig_2 = px.line(x=df_gr_1.columns, y=df_gr_1.loc[region,:],labels={
                     "x": "Date",
                     "y": "Total of emigrants"},)
        fig_2.update_traces(line=dict(color="black", width=2))
        fig_2.update_layout(hovermode="x unified")
        fig_2.add_trace(go.Scatter(x=[df_gr_1.columns[-9],df_gr_1.columns[-9]],y=[0,df_gr_1.loc[region,:].max()+df_gr_1.loc[region,:].max()*0.1],line=dict(color='#00008B'),mode='lines',showlegend=False))
        fig_2.add_annotation(x=df_gr_1.columns[-10], y=df_gr_1.loc[region,:].max()+df_gr_1.loc[region,:].max()*0.1,showarrow=False,
            text="Train",font=dict(color='#00008B'))
        fig_2.add_annotation(x=df_gr_1.columns[-8], y=df_gr_1.loc[region,:].max()+df_gr_1.loc[region,:].max()*0.1,showarrow=False,
            text="Test",font=dict(color='#00008B'))
        inde=list(df_gr_1.index).index(region)
        pred_test_r=pred_test.iloc[inde*int(len(pred_test)/49):(inde+1)*int(len(pred_test)/49),:]
        for j in range(len(pred_test_r)):
            if pred_test_r[model].iloc[j] == pred_test_r['Observed'].iloc[j]:
                fig_2.add_annotation(ax=df_gr_1.columns[-9+j],ay=df_gr_1.loc[region,df_gr_1.columns[-9+j]],
                             x=df_gr_1.columns[-8+j],y=df_gr_1.loc[region,df_gr_1.columns[-8+j]],
                             text='',xref="x", yref="y",axref = "x", ayref='y',showarrow=True,arrowsize = 1.5,
                             arrowcolor="chartreuse",arrowhead=1)
            elif abs(pred_test_r[model].iloc[j]-pred_test_r['Observed'].iloc[j])==1:
                fig_2.add_annotation(ax=df_gr_1.columns[-9+j],ay=df_gr_1.loc[region,df_gr_1.columns[-9+j]],
                             x=df_gr_1.columns[-8+j],y=df_gr_1.loc[region,df_gr_1.columns[-9+j]]-df_gr_1.loc[region,:].max()*0.1+df_gr_1.loc[region,:].max()*0.1*pred_test_r[model].iloc[j],
                             text='',xref="x", yref="y",axref = "x", ayref='y',showarrow=True,arrowsize = 1.5,
                             arrowcolor="orange",arrowhead=1)
            else :
                fig_2.add_annotation(ax=df_gr_1.columns[-9+j],ay=df_gr_1.loc[region,df_gr_1.columns[-9+j]],
                             x=df_gr_1.columns[-8+j],y=df_gr_1.loc[region,df_gr_1.columns[-9+j]]-df_gr_1.loc[region,:].max()*0.1+df_gr_1.loc[region,:].max()*0.1*pred_test_r[model].iloc[j],
                             text='',xref="x", yref="y",axref = "x", ayref='y',showarrow=True,arrowsize = 1.5,
                             arrowcolor="red",arrowhead=1)
        fig_2.add_trace(go.Scatter(x=[df_gr_1.columns[-9]],y=[1],line=dict(color="chartreuse"),mode='lines',name="Good Prediction"))
        fig_2.add_trace(go.Scatter(x=[df_gr_1.columns[-9]],y=[1],line=dict(color="orange"),mode='lines',name="Wrong Prediction"))
        fig_2.add_trace(go.Scatter(x=[df_gr_1.columns[-9]],y=[1],line=dict(color="red"),mode='lines',name="Opposite Prediction"))
        
        pred_train_r=pred_train.iloc[inde*int(len(pred_train)/49):(inde+1)*int(len(pred_train)/49),:]
        pred_tot_r=pd.concat([pred_train_r,pred_test_r])
        if fit_test=='Fit':
            fig_3 = px.imshow(confusion_matrix(pred_train_r['Observed'],pred_train_r[model],labels=[0,1,2]),
                labels=dict(x="Predicted Values", y="Observed Values"),
                x=['Decrease', 'Stable', 'Increase'],
                y=['Decrease', 'Stable', 'Increase'],
                text_auto=True)
            v=[]
            for modelo in ['RF','RF+Cov','RF+Cov+Dyn']:
                v.append(accuracy_score(pred_train_r['Observed'],pred_train_r[modelo]))
                v.append(precision_score(pred_train_r['Observed'],pred_train_r[modelo],average='macro'))
                v.append(recall_score(pred_train_r['Observed'],pred_train_r[modelo],average='macro'))
                v.append(f1_score(pred_train_r['Observed'],pred_train_r[modelo],average='macro'))
            
        elif fit_test=='Test':
            fig_3 = px.imshow(confusion_matrix(pred_test_r['Observed'],pred_test_r[model],labels=[0,1,2]),
                labels=dict(x="Predicted Values", y="Observed Values"),
                x=['Decrease', 'Stable', 'Increase'],
                y=['Decrease', 'Stable', 'Increase'],
                text_auto=True)
            v=[]
            for modelo in ['RF','RF+Cov','RF+Cov+Dyn']:
                v.append(accuracy_score(pred_test_r['Observed'],pred_test_r[modelo]))
                v.append(precision_score(pred_test_r['Observed'],pred_test_r[modelo],average='macro'))
                v.append(recall_score(pred_test_r['Observed'],pred_test_r[modelo],average='macro'))
                v.append(f1_score(pred_test_r['Observed'],pred_test_r[modelo],average='macro'))
        else : 
            fig_3 = px.imshow(confusion_matrix(pred_tot_r['Observed'],pred_tot_r[model],labels=[0,1,2]),
                labels=dict(x="Predicted Values", y="Observed Values"),
                x=['Decrease', 'Stable', 'Increase'],
                y=['Decrease', 'Stable', 'Increase'],
                text_auto=True)
            v=[]
            for modelo in ['RF','RF+Cov','RF+Cov+Dyn']:
                v.append(accuracy_score(pred_tot_r['Observed'],pred_tot_r[modelo]))
                v.append(precision_score(pred_tot_r['Observed'],pred_tot_r[modelo],average='macro'))
                v.append(recall_score(pred_tot_r['Observed'],pred_tot_r[modelo],average='macro'))
                v.append(f1_score(pred_tot_r['Observed'],pred_tot_r[modelo],average='macro'))
        text_acc= 'RF =========> {0} => Accuracy: {1:.3f}, Precision : {2:.3f}, Recall : {3:.3f}, F1-score : {4:.3f}\n RF+Cov =====> {5} => Accuracy: {6:.3f}, Precision : {7:.3f}, Recall : {8:.3f}, F1-score : {9:.3f}\n RF+Cov+Dyn => {10} => Accuracy: {11:.3f}, Precision : {12:.3f}, Recall : {13:.3f}, F1-score : {14:.3f}\n'.format(
            fit_test,v[0],v[1],v[2],v[3],fit_test,
            v[4],v[5],v[6],v[7],fit_test,v[8],v[9],v[10],v[11])   
             
    fig_3.update_xaxes(side="top")
    fig_3.update_coloraxes(showscale=False)   
    return fig_1,fig_2,fig_3,text_acc

webbrowser.open('http://127.0.0.1:8050/',new=2)

if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)

