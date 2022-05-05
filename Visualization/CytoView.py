# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:20:43 2022

@author: remit
"""
import dash
from dash.dependencies import Input, Output, State,MATCH,ALL
from dash import html
import copy
import time
from Stylesheet import Stylesheet
import dash_cytoscape as cyto
import pandas as pd
cyto.load_extra_layouts()
import NodeLayout

class CytoView():
    
    
    
    def __init__(self):
        self.Stylesheet=Stylesheet()
        self.G=[]
        self.G_default=[]
        self.nodeSave=None
        self.edgeSave=None

    def __call__(self,data_nodes,data_edges,CP):
        
        self.data_nodes=data_nodes
        self.data_edges=data_edges
        self.CP=CP
        


        degree=[10 for i in data_nodes]
        
        G=[]
        for elem in data_edges:
            degree[elem['source']]=degree[elem['source']]+1
            degree[elem['destination']]=degree[elem['destination']]+1
            G.append({
                'data':{
                'perso':'e'+str(data_edges.index(elem)),
                'source': 'n'+str(elem['source']), 
                'target': 'n'+str(elem['destination'])},
                'classes': elem['type']
                })
        
        data_n=NodeLayout.normalisation(data_nodes)
        for elem in data_n:
            if 'type'in elem:

                G.append({
                    'data':{'id': 'n'+str(elem['noeuds']),
                            'label':str(degree[elem['noeuds']]),#str((elem['positionX'],elem['positionY'])),
                            'size':degree[elem['noeuds']],
                            },
                    'classes': elem['type'],
                    'position':{'x': 20000*elem['positionX'], 'y': 20000*elem['positionY']},
                    'grabbable': False
                    })
            else :
                G.append({
                    'data':{'id': 'n'+str(elem['noeuds']),
                            'label':str(degree[elem['noeuds']]),#str((elem['positionX'],elem['positionY'])),
                            'size':degree[elem['noeuds']],
                            # 'type':elem['type'],
                            },
                    
                    'position':{'x': 20000*elem['positionX'], 'y': 20000*elem['positionY']},
                    'grabbable': False
                    })
        self.G_default=G
        self.G=G
        
    def modif(self):
        #Where G is ordered by default with edges first and nodes second
        CP=self.CP
          
        G=[]
        for i in range(len(CP.mask_edge)):
            if CP.mask_edge[i]:
                G.append(self.G_default[i])
        for j in range(len(CP.mask_node)):
            if CP.mask_node[j]:
                G.append(self.G_default[i+j+1])
        self.G=copy.deepcopy(G)
        
        
    def create_cyto(self,stylesheet):
        G=copy.deepcopy(self.G)
        self.Stylesheet.stylesheet_default(stylesheet)
        return html.Div(
            html.Div(
                children=[
                    html.Div(
                        cyto.Cytoscape(
                        id='cytoscape',
                        layout={'name': 'preset',
                                'fit':True},
                        elements=G,
                        stylesheet=self.Stylesheet.default_stylesheet,
                        style={'width': '100%', 'height': '83vh','position': 'absolute','top':'0px',
                                'left':'0px','z-index': '999'},
                        minZoom=0.01,
                        maxZoom=10,
                        
                        ),# responsive=True
                        style={
                               'z-index':'100',
                               'top':'0px',
                               'left':'0px'}),
    
                        html.Button('Reset View', id='bt-reset-view',style={'position':'absolute','z-index':'10000','right':'1em','top':'1em'}),
                        html.Button('Reset Stylesheet', id='bt-reset-stylesheet',style={'position':'absolute','z-index':'10000','right':'1em','top':'3em'}),
                        
                    ],
                id='cytoscape-elements',
                style={'position': 'relative','z-index': '100'},
                ),
            style={
                'position': 'fixed',
                'width': '80%',
                'display':'inline-block',
                'verticalAlign': 'top'
            }
                
            
                )
        
    def get_callbacks(self,app):
        

        @app.callback(Output('cytoscape', 'elements'),
        Input('bt-reset-view', 'n_clicks'),
        Input('url', 'pathname'),
        Input({"index": ALL,"type": "edge_legend","label":ALL}, "n_clicks"),
        Input({"index": ALL,"type": "node_legend","label":ALL}, "n_clicks"))
        def reset_layout_view(n_clicks,path,n_edge,n_node):
            triggered = dash.callback_context.triggered[0]['prop_id'].split('.')
            time.sleep(0.05)
            if triggered==None:
                return dash.no_update
            elif  (triggered[0][0:2]=='ur' or triggered[0][0:2]=='bt') and path =="/visualization":

                return self.G
            else:
                self.modif()
                return self.G

        

        @app.callback(Output('cytoscape', 'stylesheet'),
                      Input('cytoscape', 'tapNode'),
                      Input('bt-reset-stylesheet', 'n_clicks'))
        def generate_stylesheet(node,n_clicks):

            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

            if 'bt-reset-stylesheet' in changed_id:
                return self.Stylesheet.default_stylesheet

            if node==None:
                return self.Stylesheet.default_stylesheet
            
            self.Stylesheet.stylesheet_on_click(node)
            
            
            
            return  self.Stylesheet.stylesheet
        

        @app.callback(Output('clicked-element', 'children'),
              Input('cytoscape', 'tapNode'),
              Input('cytoscape', 'tapEdge')
              )
        def clicked_node_element_info(node,edge):

            if node==None and edge==None:
                output=None
            elif node==None :
                output=str(edge["data"])
            elif edge==None :
                output=str(node["data"])
            elif edge["timeStamp"]>node["timeStamp"]:
                output=str(edge["data"])
            else:
                output=str(node["data"])
            
            return html.P(output)
        

            

         
            
                


            
            
            