#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
# 用PageRank挖掘希拉里邮件中的重要人物关系
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# 数据加载
emails = pd.read_csv('Emails.csv')
# 读取别名文件
file = pd.read_csv('Aliases.csv')
aliases = {}
for index, row in file.iterrows():
    aliases[row['Alias']] = row['PersonId']
# 读取人名文件
file = pd.read_csv('Persons.csv')
persons = {}
for index, row in file.iterrows():
    persons[row['Id']] = row['Name']


# 针对别名进行转换
def unify_name(name):
    # 姓名统一小写
    name = str(name).lower()
    # 去掉,和@后边的内容
    name = name.replace(',', '').split('@')[0]
    # 别名转换
    if name in aliases.keys():
        return persons[aliases[name]]
    return name


# 画网络图
def show_graph(graph, layout = 'spring_layout'):
    # 使用spring layout布局，类似中心放射状
    if layout == 'circular_layout':
        positions = nx.circular_layout(graph)
    else:
        positions = nx.spring_layout(graph)
    # 设置网络图中的节点大小，大小与pagerank值相关，由于pagerank值很小，所以需要*20000
    nodesize = [x['PageRank'] * 20000 for v, x in graph.nodes(data = True)]
    # 设置网络图中边的长度
    edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data = True)]
    # 绘制节点
    nx.draw_networkx_nodes(graph, positions, node_size = nodesize, alpha = 0.4)
    # 绘制边
    nx.draw_networkx_edges(graph, positions, edge_size = edgesize, alpha = 0.2)
    # 绘制节点的label
    nx.draw_networkx_labels(graph, positions, font_size = 10)
    # 输出邮件中的所有人物关系
    plt.show()


# 将发件人和收件人的姓名规范化
emails.MetadataFrom = emails.MetadataFrom.apply(unify_name)
emails.MetadataTo = emails.MetadataTo.apply(unify_name)
# 设置边的权重等于发邮件的次数
edges_weights_temp = defaultdict(lambda: 0)
for x, y in zip(emails.MetadataFrom, emails.MetadataTo):
    temp = (x, y)
    edges_weights_temp[temp] += 1
# 转化格式 (from, to), weight -> (from, to, weight)
edges_weights = [(key[0], key[1], value) for key, value in edges_weights_temp.items()]
# 创建有向图
graph = nx.DiGraph()
# 设置有向图中的路径及权重
graph.add_weighted_edges_from(edges_weights)
# 计算每个节点的PR值，并作为节点的pagerank属性
pagerank = nx.pagerank(graph)
# 将pagerank值作为节点的属性
nx.set_node_attributes(graph, values = pagerank, name = 'PageRank')
# 画网络图
show_graph(graph)

# 将完整的图谱进行精简
# 设置PR的阈值，筛选大于阈值的重要核心节点
pagerank_threshold = 0.005
# 复制一份计算好的网络图
small_graph = graph.copy()
# 剪掉PR值小于pagerank_threshold的节点
for n, pr in graph.nodes(data = True):
    if pr['PageRank'] < pagerank_threshold:
        small_graph.remove_node(n)
# 画网络图
show_graph(small_graph, 'circular_layout')
