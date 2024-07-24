import copy
import numpy as np
import pandas as pd

########## Export Data ##########
def multi_layer_ltm_for_seed(df_flow, layer_seq, thred, seed):
    """
    thred: shock_severity / buffer_capacity
    seed: [('Ore','CHN')]
    return: activated_node_sequence for a seed
    """

    shock_severity = 0.5
    buffer_capacity = shock_severity * thred # bufer < shock

    layer_country_nodes, adj_matrix = create_supra_adj_matrix(df_flow, layer_seq)
    activated_nodes = copy.deepcopy(seed)
    activated_nodes_sequence= diffuse_all_rounds(layer_country_nodes, adj_matrix, shock_severity, buffer_capacity, activated_nodes)
    activated_nodes_sequence = activated_nodes_sequence[:-1]

    return activated_nodes_sequence


def multi_layer_ltm_loop_seed(df_flow, layer_seq, thred):
    """
    return: loop all the node
    """

    shock_severity = 0.5
    buffer_capacity = shock_severity * thred # bufer < shock

    layer_country_nodes, adj_matrix = create_supra_adj_matrix(df_flow, layer_seq)
    df_affected = pd.DataFrame(0, columns=layer_country_nodes, index=layer_country_nodes)

    for layer_country in layer_country_nodes:
        activated_nodes_sequence = multi_layer_ltm_for_seed(df_flow, layer_seq, thred, [layer_country])

        activated_nodes = []
        for seq in activated_nodes_sequence:
            activated_nodes.extend(seq)
        activated_nodes = set(activated_nodes)

        for affected in activated_nodes:
            df_affected.at[layer_country, affected] = 1

    # from single index to multi index
    new_index = pd.MultiIndex.from_tuples(df_affected.index, names = ['Layer', 'Country'])
    df_affected = pd.DataFrame(df_affected, index=new_index, columns=new_index)

    # 行求和, Valanche_Size
    df_affected['Avalanche_Size'] = df_affected.apply(lambda x: x.sum(), axis=1)
    # 列求和, Exposure
    df_affected['Exposure'] = df_affected.apply(lambda x: x.sum(), axis=0)

    return df_affected


########## Basic Method ##########
def get_nodes(df_flow, layer_seq):
    """
    return: list of (layer, country)
    """

    layer_country_nodes = []

    for layer in layer_seq:
        # countries in Layer_From
        from_country = df_flow[df_flow.Layer_From==layer].Exporter.unique()
        # countries in Layer_To
        to_country = df_flow[df_flow.Layer_To==layer].Importer.unique()
        # country seq of this layer
        layer_country =  set(from_country).union(set(to_country))
        layer_country = list(sorted(layer_country))
        # layer-country seq of this layers
        layer_country = [(layer, country) for country in layer_country]

        layer_country_nodes.extend(layer_country)

    return layer_country_nodes


def create_supra_adj_matrix(df_flow, layer_seq):

    layer_country_nodes = get_nodes(df_flow, layer_seq)
    supra_adj_matrix = np.zeros((len(layer_country_nodes), len(layer_country_nodes)))

    ## Fill in flows
    for row in df_flow.itertuples():
        # 获取 row 在 matrix 中的位置
        idx_from = layer_country_nodes.index((row.Layer_From, row.Exporter))
        idx_to = layer_country_nodes.index((row.Layer_To, row.Importer))
        # 填充 adj_matrix
        supra_adj_matrix[idx_from, idx_to] = row.NetWeight

    return layer_country_nodes, supra_adj_matrix


def node_capacity_for_supply(adj_matrix):
    """
    supply capacity = sum of inflows to node
    """
    sum_inflows = adj_matrix.sum(axis=0)
    return sum_inflows


def node_capacity_for_demand(adj_matrix):
    """
    demand supply = sum of outflows from node
    """
    sum_outflows = adj_matrix.sum(axis=1)
    sum_outflows = sum_outflows.reshape(len(sum_outflows),1)
    return sum_outflows


def get_node_successors(layer_country_nodes, adj_matrix, node):
    """
    return: successor nodes (layer, country) of given node
    """
    successors_node = []

    idx_from = layer_country_nodes.index(node)
    outflows_from_node = adj_matrix[idx_from,:].tolist()

    for idx, outflow in enumerate(outflows_from_node):
        if outflow > 0:
            successors_node.append(layer_country_nodes[idx])

    return successors_node


def get_node_prodecessors(layer_country_nodes, adj_matrix, node):
    """
    return: prodecessor nodes (layer, country) of given node
    """
    prodecessors_node = []

    idx_to = layer_country_nodes.index(node)
    inflows_to_node = adj_matrix[:, idx_to].tolist()

    for idx, inflow in enumerate(inflows_to_node):
        if inflow > 0:
            prodecessors_node.append(layer_country_nodes[idx])

    return prodecessors_node


########## Diffuse ##########
def diffuse_all_rounds(layer_country_nodes, adj_matrix, shock_severity, buffer_capacity, activated_nodes):
    """
    node capacity from supply side: sum of inflows of each node
    node capacity from demand side: sum of outflows of each node
    return: activated_nodes_sequence: list of list of activated nodes at each diffuse stage
    """

    supply_capacity = node_capacity_for_supply(adj_matrix)
    demand_capacity = node_capacity_for_demand(adj_matrix)

    # 用于存储每轮感染的节点名称
    activated_nodes_sequence = []
    # 添加初始感染的seed
    activated_nodes_sequence.append([n for n in activated_nodes])
    while True:
        # t时刻的感染节点数
        len_activated_t = len(activated_nodes)
        # t+1时刻的感染节点数
        activated_nodes, activated_nodes_this_round = diffuse_one_round(layer_country_nodes, adj_matrix, supply_capacity, demand_capacity, shock_severity, buffer_capacity, activated_nodes)
        activated_nodes_sequence.append(activated_nodes_this_round)
        if len(activated_nodes) == len_activated_t:
            ## 当activated不再发生变化时，结束循环
            break
    # 返回每轮被感染的节点序列
    activated_nodes_sequence = activated_nodes_sequence[:-1]
    return activated_nodes_sequence

def diffuse_one_round(layer_country_nodes, adj_matrix, supply_capacity, demand_capacity, shock_severity, buffer_capacity, activated_nodes):
    """
    return: activated_nodes_this_round: list of activated nodes at i-th diffuse stage
    """

    activated_nodes_this_round = set()

    ## Supply Side
    for act_node in activated_nodes:
        # act_node指向的节点，也就是下一步准备激活的节点
        successors_node_supply = get_node_successors(layer_country_nodes, adj_matrix, act_node)
        for node in successors_node_supply:
            if node in activated_nodes:
                continue
            prodecessors_node_supply = get_node_prodecessors(layer_country_nodes, adj_matrix, node) # 指向 node 的节点
            activated_prodecessors_supply = list(set(prodecessors_node_supply).intersection(set(activated_nodes)))

            influence_sum = sum_of_influence_supply(layer_country_nodes, adj_matrix, shock_severity, activated_prodecessors_supply, node)
            node_capacity_supply = supply_capacity[layer_country_nodes.index(node)]

            if influence_sum >= buffer_capacity * node_capacity_supply:
                activated_nodes_this_round.add(node)

    ## Demand Side
    for act_node in activated_nodes:
        ## 如果是 waste 层被激活，不向上做 demand side 的影响传递
        if act_node[0] == 'Waste':
            continue
        # 指向act_node的节点，也就是下一步准备激活的节点，from node v to node u, node v
        prodecessors_node_demand = get_node_prodecessors(layer_country_nodes, adj_matrix, act_node)
        for node in prodecessors_node_demand:
            if node in activated_nodes:
                continue
            successors_node_demand = get_node_successors(layer_country_nodes, adj_matrix, node) # node v 指向的节点, from node v to node u, node u
            activated_successors_demand = list(set(successors_node_demand).intersection(set(activated_nodes))) # node v 指向的节点中被感染的节点

            influence_sum = sum_of_influence_demand(layer_country_nodes, adj_matrix, shock_severity, node, activated_successors_demand)
            node_capacity_demand = demand_capacity[layer_country_nodes.index(node)]

            if influence_sum >= buffer_capacity * node_capacity_demand:
                activated_nodes_this_round.add(node)

    activated_nodes.extend(list(activated_nodes_this_round))
    return activated_nodes, list(activated_nodes_this_round)


def sum_of_influence_supply(layer_country_nodes, adj_matrix, shock_severity, from_nodes, to_node):
    """
    return sum of influence affected by all affected nodes
    """

    influence_sum = 0.0
    from_nodes_idx = []

    for n in from_nodes:
        from_nodes_idx.append(layer_country_nodes.index((n[0],n[1])))

    to_node_idx = layer_country_nodes.index((to_node[0], to_node[1]))

    for from_idx in from_nodes_idx:
        influence_sum += adj_matrix[from_idx, to_node_idx]

    influence_sum *= shock_severity
    return influence_sum


def sum_of_influence_demand(layer_country_nodes, adj_matrix, shock_severity, from_node, to_nodes):
    """
    return sum of influence affected by all affected nodes
    """

    influence_sum = 0.0
    to_nodes_idx = []

    for n in to_nodes:
        to_nodes_idx.append(layer_country_nodes.index(n))

    from_node_idx = layer_country_nodes.index(from_node)

    for to_idx in to_nodes_idx:
        influence_sum += adj_matrix[from_node_idx, to_idx]

    influence_sum *= shock_severity
    return influence_sum
