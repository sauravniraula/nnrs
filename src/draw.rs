use std::{fmt::Debug, io::Error};

use petgraph::{
    Graph,
    dot::{Config, Dot},
    graph::NodeIndex,
};

use crate::tensor::Tensor;

pub fn draw_computation_graph<T: Debug>(tensor: &Tensor<T>) -> Result<(), Error> {
    let mut graph = Graph::<String, String>::new();

    fn add_tensor<T: Debug>(graph: &mut Graph<String, String>, t: &Tensor<T>) -> NodeIndex {
        let mut nodes: Vec<NodeIndex> = vec![];
        let mut op_node: Option<NodeIndex> = None;
        if let Some(op) = t.get_op() {
            op.parents.iter().for_each(|p| {
                nodes.push(add_tensor(graph, p));
            });
            op_node = Some(graph.add_node(format!("{:?}", op.kind)));
        }

        let p_node = graph.add_node(format!("{:?}", t.get_data()));
        let c_node = if let Some(o) = op_node {
            graph.add_edge(o, p_node, "".into());
            o
        } else {
            p_node
        };
        for each_node in nodes {
            graph.add_edge(each_node, c_node, "".into());
        }
        p_node
    }

    add_tensor(&mut graph, tensor);

    let dot = format!(
        "{}",
        Dot::with_config(&graph, &[Config::RankDir(petgraph::dot::RankDir::LR)])
    );
    std::fs::write("outputs/graph.dot", dot)
}
