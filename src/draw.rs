use std::{collections::HashMap, io::Error};

use petgraph::{
    Graph,
    dot::{Config, Dot},
    graph::NodeIndex,
};

use crate::{number::Number, tensor::Tensor};

pub fn draw_computation_graph<T: Number>(tensor: &Tensor<T>) -> Result<(), Error> {
    let mut graph = Graph::<String, String>::new();
    let mut visited: HashMap<Tensor<T>, NodeIndex> = HashMap::new();

    fn add_tensor<T: Number>(
        visited: &mut HashMap<Tensor<T>, NodeIndex>,
        graph: &mut Graph<String, String>,
        t: &Tensor<T>,
    ) -> NodeIndex {
        if let Some(node) = visited.get(t) {
            return *node;
        }

        let node = graph.add_node(format!("{:?} | {:?}", t.get_data(), t.get_gradients()));
        visited.insert(t.clone(), node);

        if let Some(op) = t.get_op() {
            let op_node = graph.add_node(format!("{:?}", op.kind));
            graph.add_edge(op_node, node, "".into());

            for parent in &op.parents {
                let p_node = add_tensor(visited, graph, parent);
                graph.add_edge(p_node, op_node, "".into());
            }
        }
        node
    }

    add_tensor(&mut visited, &mut graph, tensor);

    let dot = format!(
        "{}",
        Dot::with_config(&graph, &[Config::RankDir(petgraph::dot::RankDir::LR)])
    );
    std::fs::write("outputs/graph.dot", dot)
}
