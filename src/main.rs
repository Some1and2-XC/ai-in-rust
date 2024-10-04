mod lib;

use std::sync::Arc;

use crate::lib::node::{Node, NodeLayer, InputLayer};

fn main() {

    let layer: Arc<NodeLayer<f64>> = Arc::new(NodeLayer::new_random(5, 5, None));
    let input_nodes = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = InputLayer::new(input_nodes, Some(layer));

    let response = input.evaluate();

    println!("Outputs: {:?}", response);

}
